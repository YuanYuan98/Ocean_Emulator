"""Paper-config multi-start rollouts with atomic, resumable prediction shards.

Each committed shard contains physical fields and the exact normalized two-frame
autoregressive state. Resuming never substitutes ground truth or renormalizes a
physical output. All numerical work is launched on a Slurm compute node.
"""

import argparse
import dataclasses
import hashlib
import json
import logging
import os
import shutil
import time
from pathlib import Path

import dask
import numpy as np
import torch
import xarray as xr
import yaml

from ocean_emulators.backend import init_eval_backend
from ocean_emulators.config import EvalConfig, TimeConfig
from ocean_emulators.constants import BOUNDARY_VARS, PROGNOSTIC_VARS, TensorMap
from ocean_emulators.datasets import InferenceDataset
from ocean_emulators.utils.data import DataSource, Normalize, spherical_area_weights
from ocean_emulators.utils.distributed import set_seed
from ocean_emulators.utils.writer import ZarrWriter


def atomic_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def make_dataset(src, prog, bound):
    return InferenceDataset(src=src, prognostic_var_names=prog,
                            boundary_var_names=bound, hist=1,
                            normalize_before_mask=True, masked_fill_value=0.0,
                            long_rollout=True)


@torch.no_grad()
def preflight(model, source, cfg, prog, bound, normalize, spec, output):
    """Check the original published trajectory and exact normalized-state restart."""
    dataset = make_dataset(source.slice(TimeConfig(start='2014-10-10', end='2014-11-04')), prog, bound)
    initial = dataset.initial_prognostic
    whole = model.inference(dataset, initial, num_steps=2)
    first = model.inference(dataset, initial, num_steps=1)
    state_path = output / 'restart_test.pt'
    torch.save(first.prediction[-1:].cpu(), state_path)
    restored = torch.load(state_path, map_location='cpu', weights_only=True)
    second = model.inference(dataset, restored, steps_completed=1, num_steps=1)
    torch.testing.assert_close(second.prediction, whole.prediction[1:], rtol=1e-5, atol=2e-6)
    reference = xr.open_zarr(spec['reference_predictions'], chunks=None)
    flat = whole.prediction.reshape(4, 77, *whole.prediction.shape[-2:])
    physical = normalize.unnormalize_tensor_prognostic(flat, fill_value=0.0).cpu().numpy()
    errors = {}
    for i, var in enumerate(prog):
        ref = reference[var].sel(time=whole.time.values).values
        wet = source.masks.prognostic[i].numpy()
        diff = (physical[:, i, wet] - ref[:, wet]) / float(source.stds[var])
        errors[var] = float(np.sqrt(np.mean(diff.astype(np.float64) ** 2)))
    worst = max(errors.values())
    tolerance = 0.02 if cfg.model.use_bfloat16 else 0.001
    report = {'checkpoint_sha256': spec['checkpoint_sha256'],
              'reference': spec['reference_predictions'], 'frames_checked': 4,
              'normalized_rmse_by_channel': errors, 'maximum_channel_nrmse': worst,
              'tolerance': tolerance, 'serialized_restart_matches_continuous': True,
              'status': 'passed' if np.isfinite(worst) and worst < tolerance else 'failed'}
    atomic_json(output / 'preflight.json', report)
    if report['status'] != 'passed':
        raise RuntimeError(f'Published-reference preflight failed: {worst} >= {tolerance}')
    reference.close()
    state_path.unlink()
    print('PREFLIGHT PASSED', worst, flush=True)


@torch.no_grad()
def run(args):
    campaign = Path(args.campaign)
    manifest_text = (campaign / 'manifest.json').read_text()
    signature = hashlib.sha256(manifest_text.encode()).hexdigest()
    manifest = json.loads(manifest_text)
    spec = manifest['resolutions'][args.resolution]
    output = campaign / 'runs' / args.resolution
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / 'status.json'
    atomic_json(status_path, {'status': 'loading', 'job_id': os.environ.get('SLURM_JOB_ID')})
    cfg = EvalConfig.model_validate(yaml.safe_load(Path(spec['config']).read_text()))
    device = init_eval_backend('cuda')
    set_seed(cfg.experiment.rand_seed)
    torch.set_num_threads(4)
    dask.config.set(scheduler='threads', num_workers=4)
    prog = PROGNOSTIC_VARS[cfg.experiment.prognostic_vars_key]
    bound = BOUNDARY_VARS[cfg.experiment.boundary_vars_key]
    TensorMap.init_instance(cfg.experiment.prognostic_vars_key, cfg.experiment.boundary_vars_key)
    root = cfg.experiment.resolved_data_root
    # This is the exact dask source used by Eval; avoid constructing an unused
    # duplicate non-dask source, which repeats full-record heat-flux preprocessing.
    source = DataSource.from_locations(
        data_location=root.resolve(cfg.data.data_location),
        means_location=root.resolve(cfg.data.data_means_location),
        stds_location=root.resolve(cfg.data.data_stds_location),
        prognostic_var_names=prog, boundary_var_names=bound,
        static_data_vars=cfg.data.static_data_vars, use_dask=True)
    normalize = Normalize.init_instance(source, prognostic_var_names=prog, boundary_var_names=bound)
    model = cfg.model.build(
        in_channels=162, out_channels=154, hist=1,
        wet=source.masks.prognostic_with_hist(1).to(device),
        area_weights=spherical_area_weights(source.data).to(device),
        static_data=source.data[cfg.data.static_data_vars],
        lat=torch.from_numpy(source.data.lat.values), lon=torch.from_numpy(source.data.lon.values)).to(device)
    checkpoint = torch.load(cfg.ckpt_path, map_location='cpu', weights_only=False, mmap=True)
    assert checkpoint['epoch'] == 70
    model.load_state_dict({k.removeprefix('module.'): v for k, v in checkpoint['model'].items()}, strict=True)
    del checkpoint
    model.eval()
    if not (output / 'preflight.json').exists():
        preflight(model, source, cfg, prog, bound, normalize, spec, output)
    assert json.loads((output / 'preflight.json').read_text())['status'] == 'passed'
    if args.preflight_only:
        return
    total = manifest['calls_per_rollout']
    chunk_size = cfg.num_model_steps_forward
    for start in spec['starts']:
        target = output / start['id']
        target.mkdir(exist_ok=True)
        if (target / 'complete.json').exists():
            continue
        idx = start['source_start_index']
        selected = dataclasses.replace(source, name=f'{args.resolution}_{start["id"]}',
                                       data=source.data.isel(time=slice(idx, idx + 366)))
        dataset = make_dataset(selected, prog, bound)
        assert len(dataset) == total
        assert str(dataset.get_target_time(0, total).values[-1]) == start['last_prediction']
        step = 0
        initial = None
        shards = []
        for shard in sorted(target.glob('segment_*')):
            meta = json.loads((shard / 'complete.json').read_text())
            assert meta['manifest_sha256'] == signature and meta['start_step'] == step
            step = meta['end_step']
            shards.append(str(shard / 'predictions.zarr'))
        if shards:
            initial = torch.load(Path(shards[-1]).parent / 'state.pt', map_location='cpu', weights_only=True)
        else:
            initial = dataset.initial_prognostic
        while step < total:
            if time.time() > args.deadline - 300:
                atomic_json(status_path, {'status': 'paused', 'start': start['id'], 'completed_calls': step})
                return
            count = min(chunk_size, total - step)
            atomic_json(status_path, {'status': 'running', 'start': start['id'],
                                     'completed_calls': step, 'total_calls': total,
                                     'job_id': os.environ.get('SLURM_JOB_ID'),
                                     'updated_at_unix': time.time()})
            io = model.inference(dataset, initial, steps_completed=step, num_steps=count, forcing_mode='student')
            if not torch.isfinite(io.prediction).all():
                raise FloatingPointError(f'Nonfinite output: {args.resolution} {start["id"]} step {step}')
            pending = target / f'.pending_{step:04d}'
            if pending.exists():
                shutil.rmtree(pending)
            pending.mkdir()
            writer = ZarrWriter(pending, coords=dataset.get_coords_dict(), hist=1,
                                model_path=cfg.ckpt_path, time_chunk_size=count)
            writer.record_batch(io)
            writer.write()
            # Save exactly what the original Stepper uses as next chunk input.
            initial = io.prediction[-1:].detach().cpu().clone()
            torch.save(initial, pending / 'state.pt')
            check = xr.open_zarr(pending / 'predictions.zarr', chunks=None)
            assert check.sizes['time'] == 2 * count
            assert (check.time.values == io.time.values).all()
            check.close()
            atomic_json(pending / 'complete.json', {'manifest_sha256': signature,
                        'start_step': step, 'end_step': step + count,
                        'first_time': str(io.time.values[0]), 'last_time': str(io.time.values[-1]),
                        'finite': True})
            committed = target / f'segment_{step:04d}'
            pending.rename(committed)
            shards.append(str(committed / 'predictions.zarr'))
            step += count
            del io, writer
            print(args.resolution, start['id'], step, '/', total, flush=True)
        atomic_json(target / 'complete.json', {'status': 'complete', 'manifest_sha256': signature,
                    'predicted_frames': 2 * step, 'shards': shards, **start})
    atomic_json(status_path, {'status': 'complete', 'rollouts': len(spec['starts'])})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('campaign')
    p.add_argument('resolution', choices=['onedeg', 'halfdeg', 'quarterdeg'])
    p.add_argument('--deadline', type=float, required=True)
    p.add_argument('--preflight-only', action='store_true')
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    try:
        run(args)
    except BaseException as exc:
        target = Path(args.campaign) / 'runs' / args.resolution
        target.mkdir(parents=True, exist_ok=True)
        atomic_json(target / 'status.json', {'status': 'failed', 'error': repr(exc)})
        raise


if __name__ == '__main__':
    main()
