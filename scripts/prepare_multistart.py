"""Freeze original paper configs and an explicit, equal-horizon start manifest."""

import json
import shutil
from pathlib import Path

import xarray as xr
import yaml
from audit_multistart_inputs import BASE, OUT

from ocean_emulators.config import EvalConfig

REFERENCES = {
    'onedeg': '2026-01-30-samudra_om4_onedeg_20_cap_eval',
    'halfdeg': '2026-02-05-samudra_om4_halfdeg_20_cap_fix_eval',
    'quarterdeg': '2025-02-08-samudra_om4_quarterdeg_cap_20_eval_fix',
}


def main():
    audit = json.loads((OUT / 'input_audit.json').read_text())
    assert audit['status'] == 'passed'
    (OUT / 'configs').mkdir(exist_ok=True)
    manifest = {'status': 'prepared', 'model': 'Samudra 2 only',
                'forcing': 'original historical; no interventions',
                'initialization': 'two successive real OM4 states; no perturbation',
                'calls_per_rollout': 182, 'predicted_frames': 364,
                'horizon': 'approximately five years, 364 five-day output frames; fixed call count for all starts',
                'statistical_caveat': 'Overlapping trajectories are not independent realizations.',
                'resolutions': {}}
    common_times = None
    for res, reference in REFERENCES.items():
        files = list((BASE / 'checkpoints' / reference / 'wandb').glob('run-*/files/config.yaml'))
        assert len(files) == 1
        original = yaml.safe_load(files[0].read_text())['config']['value']
        cfg = original.copy()
        cfg['data'] = dict(original['data'])
        cfg['data'].update(cfg['data'].pop('sources')[0])
        cfg['ckpt_path'] = audit['resolutions'][res]['checkpoint']
        cfg['experiment'] = dict(original['experiment'])
        cfg['experiment'].pop('train_schedule', None)  # Obsolete training-only field.
        cfg['experiment'].update(data_root=str(BASE / 'datasets' / res),
                                 base_output_dir=str(OUT / 'runs'), name=f'{res}_setup')
        cfg['experiment']['wandb'] = {'mode': 'disabled'}
        cfg['forcing_mode'] = 'student'
        cfg['model'] = dict(original['model'])
        cfg['model']['unet'] = dict(original['model']['unet'])
        assert cfg['model']['unet']['up_sampling_block'] == 'zonally_periodic_upsample'
        # The same historical option name changed implementation between branches.
        # Select the exact cell-centered interpolation of the reference commits.
        cfg['model']['unet']['up_sampling_block'] = 'cell_centered_periodic_upsample'
        # Preserve the original normalization, model precision and chunk lengths.
        EvalConfig.model_validate(cfg)
        (OUT / 'configs' / f'{res}.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
        (OUT / 'configs' / f'{res}_original.yaml').write_text(yaml.safe_dump(original, sort_keys=False))
        data = xr.open_zarr(BASE / 'datasets' / res / 'OM4.zarr', chunks=None)
        times = data.time.values
        if common_times is None:
            common_times = times
        assert (times == common_times).all()
        starts = []
        for year in (2015, 2017):
            for month in (1, 4, 7, 10):
                index = next(i for i, t in enumerate(times) if t.year == year and t.month == month)
                window = times[index:index + 366]
                assert len(window) == 366
                starts.append({'id': f'{year}{month:02}', 'source_start_index': index,
                               'initial_times': [str(t) for t in window[:2]],
                               'first_prediction': str(window[2]), 'last_prediction': str(window[-1]),
                               'last_lead_days': (window[-1] - window[1]).total_seconds() / 86400})
        manifest['resolutions'][res] = {
            'data_root': str(BASE / 'datasets' / res), 'checkpoint': cfg['ckpt_path'],
            'checkpoint_sha256': audit['resolutions'][res]['checkpoint_sha256'],
            'reference_predictions': str(BASE / 'checkpoints' / reference / 'predictions.zarr'),
            'config': str(OUT / 'configs' / f'{res}.yaml'),
            'norm': cfg['model']['unet']['core_block']['norm'],
            'bfloat16': cfg['model']['use_bfloat16'], 'starts': starts}
        manifest['resolutions'][res]['upsampling_provenance'] = {
            'implementation': 'cell_centered_periodic_upsample',
            'original_config_name': 'zonally_periodic_upsample',
            'reference_commit': json.loads(files[0].with_name('wandb-metadata.json').read_text())['git']['commit'],
            'verification': 'first four physical prediction frames against archived reference',
        }
        data.close()
    manifest['total_rollouts'] = 24
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    repo = Path(__file__).resolve().parents[1]
    shutil.copytree(repo / 'src/ocean_emulators', OUT / 'source/ocean_emulators',
                    ignore=shutil.ignore_patterns('__pycache__'), dirs_exist_ok=True)
    (OUT / 'runtime').mkdir(exist_ok=True)
    for name in ('multistart_worker.py', 'multistart_controller.py', 'multistart_node.sh'):
        shutil.copy2(repo / 'scripts' / name, OUT / 'runtime' / name)
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
