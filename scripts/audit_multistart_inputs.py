"""Verify paper-era public artifacts against local inputs, without cloning data."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

BASE = Path('/pscratch/sd/y/yy6080/workspace/Ocean_Emulator_Workspace')
OUT = BASE / 'experiments/multistart_20261019'
PUBLIC = 'https://nyu1.osn.mghpcc.org/m2lines-pubs/FOMO/v2025-11'


def digest_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def fetch(url):
    with urlopen(url, timeout=60) as response:
        return response.read()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    hf = json.loads(fetch('https://huggingface.co/api/models/M2LInES/Samudra2/tree/main?recursive=true&expand=false'))
    remote_ckpts = {f['path']: f for f in hf if f['type'] == 'file'}
    result = {'public_data_release': PUBLIC, 'latest_v2026_09_not_used': True,
              'scope': 'Full checkpoint SHA256; full consolidated metadata; every normalization chunk; all coordinates and masks; selected physical-field chunks. Not a full multi-TB data hash.', 'resolutions': {}}
    for res in ('onedeg', 'halfdeg', 'quarterdeg'):
        ckpt = BASE / 'checkpoints/model_checkpoints_release' / f'ckpt_{res}.pt/ema_ckpt.pt'
        sha = digest_file(ckpt)
        assert sha == remote_ckpts[f'{res}/ema_ckpt.pt']['lfs']['oid'], res
        report = {'checkpoint': str(ckpt), 'checkpoint_sha256': sha,
                  'checkpoint_public_exact_match': True, 'data_root': str(BASE / 'datasets' / res), 'stores': {}}
        for store in ('OM4.zarr', 'OM4_means.zarr', 'OM4_stds.zarr'):
            root = BASE / 'datasets' / res / store
            url = f'{PUBLIC}/om4_{res}/{store}'
            local_meta = json.loads((root / '.zmetadata').read_text())
            remote_meta = json.loads(fetch(url + '/.zmetadata'))
            assert local_meta == remote_meta, (res, store, 'metadata')
            meta = local_meta['metadata']
            keys = []
            for name, arr in meta.items():
                if not name.endswith('/.zarray'):
                    continue
                var = name.removesuffix('/.zarray')
                dims = meta[var + '/.zattrs']['_ARRAY_DIMENSIONS']
                if store != 'OM4.zarr' or 'time' not in dims or var == 'time':
                    keys.extend(str(p.relative_to(root)) for p in (root / var).iterdir() if p.is_file() and not p.name.startswith('.'))
                elif var in ('thetao_0', 'thetao_15', 'so_0', 'so_15', 'uo_0', 'vo_0', 'zos', 'hfds', 'tauuo', 'tauvo'):
                    assert arr['chunks'][0] == 1
                    keys.extend(f'{var}/{i}.0.0' for i in (0, 4161, 4744))

            def check(key):
                local = (root / key).read_bytes()
                remote = fetch(url + '/' + key)
                assert local == remote, (res, store, key)
                return {'key': key, 'sha256': hashlib.sha256(local).hexdigest(), 'bytes': len(local)}

            with ThreadPoolExecutor(max_workers=8) as pool:
                matches = list(pool.map(check, keys))
            report['stores'][store] = {'metadata_exact_match': True, 'verified_chunks': matches}
            print(res, store, len(matches), 'chunks identical', flush=True)
        result['resolutions'][res] = report
        (OUT / 'input_audit.partial.json').write_text(json.dumps(result, indent=2))
    result['status'] = 'passed'
    (OUT / 'input_audit.json').write_text(json.dumps(result, indent=2))
    print('ALL INPUT AUDITS PASSED', flush=True)


if __name__ == '__main__':
    main()
