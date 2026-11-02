"""Renew four-hour interactive allocations; fail loudly on scientific errors."""

import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    campaign = Path(sys.argv[1]).resolve()
    python = sys.argv[2]
    lock = (campaign / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (campaign / 'logs').mkdir(exist_ok=True)

    def status(state, **fields):
        content = {'status': state, 'controller_pid': os.getpid(),
                   'updated_at_unix': time.time(), **fields}
        tmp = campaign / 'controller.tmp'
        tmp.write_text(json.dumps(content, indent=2))
        tmp.replace(campaign / 'controller.json')
        print(json.dumps(content), flush=True)

    try:
        while True:
            states = {}
            for res in ('onedeg', 'halfdeg', 'quarterdeg'):
                path = campaign / 'runs' / res / 'status.json'
                states[res] = json.loads(path.read_text())['status'] if path.exists() else 'not_started'
            if all(s == 'complete' for s in states.values()):
                status('complete', rollouts=24)
                return
            status('allocating', resolutions=states)
            command = ['salloc', '--account=m4874_g', '--qos=interactive',
                       '--constraint=gpu&hbm80g', '--nodes=1', '--ntasks=4',
                       '--gpus=4', '--cpus-per-task=16', '--mem=220G',
                       '--time=04:00:00', '--immediate=86400', '--job-name=s2_multistart',
                       '/usr/bin/bash', str(campaign / 'runtime/multistart_node.sh'), str(campaign), python]
            result = subprocess.run(command)
            if result.returncode:
                raise RuntimeError(f'Allocation or worker failed (exit {result.returncode}); inspect logs before resuming.')
            status('allocation_finished')
    except BaseException as exc:
        status('failed', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
