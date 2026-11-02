"""Wait for completed rollouts, then reduce and plot on four-hour allocations.

This controller never publishes unreviewed results to Google Docs. It leaves a
figure manifest and executed notebook for the final scientific/visual review.
"""

import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    output=Path(sys.argv[1]).resolve()
    python=sys.argv[2]
    output.mkdir(parents=True,exist_ok=True)
    lock=(output/'controller.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    campaign=Path('/pscratch/sd/y/yy6080/workspace/Ocean_Emulator_Workspace/experiments')
    statuses={
        'samudra2':None,
        'adaloss':campaign/'loss_revision_20261003/controllers/adaloss.json',
        'frozen_final':campaign/'loss_revision_20261003/controllers/frozen_final.json',
        'tendency':campaign/'loss_revision_20261003/controllers/tendency.json',
        'neptune':campaign/'neptune_13m_om4_20261014/controllers/neptune.json',
    }
    def status(state,**details):
        content={'status':state,'updated_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'pid':os.getpid(),**details}
        temporary=output/'status.tmp';temporary.write_text(json.dumps(content,indent=2));temporary.replace(output/'status.json')
        print(json.dumps(content),flush=True)
    try:
        while True:
            remaining=[key for key in statuses if not (output/key/'complete.json').exists()]
            if not remaining:
                status('figures_ready_pending_review',next_step='Inspect scientific results and figures, then create the Google Doc through the connected tool.')
                return
            available=[key for key in remaining if statuses[key] is None or (statuses[key].exists() and json.loads(statuses[key].read_text())['status']=='complete')]
            if not available:
                failed=[key for key in remaining if statuses[key] and statuses[key].exists() and json.loads(statuses[key].read_text())['status']=='failed']
                if failed:
                    raise RuntimeError(f'Upstream inference controllers failed: {failed}')
                status('waiting_for_inference',models=remaining)
                time.sleep(60)
                continue
            # The site's shared_interactive QOS permits only two submitted jobs.
            # Preserve the training/inference requests already occupying them.
            queued=subprocess.check_output(['squeue','--noheader','--user='+os.environ['USER'],'--format=%q'],text=True)
            if sum(q.strip()=='shared_interactive' for q in queued.splitlines()) >= 2:
                status('waiting_for_shared_interactive_slot',models=available)
                time.sleep(60)
                continue
            status('allocating_diagnostics',models=available)
            command=['salloc','--account=m4874_g','--qos=shared_interactive','--constraint=gpu&hbm80g',
                     '--nodes=1','--ntasks=1','--gpus=1','--cpus-per-task=32','--mem=55G',
                     '--time=04:00:00','--immediate=86400','--job-name=s2_baseline_metrics',
                     'srun','--kill-on-bad-exit=1','--ntasks=1','--gpus-per-task=1','--cpu-bind=cores',
                     '/usr/bin/bash',str(output/'runtime/revision_comparison_node.sh'),str(output),python,*available]
            result=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            print(result.stdout,flush=True)
            if result.returncode and 'QOSMaxSubmitJobPerUserLimit' in result.stdout:
                status('waiting_for_shared_interactive_slot',models=available)
                time.sleep(60)
                continue
            result.check_returncode()
    except BaseException as exc:
        status('failed',error=str(exc))
        raise


if __name__=='__main__':
    main()
