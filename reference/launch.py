"""Finite detached supervisor; stop on any failure instead of replacing cases."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

ROOT = Path(__file__).resolve().parent


def save(name, value):
    p = ROOT / name
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(p)


def main():
    lock = (ROOT / 'supervisor.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert json.loads((ROOT / 'PREFLIGHT.json').read_text())['status'] == 'PASS'
    assert not (ROOT / 'development').exists(), 'An existing cohort needs inspection; never overwrite it'
    assert not (ROOT / 'evaluation').exists(), 'An existing cohort needs inspection; never overwrite it'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', MUJOCO_GL='egl',
               PYOPENGL_PLATFORM='egl', PYTHONHASHSEED='7')
    for key in list(env):
        if key.endswith(('API_KEY', 'ACCESS_TOKEN')) or key in ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN'):
            env.pop(key)
    save('SUPERVISOR_START.json', {'pid': os.getpid(), 'time': time.time(), 'gpu': 0,
         'sequence': ['verified_preflight', '50_development_episodes', 'freeze_selected_design', '360_evaluation_episodes'],
         'final_matched_cases': 120, 'paid_api_calls': 0})
    with (ROOT / 'run.log').open('ab') as log:
        child = subprocess.Popen([str(ROOT / '.venv/bin/python'), '-u', str(ROOT / 'experiment.py')],
                   cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        save('CHILD.json', {'pid': child.pid, 'supervisor_pid': os.getpid(), 'time': time.time()})
        code = child.wait()
    if code:
        raise RuntimeError(f'Experiment exited {code}; inspect LAST_ERROR.json and run.log. No retry/replacement.')
    status = json.loads((ROOT / 'status.json').read_text())
    assert status['state'] == 'complete'
    save('SUPERVISOR_COMPLETE.json', {'time': time.time(), 'status': status})


if __name__ == '__main__':
    try:
        main()
    except Exception:
        save('SUPERVISOR_ERROR.json', {'time': time.time(), 'traceback': traceback.format_exc()})
        raise
