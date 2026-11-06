"""Prepare an isolated, public-weights-only VLA-Cache reproduction runtime."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import urllib.request

ROOT = Path(__file__).resolve().parent
REPO_REV = 'a4909880573868dee2769343d52e793c0341678b'
TF_REV = '2302fce58afa3a4f8461625b1394f9e9c8a7f1ea'
LIBERO_REV = '8f1084e3132a39270c3a13ebe37270a43ece2a01'
MODEL = 'openvla/openvla-7b-finetuned-libero-spatial'
MODEL_REV = '962318cec55ac10993ff0f5f43eda9a270b4c873'
PROCESSOR_REV = '47a0ec7fc4ec123775a391911046cf33cf9ed83f'
PROCESSOR_SHA = '06563586bbc73d36501a14a231a3174904ede830d33174d0c625ece8e931ef1e'


def atomic(name, value):
    p = ROOT / name
    t = p.with_suffix('.tmp')
    t.write_text(json.dumps(value, indent=2) + '\n')
    t.replace(p)


def run(args):
    subprocess.run(args, check=True)


def checkout(url, dest, revision, branch=None):
    p = ROOT / dest
    if not p.exists():
        args = ['git', 'clone', '--depth', '1']
        if branch:
            args += ['--branch', branch]
        run(args + [url, str(p)])
    current = subprocess.check_output(['git', '-C', str(p), 'rev-parse', 'HEAD'], text=True).strip()
    if current != revision:
        run(['git', '-C', str(p), 'fetch', '--depth', '1', 'origin', revision])
        run(['git', '-C', str(p), 'checkout', '--detach', revision])


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    atomic('SETUP_STATUS.json', {'state': 'sources', 'pid': os.getpid(), 'time': time.time()})
    checkout('https://github.com/siyuhsu/vla-cache.git', 'upstream', REPO_REV)
    checkout('https://github.com/siyuhsu/transformers.git', 'transformers', TF_REV, 'vla-cache-openvla')
    checkout('https://github.com/Lifelong-Robot-Learning/LIBERO.git', 'LIBERO', LIBERO_REV)
    shared = Path('/home/hitarth/rpent-reproduction/.venv')
    py = ROOT / '.venv/bin/python'
    if not py.exists():
        run([str(shared / 'bin/python'), '-m', 'venv', str(ROOT / '.venv')])
    # Read-only reuse of the large existing Torch/CUDA and simulator wheels.
    # All new or overridden dependencies are installed in this private environment.
    site = ROOT / '.venv/lib/python3.11/site-packages'
    (site / 'shared_readonly.pth').write_text(str(shared / 'lib/python3.11/site-packages') + '\n')
    atomic('SETUP_STATUS.json', {'state': 'dependencies', 'pid': os.getpid(), 'time': time.time()})
    run([str(py), '-m', 'pip', 'install', '--disable-pip-version-check',
         'numpy==1.26.4', 'tensorflow==2.15.1', 'tensorflow_datasets==4.9.3',
         'tensorflow_graphics==2021.12.3', 'tensorflow-metadata==1.15.0',
         'timm==0.9.10', 'draccus==0.8.0', 'tokenizers==0.21.1',
         'huggingface_hub==0.30.2', 'accelerate==1.5.2', 'peft==0.11.1',
         'scikit-image==0.25.0', 'json-numpy', 'jsonlines', 'rich', 'wandb',
         'sentencepiece==0.1.99', 'protobuf<5', 'imageio[ffmpeg]', 'bddl', 'easydict',
         'robosuite==1.4.1'])
    run([str(py), '-m', 'pip', 'install', '--no-deps', '-e', str(ROOT / 'transformers')])
    run([str(py), '-m', 'pip', 'install', '--no-deps', '-e', str(ROOT / 'LIBERO')])
    # Model files are fetched anonymously over HTTPS; no implicit account tokens.
    atomic('SETUP_STATUS.json', {'state': 'model_download', 'pid': os.getpid(), 'time': time.time()})
    url = f'https://huggingface.co/api/models/{MODEL}/tree/{MODEL_REV}?recursive=true'
    with urllib.request.urlopen(url, timeout=60) as response:
        files = [f for f in json.load(response) if f['type'] == 'file' and
                 (f['path'].endswith(('.json', '.safetensors', '.model')))]
    model = ROOT / 'model'
    model.mkdir(exist_ok=True)

    def download(item):
        p = model / item['path']
        p.parent.mkdir(parents=True, exist_ok=True)
        expected = item.get('lfs', {}).get('oid')
        if not p.exists() or p.stat().st_size != item['size']:
            with urllib.request.urlopen(
                f"https://huggingface.co/{MODEL}/resolve/{MODEL_REV}/{item['path']}", timeout=180
            ) as response, p.with_suffix(p.suffix + '.partial').open('wb') as out:
                while block := response.read(8 * 1024 * 1024):
                    out.write(block)
            p.with_suffix(p.suffix + '.partial').replace(p)
        assert p.stat().st_size == item['size'], str(p)
        digest = sha(p)
        if expected:
            assert digest == expected, (str(p), digest, expected)
        print('VERIFIED', p.name, p.stat().st_size, flush=True)
        return {'path': item['path'], 'size': p.stat().st_size, 'sha256': digest, 'lfs_verified': bool(expected)}

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        verified = list(pool.map(download, files))
    atomic('MODEL_READY.json', {'model': MODEL, 'revision': MODEL_REV, 'files': verified,
                                'anonymous_download': True, 'paid_api_calls': 0})
    processor = ROOT / 'canonical-processor'
    processor.mkdir(exist_ok=True)
    url = f'https://huggingface.co/openvla/openvla-7b/resolve/{PROCESSOR_REV}/processing_prismatic.py'
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read()
    assert hashlib.sha256(data).hexdigest() == PROCESSOR_SHA
    (processor / 'processing_prismatic.py').write_bytes(data)
    atomic('PROCESSOR_READY.json', {'repo': 'openvla/openvla-7b', 'revision': PROCESSOR_REV,
                'filename': 'processing_prismatic.py', 'sha256': PROCESSOR_SHA, 'anonymous_download': True})
    atomic('SETUP_STATUS.json', {'state': 'complete', 'time': time.time(), 'pid': os.getpid(),
                                 'repo_revision': REPO_REV, 'transformers_revision': TF_REV,
                                 'shared_dependencies': str(shared), 'private_environment': str(py)})


if __name__ == '__main__':
    try:
        main()
    except Exception:
        atomic('SETUP_ERROR.json', {'time': time.time(), 'traceback': traceback.format_exc()})
        raise
