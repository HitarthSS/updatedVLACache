"""Load the released VLA-Cache implementation; retain all native action semantics."""
import os
os.environ.update(HF_HUB_OFFLINE='1', HF_HUB_DISABLE_IMPLICIT_TOKEN='1',
                  TRANSFORMERS_OFFLINE='1', WANDB_MODE='disabled', WANDB_DISABLED='true',
                  MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', TF_CPP_MIN_LOG_LEVEL='3',
                  OMP_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false')
import contextlib
import hashlib
import inspect
import io
import json
from pathlib import Path
import shutil
import sys
import time
import types

ROOT = Path(__file__).resolve().parent
os.environ['HF_HOME'] = str(ROOT / 'hf-cache')
os.environ['HF_MODULES_CACHE'] = str(ROOT / 'hf-modules')
# Checkpoint auto_map references the base repository's processor Python source.
# Populate an isolated offline snapshot with that exact checksum-pinned file.
processor_receipt = json.loads((ROOT / 'PROCESSOR_READY.json').read_text())
processor_cache = ROOT / 'hf-cache/hub/models--openvla--openvla-7b'
snapshot = processor_cache / 'snapshots' / processor_receipt['revision']
snapshot.mkdir(parents=True, exist_ok=True)
processor_source = (ROOT / 'canonical-processor/processing_prismatic.py').read_bytes()
assert hashlib.sha256(processor_source).hexdigest() == processor_receipt['sha256']
(snapshot / 'processing_prismatic.py').write_bytes(processor_source)
(processor_cache / 'refs').mkdir(exist_ok=True)
(processor_cache / 'refs/main').write_text(processor_receipt['revision'])
SRC = ROOT / 'upstream/src/openvla'
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT / 'transformers/src'))
sys.path.insert(0, str(ROOT / 'LIBERO'))
config_dir = ROOT / 'libero-config'
config_dir.mkdir(exist_ok=True)
libero_root = ROOT / 'LIBERO/libero/libero'
private_paths = {'benchmark_root': str(libero_root), 'bddl_files': str(libero_root / 'bddl_files'),
                 'init_states': str(libero_root / 'init_files'), 'assets': str(libero_root / 'assets'),
                 'datasets': str(ROOT / 'LIBERO/libero/datasets')}
(config_dir / 'config.yaml').write_text(json.dumps(private_paths))
os.environ['LIBERO_CONFIG_PATH'] = str(config_dir)
# The HF inference module is standalone. Skip the unrelated training registry's
# root __init__ (which imports RLDS training loaders), not any model operation.
namespace = types.ModuleType('prismatic')
namespace.__path__ = [str(SRC / 'prismatic')]
sys.modules['prismatic'] = namespace
# Official LIBERO uses a namespace top-level package; otherwise the old RPent
# dependency's regular package takes precedence despite sys.path ordering.
libero_namespace = types.ModuleType('libero')
libero_namespace.__path__ = [str(ROOT / 'LIBERO/libero')]
sys.modules['libero'] = libero_namespace
import numpy as np
import torch
torch.set_num_threads(2)
# Released initial-state banks contain NumPy arrays. Keep modern safe loading,
# explicitly allowing only the NumPy constructors/dtypes those banks require.
torch.serialization.add_safe_globals([np.core.multiarray._reconstruct, np.ndarray, np.dtype,
                                     type(np.dtype('float64')), type(np.dtype('float32'))])
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(2)
from experiments.robot import openvla_utils as native
from experiments.robot import robot_utils
from experiments.robot.libero import libero_utils
from my_harness import Controller


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


ORIGINAL_ACTION = native.get_vla_action
ORIGINAL_SELECTION = native.task_relevant_selection
SOURCE = inspect.getsource(ORIGINAL_ACTION)
# The release calls len(None) when no static tokens survive. In that case only,
# invalidate the old prompt and run the native full prediction. Common to arms.
needle = 'vla.language_model.config.proportion_attn_var = get_layer_mask_schedule(prev_attn)'
assert SOURCE.count(needle) == 1
GUARDED_SOURCE = SOURCE.replace(needle, '''vla.language_model.config.proportion_attn_var = get_layer_mask_schedule(prev_attn) if mask_indices is not None else None
            if mask_indices is None:
                prompt_cache = None''')
scope = {}
exec(compile(GUARDED_SOURCE, '<vla-cache-empty-mask-guard>', 'exec'), native.__dict__, scope)
GUARDED_ACTION = scope['get_vla_action']


class Policy:
    def __init__(self):
        torch.cuda.set_per_process_memory_fraction(.65)
        robot_utils.set_seed_everywhere(7)
        # Keep raw public configs/weights intact; native loader updates only this
        # runtime config/source copy. Weight files are read-only symlinks.
        model = ROOT / 'runtime-model'
        model.mkdir(exist_ok=True)
        for p in (ROOT / 'model').iterdir():
            dest = model / p.name
            if not dest.exists():
                if p.suffix == '.safetensors':
                    dest.symlink_to(p)
                else:
                    shutil.copy2(p, dest)
        self.cfg = types.SimpleNamespace(pretrained_checkpoint=str(model), model_family='openvla',
                       load_in_8bit=False, load_in_4bit=False, center_crop=True,
                       use_vla_cache=True, unnorm_key='libero_spatial')
        os.chdir(SRC)
        self.model = native.get_vla(self.cfg).eval().requires_grad_(False)
        self.processor = native.get_processor(self.cfg)
        assert sha(inspect.getfile(type(self.processor))) == processor_receipt['sha256']
        self.layers = self.model.language_model.model
        self.logical_tokens = 0
        self.decode_tokens = 0
        self.reused_token_layers = 0
        self.layer_reads = 0
        self.input_tokens = 0
        def layer_count(module, args, kwargs):
            hs = args[0] if args else kwargs['hidden_states']
            n = int(hs.shape[1])
            if n > 1:
                self.logical_tokens += n
                self.layer_reads += 1
            else:
                self.decode_tokens += n
        for layer in self.layers.layers:
            layer.register_forward_pre_hook(layer_count, with_kwargs=True)

    def start(self, arm):
        self.arm = arm
        self.controller = Controller(arm) if arm not in ('authors', 'bare') else None
        self.last_caches = None
        self.previous_image = None
        self.cfg.use_vla_cache = arm != 'bare'
        self.model.language_model.config.proportion_attn_var = None
        self.model.language_model.config.reusable_patches = None

    def act(self, image, position, fingers, instruction, original=False):
        obs = {'full_image': image, 'prev_image': image if self.previous_image is None else self.previous_image}
        if self.controller and self.controller.before(image, position, fingers):
            self.last_caches = None
        cropped = None
        self.last_empty_guard = False
        def select(attn, crop, stable):
            nonlocal cropped
            cropped = np.asarray(crop)
            overlay, indices = ORIGINAL_SELECTION(attn, crop, stable)
            if self.controller:
                indices = self.controller.filter_indices(crop, indices)
            if not indices:
                self.last_empty_guard = True
            return overlay, indices
        native.task_relevant_selection = select
        before = self.logical_tokens, self.decode_tokens, self.layers.all_FLOPs, self.layers.total_cuda_time
        start = time.monotonic()
        try:
            with torch.inference_mode(), contextlib.redirect_stdout(io.StringIO()):
                fn = ORIGINAL_ACTION if original else GUARDED_ACTION
                action, caches, _ = fn(self.cfg, self.model, self.processor,
                      self.cfg.pretrained_checkpoint, obs, instruction, self.cfg.unnorm_key,
                      center_crop=True, last_caches=self.last_caches)
            torch.cuda.synchronize()
        finally:
            native.task_relevant_selection = ORIGINAL_SELECTION
        elapsed = time.monotonic() - start
        self.last_caches = caches
        self.previous_image = np.array(image, copy=True)
        if cropped is None:
            cropped = np.asarray(native.process_image(__import__('PIL').Image.fromarray(image)))
        reused = self.model.language_model.config.proportion_attn_var is not None
        if self.controller:
            self.controller.after(action, cropped, reused)
        assert action.shape == (7,) and np.isfinite(action).all()
        stats = {'policy_seconds': elapsed, 'prefill_token_layers': self.logical_tokens - before[0],
                 'decode_token_layers': self.decode_tokens - before[1],
                 'decoder_flops': float(self.layers.all_FLOPs - before[2]),
                 'decoder_cuda_ms': float(self.layers.total_cuda_time - before[3]),
                 'empty_mask_full_refresh': self.last_empty_guard,
                 'reusable_tokens': 0 if not reused else int(len(self.model.language_model.config.reusable_patches)),
                 'input_tokens': int(caches['past_key_values']._seen_tokens),
                 'paid_api_calls': 0, 'external_tokens': 0}
        if self.controller:
            stats.update(self.controller.diagnostics())
        return action.copy(), stats

    def weights_sha(self):
        h = hashlib.sha256()
        for name, tensor in sorted(self.model.state_dict().items()):
            h.update(name.encode())
            h.update(str(tensor.dtype).encode())
            h.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        return h.hexdigest()


def install_renderer():
    import rendering
    rendering.install()


def check_camera(obs):
    import rendering
    image = np.asarray(obs['agentview_image'])
    assert image.shape == (256, 256, 3) and image.dtype == np.uint8
    metrics = rendering.health_metrics(image)
    if metrics['std'] < 1 / 255 or metrics['edge_to_variance'] > .8:
        raise RuntimeError('Invalid camera: ' + repr(metrics))
    return metrics
