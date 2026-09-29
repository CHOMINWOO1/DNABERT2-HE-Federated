"""Record software and local model/tokenizer source hashes without changing the run."""
import sys, platform, subprocess, importlib.metadata
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import core as c
import torch

packages = {}
for dist in importlib.metadata.distributions():
    name = dist.metadata.get('Name')
    if name:
        packages[name] = dist.version
sources = {}
for path in sorted(c.SNAP.iterdir()):
    if path.is_file() and path.suffix in ('.json', '.txt', '.py', '.model'):
        sources[path.name] = c.sha(path)
driver = subprocess.run(
    ['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'],
    capture_output=True, text=True, check=True).stdout.strip()
c.dump('runtime_inventory.json', {
    'recorded_after_execution_started': True,
    'python_executable': sys.executable,
    'python_version': sys.version,
    'platform': platform.platform(),
    'torch_cuda_runtime': torch.version.cuda,
    'cudnn_version': torch.backends.cudnn.version(),
    'gpu_driver': driver,
    'installed_packages': dict(sorted(packages.items())),
    'snapshot_auxiliary_sources_sha256': sources,
    'pretrained_weight_and_dataset_hashes': c.read(c.OUT/'environment.json')['dependencies'],
    'note': 'Read-only post-run inventory; training and decoding precision settings are in the frozen protocol and prediction metadata.'
})
c.event(event='runtime_inventory_saved', packages=len(packages), model_source_files=len(sources))
