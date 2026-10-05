"""Dependency check, CUDA torch and Hugging Face paths. Runs before any model library is imported."""
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEPENDENCIES = {
    "torch": "torch",
    "transformers": "transformers",
    "timm": "timm",
    "PIL": "pillow",
    "pillow_heif": "pillow-heif",
    "sentencepiece": "sentencepiece",
}
TORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu128"


def install_cuda_torch(*extra):
    subprocess.check_call([sys.executable, "-m", "pip", "install", *extra, "torch", "torchvision", "--index-url", TORCH_CUDA_INDEX])


def ensure_dependencies():
    missing = [pkg for mod, pkg in DEPENDENCIES.items() if importlib.util.find_spec(mod) is None]
    nvidia = shutil.which("nvidia-smi") is not None
    if missing:
        answer = input(f"Missing packages: {' '.join(missing)}. Install now? [y/N]: ").strip().lower()
        if answer not in ("y", "yes"):
            raise SystemExit(f"Install manually: {sys.executable} -m pip install -U {' '.join(missing)}")
        if nvidia and "torch" in missing:
            install_cuda_torch()
            missing.remove("torch")
        if missing:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", *missing])
    if not nvidia:
        return
    import torch
    if torch.version.cuda:
        return
    answer = input("NVIDIA GPU found but torch is CPU-only. Reinstall torch with CUDA? [y/N]: ").strip().lower()
    if answer in ("y", "yes"):
        install_cuda_torch("--force-reinstall", "--no-deps")
        raise SystemExit(subprocess.call([sys.executable, *sys.argv]))


sys.stdout.reconfigure(errors="replace")
ensure_dependencies()
HF_GLOBAL_HOME = Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")
os.environ.setdefault("HF_TOKEN_PATH", str(HF_GLOBAL_HOME / "token"))
GLOBAL_ENV = dict(os.environ)
os.environ.setdefault("HF_HOME", str(PROJECT_DIR / "models"))
