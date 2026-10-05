"""Hugging Face CLI install and login."""
import shutil
import subprocess
import sys
from huggingface_hub import get_token
from pathlib import Path

from .logs import log
from .setup import GLOBAL_ENV


def find_hf_cli():
    cli_dir = Path(GLOBAL_ENV["HF_HOME"]) / "cli" if "HF_HOME" in GLOBAL_ENV else Path.home() / ".hf-cli"
    candidates = [Path.home() / ".local" / "bin" / "hf", cli_dir / "venv" / "bin" / "hf", cli_dir / "venv" / "Scripts" / "hf.exe"]
    found = shutil.which("hf", path=GLOBAL_ENV.get("PATH"))
    return Path(found) if found else next((c for c in candidates if c.exists()), None)


def install_hf_cli():
    if sys.platform == "win32":
        cmd = ["powershell", "-ExecutionPolicy", "ByPass", "-c", "irm https://hf.co/cli/install.ps1 | iex"]
    else:
        cmd = ["bash", "-c", "curl -LsSf https://hf.co/cli/install.sh | bash"]
    subprocess.check_call(cmd, env=GLOBAL_ENV)


def ensure_hf_login():
    if get_token():
        log("Hugging Face login found, proceeding.")
        return
    if input("Log in to Hugging Face (needed for gated models like PixAI)? [y/N]: ").strip().lower() not in ("y", "yes"):
        return
    cli = find_hf_cli()
    if cli is None:
        if input("Hugging Face CLI not found. Install it globally? [y/N]: ").strip().lower() not in ("y", "yes"):
            return
        install_hf_cli()
        cli = find_hf_cli()
        if cli is None:
            log("Hugging Face CLI install not found, skipping login.")
            return
    subprocess.call([str(cli), "auth", "login"], env=GLOBAL_ENV)
