"""Console output and per-session log files."""
import sys
import traceback
from datetime import datetime
from pathlib import Path

from .setup import PROJECT_DIR


LOG_FILE = PROJECT_DIR / "Log" / f"{Path(sys.argv[0]).stem}_{datetime.now():%Y%m%d_%H%M%S}.log"


def write_log(text):
    try:
        LOG_FILE.parent.mkdir(exist_ok=True)
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(text + "\n")
    except OSError:
        pass


def log_exception(kind, value, tb):
    write_log("".join(traceback.format_exception(kind, value, tb)))
    sys.__excepthook__(kind, value, tb)


sys.excepthook = log_exception


write_log(f"Command: {' '.join(sys.argv)}")


def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line)
    write_log(line)


def progress(stage, done, total):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {stage} {done}/{total} ({done * 100 // max(total, 1)}%)"
    print(f"\r{line}", end="\n" if done >= total else "", flush=True)
    if done >= total:
        write_log(line)
