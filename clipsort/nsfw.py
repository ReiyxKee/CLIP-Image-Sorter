"""R18 check."""
from PIL import Image

from .logs import progress


def detect_nsfw(clf, paths, batch_size, threshold):
    flags = []
    for n, res in enumerate(clf((Image.open(p).convert("RGB") for p in paths), batch_size=batch_size), 1):
        flags.append(next(r["score"] for r in res if r["label"] == "nsfw") >= threshold)
        progress("Checking R18", n, len(paths))
    return flags
