"""Image listing, moving and folder cleanup."""
import shutil

from .config import EXTENSIONS, IGNORE_MARKER, RENAME_FILES
from .names import legalize


def collect_images(target, output):
    """Images under target, skipping the output folder (unless it is the target) and any folder
    holding a .clip_ignore marker, such as the duplicates folder."""
    if target.is_file():
        return [target]
    ignored = {m.parent for m in target.rglob(IGNORE_MARKER)}
    return sorted(
        p for p in target.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTENSIONS and (target == output or output not in p.parents)
        and not ignored.intersection(p.parents)
    )


def original_stem(path):
    prefix = f"{legalize(path.parent.name).lower()}_"
    return path.stem[len(prefix):] if path.stem.startswith(prefix) else path.stem


def sorted_stem(path, char, in_sorted_folder=True):
    """File name (without extension) for an image filed under character char ([files] rename).
    Renaming gives <char>_<first 10 letters of the original name>; with it off the name is kept."""
    if not RENAME_FILES:
        return path.stem
    original = original_stem(path) if in_sorted_folder else path.stem
    return f"{char}_{original[:10]}" if char else original


def remove_empty_dirs(root):
    for d in sorted((d for d in root.rglob("*") if d.is_dir()), key=lambda d: len(d.parts), reverse=True):
        if not any(d.iterdir()):
            d.rmdir()


def transfer_unique(src, dest_dir, mode, stem):
    dest = dest_dir / f"{stem}{src.suffix}"
    n = 1
    while dest.exists() and dest != src:
        dest = dest_dir / f"{stem}_{n}{src.suffix}"
        n += 1
    if dest == src:
        return src
    dest_dir.mkdir(parents=True, exist_ok=True)
    (shutil.copy2 if mode == "copy" else shutil.move)(src, dest)
    return dest
