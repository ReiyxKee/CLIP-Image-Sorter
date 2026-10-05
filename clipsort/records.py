"""Index, suspicious list, aliases and series corrections."""
import json
from pathlib import Path

from .config import ALIASES_FILE, ANIME, INDEX_FILE, OVERRIDES_FILE, SUS_FILE
from .files import collect_images
from .names import legalize


def load_aliases():
    return json.loads(ALIASES_FILE.read_text(encoding="utf-8")) if ALIASES_FILE.exists() else {}


def save_aliases(aliases):
    ALIASES_FILE.parent.mkdir(parents=True, exist_ok=True)
    ALIASES_FILE.write_text(json.dumps(aliases, indent=1, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def load_overrides():
    return json.loads(OVERRIDES_FILE.read_text(encoding="utf-8")) if OVERRIDES_FILE.exists() else {}


def save_overrides(overrides):
    OVERRIDES_FILE.parent.mkdir(parents=True, exist_ok=True)
    OVERRIDES_FILE.write_text(json.dumps(overrides, indent=1, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def load_index(root):
    index, path = {}, root / INDEX_FILE
    if not path.exists():
        return index
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        source = Path(entry.get("source") or "")
        if root in source.parents:
            index.pop(source.relative_to(root).as_posix(), None)
        index[entry["dest"]] = entry
    return index


def save_index(root, index):
    lines = (json.dumps(e, ensure_ascii=False) for e in index.values())
    (root / INDEX_FILE).write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def append_index(root, dest, source, label):
    entry = {"dest": dest.relative_to(root).as_posix(), "source": str(source), "title": label[0], "char": label[1]}
    with (root / INDEX_FILE).open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_sus():
    return json.loads(SUS_FILE.read_text(encoding="utf-8")) if SUS_FILE.exists() else []


def save_sus(entries):
    SUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SUS_FILE.write_text(json.dumps(entries, indent=1, ensure_ascii=False), encoding="utf-8")


def labeled_images(root):
    items = []
    for p in collect_images(root, root):
        parts = p.relative_to(root).parts
        if ANIME in parts:
            k = parts.index(ANIME)
            if len(parts) == k + 4:
                items.append((p, (legalize(parts[k + 1]).lower(), legalize(parts[k + 2]).lower())))
    if not items:
        raise SystemExit(f"No images in {ANIME}/<Title>/<Character>/ folders under {root}")
    return items
