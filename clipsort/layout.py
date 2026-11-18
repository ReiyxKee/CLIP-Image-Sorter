"""Keep a sorted folder's top-level folder names in line with clip_config.ini [folders].

Each sorted folder remembers the names it was sorted with in .folder_names.json. When the
settings change, the folders are renamed on disk and every record that stores sorted paths
(.sorted_index.jsonl, .multi_labels.json, suspicious.json) is rewritten to match, so the trainer
does not mistake the rename for hand-moved images.
"""
import json
import os
from pathlib import Path

from .config import INDEX_FILE, LAYOUT_FILE, MULTI_FILE, SESSION_FILE, SUS_FILE
from .logs import log
from .settings import CONFIG_FILE, DEFAULT_FOLDERS, FOLDERS

CATEGORY_KEYS = [k for k in FOLDERS if k != "r18"]


def load_layout(root):
    path = root / LAYOUT_FILE
    if path.exists():
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
            return {k: stored.get(k, DEFAULT_FOLDERS[k]) for k in DEFAULT_FOLDERS}
        except ValueError:
            log(f"Ignoring unreadable {path}")
    # sorted before folder names were configurable: those folders carry the old fixed names
    if any((root / name).is_dir() for name in DEFAULT_FOLDERS.values()):
        return dict(DEFAULT_FOLDERS)
    return None


def save_layout(root):
    root.mkdir(parents=True, exist_ok=True)
    (root / LAYOUT_FILE).write_text(json.dumps(FOLDERS, indent=1, ensure_ascii=False), encoding="utf-8")


def remap(rel, old, new):
    """Translate a sorted path (relative, '/'-separated) from the old folder names to the new ones."""
    parts = rel.split("/")
    i = 0
    if parts and parts[0] == old["r18"]:
        parts[0], i = new["r18"], 1
    if len(parts) > i:
        for key in CATEGORY_KEYS:
            if parts[i] == old[key]:
                parts[i] = new[key]
                break
    return "/".join(parts)


def remap_abs(path_text, root, old, new):
    path = Path(path_text)
    if root not in path.parents:
        return path_text
    return str(root / remap(path.relative_to(root).as_posix(), old, new))


def same_dir(a, b):
    try:
        return a.exists() and b.exists() and os.path.samefile(a, b)
    except OSError:
        return False


def plan_renames(root, old, new):
    """Folder moves in run order, R18 parent first: (path now, path when its turn comes, new path).
    Several categories can end up in one folder; the later ones are then merged into it."""
    plan, seen = [], set()
    r18_old, r18_new = root / old["r18"], root / new["r18"]
    if old["r18"] != new["r18"] and r18_old.is_dir():
        plan.append((r18_old, r18_old, r18_new))
    for key in CATEGORY_KEYS:
        if old[key] == new[key]:
            continue
        for parent_now, parent_then in ((root, root), (r18_old, r18_new)):
            src = parent_now / old[key]
            if src.is_dir() and (src, new[key]) not in seen:
                seen.add((src, new[key]))
                plan.append((src, parent_then / old[key], parent_then / new[key]))
    return plan


def split_folders(old, new):
    """Old folders shared by categories that now go to different folders (cannot be told apart)."""
    by_old = {}
    for key in CATEGORY_KEYS:
        by_old.setdefault(old[key], set()).add(new[key])
    return [name for name, targets in by_old.items() if len(targets) > 1]


def merge_problems(root, plan, old, new):
    """Targets that exist but are not part of the layout, and file names that would clash in a merge."""
    keeps = {(k, new[k]) for k in CATEGORY_KEYS if old[k] == new[k]}
    problems, groups = [], {}
    for src, _, dst in plan:
        groups.setdefault(dst, []).append(src)
    for dst, srcs in groups.items():
        is_category = dst.parent in (root, root / new["r18"]) and any(k for k, n in keeps if n == dst.name)
        if dst.exists() and not any(same_dir(s, dst) for s in srcs) and not is_category:
            problems.append(f"{dst} already exists")
            continue
        names = {}
        sources = ([dst] if dst.exists() and not any(same_dir(s, dst) for s in srcs) else []) + srcs
        for folder in sources:
            for child in folder.iterdir():
                other = names.setdefault(child.name.lower(), folder)
                if other != folder:
                    problems.append(f"{child.name} is in both {other.relative_to(root)} and {folder.relative_to(root)}")
    return problems


def move_into(src, dst):
    """Rename src to dst, or move its contents into dst when dst already exists (a merge)."""
    if not dst.exists() or same_dir(src, dst):
        src.rename(dst)
        return
    for child in list(src.iterdir()):
        child.rename(dst / child.name)
    src.rmdir()


def rewrite_records(root, old, new):
    index = root / INDEX_FILE
    if index.exists():
        lines = []
        for line in index.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                entry = json.loads(line)
            except ValueError:
                lines.append(line)
                continue
            entry["dest"] = remap(entry.get("dest", ""), old, new)
            if entry.get("source"):
                entry["source"] = remap_abs(entry["source"], root, old, new)
            lines.append(json.dumps(entry, ensure_ascii=False))
        index.write_text("".join(f"{l}\n" for l in lines), encoding="utf-8")
    multi = root / MULTI_FILE
    if multi.exists():
        data = json.loads(multi.read_text(encoding="utf-8"))
        multi.write_text(json.dumps({remap(k, old, new): v for k, v in data.items()}, indent=1, ensure_ascii=False), encoding="utf-8")
    if SUS_FILE.exists():
        entries = json.loads(SUS_FILE.read_text(encoding="utf-8"))
        for e in entries:
            e["path"] = remap_abs(e["path"], root, old, new)
        SUS_FILE.write_text(json.dumps(entries, indent=1, ensure_ascii=False), encoding="utf-8")


def check_layout(root, ask=True):
    """Call once the sorted folder is known, before anything reads or writes it."""
    if not root.is_dir():
        return
    old = load_layout(root)
    if old is None or old == FOLDERS:
        save_layout(root)
        return
    changes = {k: (old[k], FOLDERS[k]) for k in FOLDERS if old[k] != FOLDERS[k]}
    plan = plan_renames(root, old, FOLDERS)
    if not plan:
        save_layout(root)
        return
    log(f"Folder names in {CONFIG_FILE.name} changed: " + ", ".join(sorted({f"{a} -> {b}" for a, b in changes.values()})))
    if SESSION_FILE.exists() and json.loads(SESSION_FILE.read_text(encoding="utf-8")).get("root") == str(root):
        raise SystemExit("An unfinished correction session uses the old names. Put the old names back in "
                         f"{CONFIG_FILE.name} [folders], finish it with clip_sorter_train.py option 2, then change them again.")
    split = split_folders(old, FOLDERS)
    if split:
        raise SystemExit(f"Cannot split the shared folder(s) {', '.join(split)} into separate ones by renaming: "
                         "the images are not marked by category. Keep them shared, or re-sort with mode = recategorize.")
    problems = merge_problems(root, plan, old, FOLDERS)
    if problems:
        raise SystemExit("Cannot rename (nothing was changed):\n  " + "\n  ".join(problems[:20])
                         + (f"\n  ...and {len(problems) - 20} more" if len(problems) > 20 else ""))
    for src, _, dst in plan:
        log(f"  {src.relative_to(root)}  ->  {dst.relative_to(root)}{'  (merge)' if dst.exists() and not same_dir(src, dst) else ''}")
    if ask and input(f"Rename these {len(plan)} folders in {root}? [Y/n]: ").strip().lower() not in ("", "y", "yes"):
        raise SystemExit(f"Not renamed. Put the old names back in {CONFIG_FILE} [folders] to keep using this folder.")
    for _, src, dst in plan:
        move_into(src, dst)
    rewrite_records(root, old, FOLDERS)
    save_layout(root)
    log(f"Renamed {len(plan)} folders and updated the sorted records")
