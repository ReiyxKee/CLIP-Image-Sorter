"""Fix-wrong-category stages and recovery."""
import json
import re
import torch
from pathlib import PurePosixPath

from .config import ANIME, CAST_LIMIT, DISCOVER_GALLERY, DISCOVER_TAGGER, LINKED_FILE, REF_IMAGES, RETRAIN_FILE, SESSION_FILE, TAGGER_ID
from .danbooru import danbooru_find_tags, danbooru_reference_images, danbooru_related
from .files import collect_images, original_stem, remove_empty_dirs, transfer_unique
from .gallery import Gallery
from .linking import ask_tag, link_new_characters
from .logs import log, progress
from .names import base_name, legalize, pretty, title_name
from .records import load_aliases, load_index, load_overrides, load_sus, save_aliases, save_index, save_overrides, save_sus
from .siglip import Sorter, pick_device
from .tagger import Tagger


def slug_pair(text):
    *_, series, char = [""] + [part for part in re.split(r"[\\/]", text.strip()) if part]
    return legalize(series).lower(), legalize(base_name(char)).lower()


def parse_corrections(path):
    corrections = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        old, _, new = line.partition("->")
        corrections.append((slug_pair(old), slug_pair(new) if new.strip() else None))
    return corrections


def apply_corrections(root, corrections, gallery):
    review = []
    for anime_dir in (d for d in root.rglob(ANIME) if d.is_dir()):
        folders = {(legalize(s.name).lower(), legalize(c.name).lower()): c
                   for s in anime_dir.iterdir() if s.is_dir() for c in s.iterdir() if c.is_dir()}
        for old, new in corrections:
            folder = folders.get(old)
            if folder is None:
                continue
            images = collect_images(folder, folder)
            if new is None:
                review += [(p, anime_dir, folder.parent.name, folder.name, None) for p in images]
                continue
            target = anime_dir / pretty(new[0]) / pretty(new[1])
            for p in images:
                try:
                    reindex(root, p, transfer_unique(p, target, "move", f"{new[1]}_{original_stem(p)[:10]}"))
                except Exception as e:
                    log(f"Skip {p.name}: {e}")
            log(f"Moved {len(images)} images: {folder.relative_to(root)} -> {target.relative_to(root)}")
    for old, new in corrections:
        changed = gallery.relabel(old, new)
        if changed:
            log(f"Gallery: {'relabeled' if new else 'removed'} {changed} pictures of {pretty(old[0])}/{pretty(old[1])}")
    gallery.save()
    remove_empty_dirs(root)
    log(f"{len(review)} images queued for review")
    return review


def label_of_rel(rel):
    parts = PurePosixPath(rel).parts
    if ANIME not in parts:
        return None, None
    k = parts.index(ANIME)
    depth = len(parts) - k
    if depth == 4:
        return legalize(parts[k + 1]).lower(), legalize(parts[k + 2]).lower()
    return (legalize(parts[k + 1]).lower(), None) if depth == 3 else (None, None)


def mark_retrain(*labels):
    marked = {tuple(l) for l in json.loads(RETRAIN_FILE.read_text(encoding="utf-8"))} if RETRAIN_FILE.exists() else set()
    marked |= {(t, c) for t, c in labels if c}
    RETRAIN_FILE.parent.mkdir(parents=True, exist_ok=True)
    RETRAIN_FILE.write_text(json.dumps(sorted(marked)), encoding="utf-8")


def reindex(root, old_path, new_path):
    index = load_index(root)
    new_rel = new_path.relative_to(root).as_posix()
    old_rel = old_path.relative_to(root).as_posix()
    entry = index.pop(old_rel, {"source": ""})
    title, char = label_of_rel(new_rel)
    old_label = label_of_rel(old_rel)
    if old_label[1] and old_label != (title, char):
        mark_retrain(old_label, (title, char))
    index[new_rel] = {**entry, "dest": new_rel, "title": title, "char": char}
    save_index(root, index)


def load_session(root):
    if not SESSION_FILE.exists():
        return None
    session = json.loads(SESSION_FILE.read_text(encoding="utf-8"))
    return session if session.get("root") == str(root) else None


def save_session(session):
    SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
    SESSION_FILE.write_text(json.dumps(session, indent=1, ensure_ascii=False), encoding="utf-8")


def choose_series_tag(slug):
    found = danbooru_find_tags(slug, 3)
    exact = [name for name, _ in found if legalize(title_name(name)).lower() == slug or legalize(base_name(name)).lower() == slug]
    if exact:
        return exact[0]
    return ask_tag(f"Danbooru series for {pretty(slug)}", found, 3)


def detect_drift(root):
    index = load_index(root)
    current = {p.relative_to(root).as_posix(): p for p in collect_images(root, root)}
    if not index:
        for rel in current:
            title, char = label_of_rel(rel)
            index[rel] = {"dest": rel, "source": "", "title": title, "char": char}
        save_index(root, index)
        log(f"No index yet. Recorded the current layout of {len(index)} images as the baseline.")
        return []
    unindexed = {}
    for rel in current:
        if rel not in index:
            unindexed.setdefault(PurePosixPath(rel).name, []).append(rel)
    drifts, missing = [], []
    for rel, entry in list(index.items()):
        if rel in current:
            continue
        candidates = unindexed.get(PurePosixPath(rel).name, [])
        if len(candidates) == 1:
            drifts.append({"old": rel, "new": candidates.pop(), "entry": entry, "done": False})
        else:
            missing.append(rel)
    for rel in missing:
        index.pop(rel)
    save_index(root, index)
    if missing:
        log(f"{len(missing)} indexed images were deleted or are ambiguous, dropped from the index")
    log(f"{len(drifts)} moved images found")
    return drifts


def apply_drift(root, model_id, session):
    todo = [d for d in session["drifts"] if not d["done"]]
    if not todo:
        return
    log(f"Applying {len(todo)} moves to index, gallery and series...")
    sorter = Sorter(model_id, pick_device())
    gallery, overrides = Gallery(sorter, model_id), load_overrides()
    for n, drift in enumerate(todo, 1):
        old, new, entry = drift["old"], drift["new"], drift["entry"]
        old_label, new_label = (entry.get("title"), entry.get("char")), label_of_rel(new)
        path = root / new
        try:
            if old_label[1] and new_label[1] and old_label[1] != new_label[1] and path.stem.startswith(f"{old_label[1]}_"):
                path = transfer_unique(path, path.parent, "move", f"{new_label[1]}_{path.stem[len(old_label[1]) + 1:]}")
                new = path.relative_to(root).as_posix()
            _, embs = sorter.embed_images([path], 1)
            if len(embs):
                gallery.relabel_near(embs[0], new_label if new_label[1] else None)
                if new_label[1]:
                    gallery.add(embs, [new_label])
        except Exception as e:
            log(f"Skip {new}: {e}")
            drift["done"] = True
            save_session(session)
            continue
        if old_label[1] and old_label[1] == new_label[1] and new_label[0] and (old_label[0] or "").lower() != new_label[0]:
            overrides[new_label[1]] = new_label[0]
        index = load_index(root)
        index.pop(old, None)
        index[new] = {**entry, "dest": new, "title": new_label[0], "char": new_label[1]}
        save_index(root, index)
        if label_of_rel(old)[1] and label_of_rel(old) != new_label:
            mark_retrain(label_of_rel(old), new_label)
        save_sus([e for e in load_sus() if e["path"] != str(root / old)])
        save_overrides(overrides)
        drift["done"] = True
        save_session(session)
        log(f"[{n}/{len(todo)}] {old} -> {new}")
    gallery.save()
    remove_empty_dirs(root)


def loose_series_images(root):
    loose = {}
    for p in collect_images(root, root):
        parts = p.relative_to(root).parts
        if ANIME in parts and len(parts) - parts.index(ANIME) == 3:
            loose.setdefault(root.joinpath(*parts[:parts.index(ANIME) + 2]), []).append(p)
    return loose


def discover(root, model_id, session, relink=False):
    loose = loose_series_images(root)
    if not loose:
        return
    log(f"Discovering characters for {sum(map(len, loose.values()))} images in {len(loose)} series folders...")
    device = pick_device()
    sorter, tagger = Sorter(model_id, device), Tagger(TAGGER_ID, device)
    gallery = Gallery(sorter, model_id)
    linked = json.loads(LINKED_FILE.read_text(encoding="utf-8")) if LINKED_FILE.exists() else {}
    proposed = {p["path"] for p in session["proposals"]}
    for series_dir, paths in loose.items():
        slug = legalize(series_dir.name).lower()
        key = f"series|{slug}"
        if key not in linked or (relink and linked[key] is None):
            try:
                linked[key] = choose_series_tag(slug)
            except Exception as e:
                log(f"Series lookup failed for {series_dir.name}, will retry next time: {e}")
                continue
            LINKED_FILE.write_text(json.dumps(linked, indent=1, ensure_ascii=False), encoding="utf-8")
        series_tag = linked[key]
        if not series_tag:
            log(f"{series_dir.name}: not found on Danbooru, label these with option 1")
            continue
        if series_tag not in session["casts"]:
            try:
                session["casts"][series_tag] = danbooru_related(series_tag, "character", CAST_LIMIT)
            except Exception as e:
                log(f"Cast lookup failed for {series_tag}, will retry next time: {e}")
                continue
            save_session(session)
        cast = {tag: (slug, legalize(base_name(tag)).lower()) for tag in session["casts"][series_tag]}
        tag_of = {label: tag for tag, label in reversed(list(cast.items()))}
        todo = [tag for tag in cast if tag not in gallery.fetched]
        for n, tag in enumerate(todo, 1):
            progress(f"Fetching {series_dir.name} cast pictures", n, len(todo))
            images = danbooru_reference_images(tag, REF_IMAGES)
            if images is None:
                continue
            gallery.fetched.add(tag)
            if images:
                gallery.add(sorter.embed_pil(images), [cast[tag]] * len(images))
        gallery.save()
        cast_rows = [(i, name) for i, name in enumerate(tagger.char_names) if name in cast]
        for n, path in enumerate(paths, 1):
            progress(f"Matching {series_dir.name} images", n, len(paths))
            rel = path.relative_to(root).as_posix()
            if rel in proposed:
                continue
            label, score, source, tag = None, 0.0, "", None
            try:
                if cast_rows:
                    probs = tagger.probabilities([path])[0][tagger.char_idx]
                    best = max(cast_rows, key=lambda r: probs[r[0]])
                    if probs[best[0]] >= DISCOVER_TAGGER:
                        tag, score, source = best[1], probs[best[0]].item(), "tagger"
                        label = cast[tag]
                if label is None:
                    _, embs = sorter.embed_images([path], 1)
                    match = gallery.match(embs, DISCOVER_GALLERY, set(cast.values()))[0] if len(embs) else None
                    if match:
                        label, score, source = match[0], match[1], "reference"
                        tag = tag_of.get(label)
            except Exception as e:
                log(f"Skip {path.name}: {e}")
                continue
            session["proposals"].append({"path": rel, "label": list(label) if label else None, "tag": tag, "score": round(score, 3), "source": source, "moved": False})
            save_session(session)
    LINKED_FILE.write_text(json.dumps(linked, indent=1, ensure_ascii=False), encoding="utf-8")


def finalize(root, session):
    aliases = load_aliases()
    todo = [p for p in session["proposals"] if p["label"] and not p["moved"]]
    unsure = sum(1 for p in session["proposals"] if not p["label"])
    for n, prop in enumerate(todo, 1):
        path = root / prop["path"]
        if path.exists():
            parts = path.relative_to(root).parts
            anime_dir = root.joinpath(*parts[:parts.index(ANIME) + 1])
            title, char = prop["label"]
            try:
                dest = transfer_unique(path, anime_dir / pretty(title) / pretty(char), "move", f"{char}_{original_stem(path)[:10]}")
                reindex(root, path, dest)
                log(f"[{n}/{len(todo)}] {prop['path']} -> {dest.relative_to(root)} ({prop['source']} {prop['score']:.2f})")
            except Exception as e:
                log(f"Skip {path.name}: {e}")
            if prop["tag"]:
                aliases[prop["tag"]] = [title, char]
                save_aliases(aliases)
        prop["moved"] = True
        save_session(session)
    log(f"Sorted {len(todo)} images into character folders, {unsure} left in series folders for option 1")


def run_corrections(root, args, resume):
    session = load_session(root) if resume else None
    if session is None:
        if SESSION_FILE.exists():
            log("Starting a new correction session, the unfinished one is replaced")
        session = {"root": str(root), "stage": "moves", "drifts": detect_drift(root), "casts": {}, "proposals": []}
        save_session(session)
    for stage, step in (("moves", lambda: apply_drift(root, args.model, session)),
                        ("link", lambda: None if args.no_link else link_new_characters(root, args.model, args.relink)),
                        ("discover", lambda: discover(root, args.model, session, args.relink)),
                        ("finalize", lambda: finalize(root, session))):
        if session["stage"] == stage:
            log(f"Correction stage: {stage}")
            step()
            stages = ["moves", "link", "discover", "finalize", "train"]
            session["stage"] = stages[stages.index(stage) + 1]
            save_session(session)
            torch.cuda.empty_cache()
