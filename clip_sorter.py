import clipsort
import argparse
import json
from collections import Counter
from pathlib import Path
from transformers import pipeline

from clipsort.config import CATEGORY_PROMPTS, CHUNK_SIZE, EXPAND_THRESHOLD, FALLBACK_COUNT, FALLBACK_THRESHOLD, GALLERY_THRESHOLD, MODEL_ID, NSFW_ID, NSFW_THRESHOLD, R18, SERIES_COUNT, SESSION_FILE, TAGGER_ID, TRAINED_DIR, TRAINED_THRESHOLD
from clipsort.files import collect_images, remove_empty_dirs
from clipsort.gallery import Gallery, learn_from_folder
from clipsort.hf_login import ensure_hf_login
from clipsort.logs import log
from clipsort.names import describe, describe_series
from clipsort.pools import Expander, build_pool
from clipsort.siglip import Sorter, pick_device
from clipsort.sorting import process_chunk
from clipsort.tagger import Tagger
from clipsort.trained import TrainedModel


def parse_args():
    p = argparse.ArgumentParser(description="Sort images by style and character using SigLIP2 and WD Tagger.")
    p.add_argument("--target-path", type=Path, help="Folder (or single image) to categorize, or sorted folder to recategorize")
    p.add_argument("--output-path", type=Path, help="Folder to copy sorted images into")
    p.add_argument("--model", default=MODEL_ID)
    p.add_argument("--tagger", default=TAGGER_ID)
    p.add_argument("--nsfw-model", default=NSFW_ID)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--chunk-size", type=int, help="Images to fully sort and copy per cycle")
    p.add_argument("--fallback-count", type=int, help="Top Danbooru characters in the fallback pool, 0 disables")
    p.add_argument("--fallback-threshold", type=float, help="Fallback match cutoff 0-1, higher = stricter")
    p.add_argument("--expand-threshold", type=float, help="Series-guided search match cutoff 0-1, higher = stricter")
    p.add_argument("--trained-threshold", type=float, help="Trained model match cutoff 0-1, higher = stricter, -1 disables")
    p.add_argument("--gallery-threshold", type=float, help="Reference picture match cutoff 0-1, higher = stricter, -1 disables")
    p.add_argument("--mode", choices=("copy", "move", "recategorize", "learn"), help="Copy (default), move, re-sort an output folder in place, or learn from a sorted folder")
    p.add_argument("--category-threshold", type=float, default=0.45)
    p.add_argument("--character-threshold", type=float, default=0.9)
    p.add_argument("--nsfw-threshold", type=float, help="R18 cutoff 0-1, higher flags fewer images, -1 disables")
    return p.parse_args()


def resolve_paths(args):
    if args.mode is None:
        answer = input("Copy, move, recategorize, or learn from a sorted folder? [C/m/r/l]: ").strip().lower()
        args.mode = {"m": "move", "r": "recategorize", "l": "learn"}.get(answer, answer if answer in ("move", "recategorize", "learn") else "copy")
    recategorize = args.mode in ("recategorize", "learn")
    prompt = f"Sorted folder to {args.mode}: " if recategorize else "Folder to categorize: "
    target = args.target_path or (args.output_path if recategorize else None) or Path(input(prompt).strip().strip("'\""))
    target = target.expanduser().resolve()
    if not target.exists():
        raise SystemExit(f"Target not found: {target}")
    output = target if recategorize else args.output_path
    if args.mode == "learn":
        return target, target
    if output is None:
        default_out = (target.parent if target.is_dir() else target.parent.parent) / f"{target.stem}_sorted"
        answer = input(f"Output folder [{default_out}]: ").strip().strip("'\"")
        output = Path(answer) if answer else default_out
    if args.chunk_size is None:
        answer = input(f"Images per batch cycle [{CHUNK_SIZE}]: ").strip()
        args.chunk_size = int(answer) if answer else CHUNK_SIZE
    if args.nsfw_threshold is None:
        answer = input(f"R18 threshold 0-1, higher = stricter, -1 = off [{NSFW_THRESHOLD}]: ").strip()
        args.nsfw_threshold = float(answer) if answer else NSFW_THRESHOLD
    if args.fallback_count is None:
        answer = input(f"Fallback character pool size, 0 = off [{FALLBACK_COUNT}]: ").strip()
        args.fallback_count = int(answer) if answer else FALLBACK_COUNT
    if args.fallback_count > 0 and args.fallback_threshold is None:
        answer = input(f"Fallback match threshold 0-1, higher = stricter [{FALLBACK_THRESHOLD}]: ").strip()
        args.fallback_threshold = float(answer) if answer else FALLBACK_THRESHOLD
    if args.fallback_count > 0 and args.expand_threshold is None:
        answer = input(f"Series-guided search threshold 0-1, higher = stricter [{EXPAND_THRESHOLD}]: ").strip()
        args.expand_threshold = float(answer) if answer else EXPAND_THRESHOLD
    if args.gallery_threshold is None:
        answer = input(f"Reference picture match threshold 0-1, higher = stricter, -1 = off [{GALLERY_THRESHOLD}]: ").strip()
        args.gallery_threshold = float(answer) if answer else GALLERY_THRESHOLD
    if args.trained_threshold is None:
        args.trained_threshold = -1
        if (TRAINED_DIR / "head.safetensors").exists():
            answer = input(f"Trained model match threshold 0-1, higher = stricter, -1 = off [{TRAINED_THRESHOLD}]: ").strip()
            args.trained_threshold = float(answer) if answer else TRAINED_THRESHOLD
    return target, output.expanduser().resolve()


def main():
    args = parse_args()
    ensure_hf_login()
    if SESSION_FILE.exists():
        pending = json.loads(SESSION_FILE.read_text(encoding="utf-8"))
        log(f"Warning: unfinished correction session found for {pending.get('root')} (stopped at stage '{pending.get('stage')}')")
        if input("Sort anyway? Recover first with clip_sorter_train.py option 2 [y/N]: ").strip().lower() not in ("y", "yes"):
            raise SystemExit(f'Recover with: python clip_sorter_train.py --target-path "{pending.get("root")}"')
    target, output = resolve_paths(args)
    if args.mode == "learn":
        device = pick_device()
        log(f"Loading {args.model} on {device}...")
        sorter = Sorter(args.model, device)
        learn_from_folder(target, sorter, Gallery(sorter, args.model), args.batch_size)
        return
    recategorize = args.mode == "recategorize"
    done_file = output / ".sorted_sources.txt"
    done = set(done_file.read_text(encoding="utf-8", errors="replace").splitlines()) if done_file.exists() and not recategorize else set()
    paths = [p for p in collect_images(target, output) if str(p) not in done]
    if not paths:
        raise SystemExit(f"No new images found in {target}")
    log(f"Found {len(paths)} images to {'recategorize' if recategorize else 'sort'} ({len(done)} already sorted).")

    device = pick_device()
    log(f"Loading [1/4] {args.model} on {device}...")
    sorter = Sorter(args.model, device)
    class_embs = sorter.embed_groups(CATEGORY_PROMPTS.values())
    nsfw = None
    if args.nsfw_threshold >= 0:
        log(f"Loading [2/4] {args.nsfw_model}...")
        nsfw = pipeline("image-classification", model=args.nsfw_model, device=device)
    log(f"Loading [3/4] {args.tagger}...")
    tagger = Tagger(args.tagger, device)
    log("Loading [4/4] fallback character and series lists...")
    fallback = build_pool(sorter, args.model, "characters", 4, args.fallback_count, describe)
    series_pool = build_pool(sorter, args.model, "series", 3, SERIES_COUNT if fallback else 0, describe_series)
    expander = Expander(sorter, args.model, series_pool, fallback) if series_pool else None
    gallery = Gallery(sorter, args.model) if args.gallery_threshold >= 0 else None
    if gallery:
        log(f"Reference gallery: {len(gallery.labels)} pictures")
    trained = None
    if args.trained_threshold >= 0 and not all((TRAINED_DIR / f).exists() for f in ("config.json", "head.safetensors", "labels.json")):
        log(f"No trained model in {TRAINED_DIR}, skipping it. Run clip_sorter_train.py first.")
    elif args.trained_threshold >= 0:
        log(f"Loading trained model from {TRAINED_DIR}...")
        try:
            trained = TrainedModel(TRAINED_DIR, device)
        except Exception as e:
            log(f"Trained model failed to load, skipping it: {e}")
    models = (sorter, class_embs, nsfw, tagger, fallback, expander, gallery, trained)
    log("All models ready.")

    output.mkdir(parents=True, exist_ok=True)
    totals, r18_total = Counter(), 0
    chunk_count = -(-len(paths) // args.chunk_size)
    for start in range(0, len(paths), args.chunk_size):
        chunk = paths[start:start + args.chunk_size]
        label = f"Chunk {start // args.chunk_size + 1}/{chunk_count}"
        log(f"{label}: images {start + 1}-{start + len(chunk)} of {len(paths)}")
        try:
            categories, r18 = process_chunk(chunk, models, args, output, label, None if recategorize else done_file)
        except Exception as e:
            log(f"{label} failed, skipping: {e}")
            continue
        tagger.save_cache()
        if expander:
            expander.save()
        if gallery:
            gallery.save()
        totals.update(categories)
        r18_total += sum(r18)

    if recategorize:
        remove_empty_dirs(output)
    log(f"Done. Summary: {dict(totals)}, {R18}: {r18_total}")


if __name__ == "__main__":
    main()
