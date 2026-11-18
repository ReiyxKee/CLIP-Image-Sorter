import clipsort
import argparse
import json
from collections import Counter
from pathlib import Path
from transformers import pipeline

from clipsort.config import CATEGORY_PROMPTS, CHUNK_SIZE, DEDUPE_MODES, EXPAND_THRESHOLD, FALLBACK_COUNT, FALLBACK_THRESHOLD, GALLERY_THRESHOLD, MODEL_ID, NSFW_ID, NSFW_THRESHOLD, R18, REAL, REALISTIC_THRESHOLD, SERIES_COUNT, SESSION_FILE, TAGGER_ID, TRAINED_DIR, TRAINED_THRESHOLD
from clipsort.files import collect_images, remove_empty_dirs
from clipsort.gallery import Gallery, learn_from_folder
from clipsort import settings
from clipsort.hf_login import ensure_hf_login
from clipsort.dedupe import Deduper
from clipsort.layout import check_layout, save_layout
from clipsort.logs import log
from clipsort.names import describe, describe_series
from clipsort.pools import Expander, build_pool
from clipsort.siglip import Sorter, pick_device
from clipsort.sorting import process_chunk
from clipsort.tagger import Tagger
from clipsort.settings import CONFIG_FILE, FOLDERS
from clipsort.trained import TrainedModel


MODES = ("copy", "move", "recategorize", "learn")
# option: (type, built-in default when the settings file leaves it empty; None = ask at start)
SETTINGS = {
    "target_path": (Path, None),
    "output_path": (Path, None),
    "mode": (str, None),
    "r18": (bool, True),
    "nsfw_threshold": (float, None),
    "chunk_size": (int, None),
    "batch_size": (int, 8),
    "category_threshold": (float, 0.45),
    "character_threshold": (float, 0.9),
    "fallback_count": (int, None),
    "fallback_threshold": (float, None),
    "expand_threshold": (float, None),
    "gallery_threshold": (float, None),
    "trained_threshold": (float, None),
    "realistic_threshold": (float, REALISTIC_THRESHOLD),
    "dedupe": (str, "off"),
    "duplicates_path": (Path, None),
    "similar_distance": (int, 4),
    "model": (str, MODEL_ID),
    "tagger": (str, TAGGER_ID),
    "nsfw_model": (str, NSFW_ID),
}


def parse_args():
    p = argparse.ArgumentParser(description=f"Sort images by style and character using SigLIP2 and WD Tagger. Defaults come from {CONFIG_FILE.name}.")
    p.add_argument("--config", help=f"Settings file to use instead of {CONFIG_FILE}")
    p.add_argument("--target-path", type=Path, help="Folder (or single image) to categorize, or sorted folder to recategorize")
    p.add_argument("--output-path", type=Path, help="Folder to copy sorted images into")
    p.add_argument("--model")
    p.add_argument("--tagger")
    p.add_argument("--nsfw-model")
    p.add_argument("--batch-size", type=int)
    p.add_argument("--chunk-size", type=int, help="Images to fully sort and copy per cycle")
    p.add_argument("--fallback-count", type=int, help="Top Danbooru characters in the fallback pool, 0 disables")
    p.add_argument("--fallback-threshold", type=float, help="Fallback match cutoff 0-1, higher = stricter")
    p.add_argument("--expand-threshold", type=float, help="Series-guided search match cutoff 0-1, higher = stricter")
    p.add_argument("--trained-threshold", type=float, help="Trained model match cutoff 0-1, higher = stricter, -1 disables")
    p.add_argument("--gallery-threshold", type=float, help="Reference picture match cutoff 0-1, higher = stricter, -1 disables")
    p.add_argument("--mode", choices=MODES, help="Copy (default), move, re-sort an output folder in place, or learn from a sorted folder")
    p.add_argument("--category-threshold", type=float)
    p.add_argument("--character-threshold", type=float)
    p.add_argument("--r18", action=argparse.BooleanOptionalAction, help=f"Split adult images into the {R18} folder (--no-r18 turns it off)")
    p.add_argument("--nsfw-threshold", type=float, help="R18 cutoff 0-1, higher flags fewer images, -1 disables")
    p.add_argument("--realistic-threshold", type=float, help=f"Move anime-classified images the tagger rates this realistic to {REAL}, -1 disables")
    p.add_argument("--dedupe", choices=DEDUPE_MODES, help="Send duplicates (exact = same file, similar = same picture resized/re-encoded) to the duplicates folder")
    p.add_argument("--duplicates-path", type=Path, help="Where duplicates go, default <output>/duplicates")
    p.add_argument("--similar-distance", type=int, help="dedupe similar: allowed difference 0-64 bits, default 4")
    args = p.parse_args()
    cli = {k for k, v in vars(args).items() if v is not None}
    settings.apply(args, "sorter", SETTINGS)
    if args.mode is not None and args.mode not in MODES:
        raise SystemExit(f"{CONFIG_FILE}: [sorter] mode must be one of {', '.join(MODES)}")
    if args.dedupe not in DEDUPE_MODES:
        raise SystemExit(f"{CONFIG_FILE}: [sorter] dedupe must be one of {', '.join(DEDUPE_MODES)}")
    if not 0 <= args.similar_distance <= 64:
        raise SystemExit("similar_distance must be 0-64")
    if not args.r18 and "nsfw_threshold" not in cli:
        args.nsfw_threshold = -1
    return args, cli


def ask_value(asked, key, prompt, default, kind=float):
    answer = input(prompt).strip()
    value = kind(answer) if answer else default
    asked[key] = value
    return value


def resolve_paths(args, cli):
    asked = {}
    if args.mode is None:
        answer = input("Copy, move, recategorize, or learn from a sorted folder? [C/m/r/l]: ").strip().lower()
        args.mode = asked["mode"] = {"m": "move", "r": "recategorize", "l": "learn"}.get(answer, answer if answer in ("move", "recategorize", "learn") else "copy")
    recategorize = args.mode in ("recategorize", "learn")
    prompt = f"Sorted folder to {args.mode}: " if recategorize else "Folder to categorize: "
    # [sorter] target_path is the unsorted input; recategorize/learn work on the sorted folder instead
    target = (args.target_path if "target_path" in cli or not recategorize else None) or (args.output_path if recategorize else None)
    if target is None:
        target = Path(input(prompt).strip().strip("'\""))
        asked["output_path" if recategorize else "target_path"] = target
    target = target.expanduser().resolve()
    if not target.exists():
        raise SystemExit(f"Target not found: {target}")
    output = target if recategorize else args.output_path
    if args.mode == "learn":
        return target, target, asked
    if output is None:
        default_out = (target.parent if target.is_dir() else target.parent.parent) / f"{target.stem}_sorted"
        answer = input(f"Output folder [{default_out}]: ").strip().strip("'\"")
        output = asked["output_path"] = Path(answer) if answer else default_out
    if args.chunk_size is None:
        args.chunk_size = ask_value(asked, "chunk_size", f"Images per batch cycle [{CHUNK_SIZE}]: ", CHUNK_SIZE, int)
    if args.nsfw_threshold is None:
        args.nsfw_threshold = ask_value(asked, "nsfw_threshold", f"R18 threshold 0-1, higher = stricter, -1 = off [{NSFW_THRESHOLD}]: ", NSFW_THRESHOLD)
    if args.fallback_count is None:
        args.fallback_count = ask_value(asked, "fallback_count", f"Fallback character pool size, 0 = off [{FALLBACK_COUNT}]: ", FALLBACK_COUNT, int)
    if args.fallback_count > 0 and args.fallback_threshold is None:
        args.fallback_threshold = ask_value(asked, "fallback_threshold", f"Fallback match threshold 0-1, higher = stricter [{FALLBACK_THRESHOLD}]: ", FALLBACK_THRESHOLD)
    if args.fallback_count > 0 and args.expand_threshold is None:
        args.expand_threshold = ask_value(asked, "expand_threshold", f"Series-guided search threshold 0-1, higher = stricter [{EXPAND_THRESHOLD}]: ", EXPAND_THRESHOLD)
    if args.gallery_threshold is None:
        args.gallery_threshold = ask_value(asked, "gallery_threshold", f"Reference picture match threshold 0-1, higher = stricter, -1 = off [{GALLERY_THRESHOLD}]: ", GALLERY_THRESHOLD)
    if args.trained_threshold is None:
        args.trained_threshold = -1
        if (TRAINED_DIR / "head.safetensors").exists():
            args.trained_threshold = ask_value(asked, "trained_threshold", f"Trained model match threshold 0-1, higher = stricter, -1 = off [{TRAINED_THRESHOLD}]: ", TRAINED_THRESHOLD)
    return target, output.expanduser().resolve(), asked


def make_deduper(args, output):
    if args.dedupe == "off":
        return None
    dup_dir = args.duplicates_path.expanduser() if args.duplicates_path else Path("duplicates")
    dup_dir = (dup_dir if dup_dir.is_absolute() else output / dup_dir).resolve()
    if dup_dir == output or dup_dir in output.parents:
        raise SystemExit(f"duplicates_path {dup_dir} must not be the output folder or above it")
    top = dup_dir.relative_to(output).parts[0].lower() if output in dup_dir.parents else None
    if top in {name.lower() for name in FOLDERS.values()}:
        raise SystemExit(f"duplicates_path {dup_dir} is inside a sorted category folder; pick another name")
    log(f"Dedupe ({args.dedupe}): duplicates go to {dup_dir}")
    return Deduper(output, dup_dir, args.dedupe, args.similar_distance, "copy" if args.mode == "copy" else "move")


def main():
    args, cli = parse_args()
    ensure_hf_login()
    if SESSION_FILE.exists():
        pending = json.loads(SESSION_FILE.read_text(encoding="utf-8"))
        log(f"Warning: unfinished correction session found for {pending.get('root')} (stopped at stage '{pending.get('stage')}')")
        if input("Sort anyway? Recover first with clip_sorter_train.py option 2 [y/N]: ").strip().lower() not in ("y", "yes"):
            raise SystemExit(f'Recover with: python clip_sorter_train.py --target-path "{pending.get("root")}"')
    target, output, asked = resolve_paths(args, cli)
    settings.offer_save("sorter", asked)
    check_layout(output)
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

    save_layout(output)
    deduper = make_deduper(args, output)
    totals, r18_total = Counter(), 0
    chunk_count = -(-len(paths) // args.chunk_size)
    for start in range(0, len(paths), args.chunk_size):
        chunk = paths[start:start + args.chunk_size]
        label = f"Chunk {start // args.chunk_size + 1}/{chunk_count}"
        log(f"{label}: images {start + 1}-{start + len(chunk)} of {len(paths)}")
        try:
            categories, r18 = process_chunk(chunk, models, args, output, label, None if recategorize else done_file, deduper)
        except Exception as e:
            log(f"{label} failed, skipping: {e}")
            continue
        tagger.save_cache()
        if expander:
            expander.save()
        if gallery:
            gallery.save()
        if deduper:
            deduper.save()
        totals.update(categories)
        r18_total += sum(r18)

    if recategorize:
        remove_empty_dirs(output)
    log(f"Done. Summary: {dict(totals)}, {R18}: {r18_total}")


if __name__ == "__main__":
    main()
