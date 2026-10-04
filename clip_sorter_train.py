import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from difflib import get_close_matches
from pathlib import Path

from clip_sorter import (
    ANIME, MODEL_ID, QUALIFIER, TRAINED_DIR, base_name, collect_images, features, labeled_images,
    legalize, log, pick_device, pretty, progress, title_name, transfer_unique,
)

import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file, save_file
from torchvision import transforms
from transformers import AutoModel, AutoProcessor


def parse_args():
    p = argparse.ArgumentParser(description="Label unknown anime images and fine-tune a SigLIP2 copy as the last fallback.")
    p.add_argument("--target-path", type=Path, help="Sorted output folder")
    p.add_argument("--model", default=MODEL_ID, help="Base model when no trained copy exists yet")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--unfreeze", type=int, default=2, help="Last vision layers to fine-tune, 0 = classifier only")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--skip-labeling", action="store_true")
    return p.parse_args()


def open_image(path):
    if sys.platform == "win32":
        os.startfile(path)
    else:
        subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def unlabeled_images(root):
    items = []
    for p in collect_images(root, root):
        parts = p.relative_to(root).parts
        if ANIME in parts:
            k = parts.index(ANIME)
            depth = len(parts) - k
            if depth in (2, 3):
                items.append((p, root.joinpath(*parts[:k + 1]), parts[k + 1] if depth == 3 else ""))
    return items


def known_names(root):
    series, chars = set(), {}
    for anime_dir in (d for d in root.rglob(ANIME) if d.is_dir()):
        for folder in (d for d in anime_dir.iterdir() if d.is_dir()):
            slug = legalize(folder.name).lower()
            series.add(slug)
            chars.setdefault(slug, set()).update(legalize(c.name).lower() for c in folder.iterdir() if c.is_dir())
    for name in ("series", "characters"):
        pool = Path(os.environ["HF_HOME"]) / f"fallback_{name}.pt"
        if not pool.exists():
            continue
        for tag in torch.load(pool)["tags"]:
            if name == "series":
                series.add(legalize(title_name(tag)).lower())
                continue
            qualifiers = QUALIFIER.findall(tag)
            title = legalize(title_name(qualifiers[-1])).lower() if qualifiers else ""
            chars.setdefault(title, set()).add(legalize(base_name(tag)).lower())
    return series, chars


def pick(kind, slug, known):
    if slug in known:
        return slug
    close = get_close_matches(slug, known, n=3, cutoff=0.75)
    if not close:
        return slug
    options = "  ".join(f"{i}) {pretty(c)}" for i, c in enumerate(close, 1))
    answer = input(f"  {kind} '{pretty(slug)}' not found. Similar: {options}  [number / Enter = keep]: ").strip()
    return close[int(answer) - 1] if answer.isdigit() and 1 <= int(answer) <= len(close) else slug


def label_images(root):
    items = unlabeled_images(root)
    log(f"{len(items)} anime images without a character. Enter = skip, q = stop and train.")
    series_known, chars_known = known_names(root)
    all_chars = set().union(*chars_known.values()) if chars_known else set()
    last_series = ""
    for n, (path, anime_dir, series_folder) in enumerate(items, 1):
        try:
            open_image(path)
        except Exception as e:
            log(f"Could not open viewer for {path.name}: {e}")
        while True:
            char = input(f"[{n}/{len(items)}] {path.name} character: ").strip()
            if char.lower() == "q":
                return
            if not char:
                break
            default = series_folder or last_series
            series = input(f"  series [{default}]: ").strip() or default
            if not series:
                log("  Series is required")
                continue
            title = pick("Series", legalize(title_name(series)).lower(), series_known)
            slug = legalize(base_name(char)).lower()
            local = chars_known.get(title, set())
            slug = pick("Character", slug, local if slug in local or get_close_matches(slug, local, 1, 0.75) else all_chars)
            target = anime_dir / pretty(title) / pretty(slug)
            answer = input(f"  Move to {target.relative_to(root)}? [Y/n = skip/e = edit]: ").strip().lower()
            if answer == "e":
                continue
            if answer in ("", "y", "yes"):
                try:
                    dest = transfer_unique(path, target, "move", f"{slug}_{path.stem[:10]}")
                    log(f"  -> {dest.relative_to(root)}")
                    last_series = pretty(title)
                    series_known.add(title)
                    chars_known.setdefault(title, set()).add(slug)
                    all_chars.add(slug)
                except Exception as e:
                    log(f"  Skip {path.name}: {e}")
            break


def load_image(path, augment):
    try:
        return augment(Image.open(path).convert("RGB"))
    except Exception as e:
        log(f"Skip {path.name}: {e}")
        return None


def train(items, args, device):
    resume = (TRAINED_DIR / "head.safetensors").exists()
    source = TRAINED_DIR if resume else args.model
    log(f"Loading {source} on {device}...")
    model = AutoModel.from_pretrained(source).to(device)
    processor = AutoProcessor.from_pretrained(source)

    labels = sorted({label for _, label in items})
    index = {label: i for i, label in enumerate(labels)}
    head = torch.nn.Linear(model.config.vision_config.hidden_size, len(labels)).to(device)
    if resume:
        old = load_file(TRAINED_DIR / "head.safetensors")
        old_labels = [tuple(l) for l in json.loads((TRAINED_DIR / "labels.json").read_text(encoding="utf-8"))]
        with torch.no_grad():
            for j, label in enumerate(old_labels):
                if label in index:
                    head.weight[index[label]] = old["weight"][j].to(device)
                    head.bias[index[label]] = old["bias"][j].to(device)

    for param in model.parameters():
        param.requires_grad = False
    vision = model.vision_model
    tuned = [*vision.encoder.layers[-args.unfreeze:], vision.post_layernorm, getattr(vision, "head", None)] if args.unfreeze > 0 else []
    backbone = [param for module in tuned if module is not None for param in module.parameters()]
    for param in backbone:
        param.requires_grad = True
    optimizer = torch.optim.AdamW([{"params": backbone, "lr": args.lr}, {"params": head.parameters(), "lr": 1e-3}])
    augment = transforms.Compose([transforms.RandomResizedCrop(384, scale=(0.6, 1.0)), transforms.RandomHorizontalFlip()])
    log(f"Training on {len(items)} images, {len(labels)} characters, last {args.unfreeze} layers unfrozen")

    for epoch in range(1, args.epochs + 1):
        random.shuffle(items)
        model.train()
        seen = correct = 0
        total_loss = 0.0
        for i in range(0, len(items), args.batch_size):
            batch = [(load_image(p, augment), label) for p, label in items[i:i + args.batch_size]]
            batch = [(img, label) for img, label in batch if img is not None]
            if not batch:
                continue
            try:
                inputs = processor(images=[img for img, _ in batch], return_tensors="pt").to(device)
                target = torch.tensor([index[label] for _, label in batch], device=device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                    logits = head(features(model.get_image_features(**inputs)).float())
                loss = F.cross_entropy(logits.float(), target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            except Exception as e:
                log(f"Batch skipped: {e}")
                continue
            seen += len(batch)
            correct += (logits.argmax(dim=-1) == target).sum().item()
            total_loss += loss.item() * len(batch)
            progress(f"Epoch {epoch}/{args.epochs}", min(i + args.batch_size, len(items)), len(items))
        log(f"Epoch {epoch}: loss {total_loss / max(seen, 1):.3f}, train accuracy {correct * 100 // max(seen, 1)}%")

    staging = TRAINED_DIR.with_name(TRAINED_DIR.name + "_new")
    shutil.rmtree(staging, ignore_errors=True)
    model.save_pretrained(staging)
    processor.save_pretrained(staging)
    save_file({"weight": head.weight.detach().cpu().contiguous(), "bias": head.bias.detach().cpu().contiguous()}, staging / "head.safetensors")
    (staging / "labels.json").write_text(json.dumps([list(l) for l in labels]), encoding="utf-8")
    del model
    shutil.rmtree(TRAINED_DIR, ignore_errors=True)
    staging.rename(TRAINED_DIR)
    log(f"Saved trained model to {TRAINED_DIR}")


def main():
    args = parse_args()
    root = (args.target_path or Path(input("Sorted output folder: ").strip().strip("'\""))).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Folder not found: {root}")
    if not args.skip_labeling:
        label_images(root)
    items = labeled_images(root)
    if len({label for _, label in items}) < 2:
        raise SystemExit("Need at least 2 characters in <Series>/<Character>/ folders to train")
    train(items, args, pick_device())


if __name__ == "__main__":
    main()
