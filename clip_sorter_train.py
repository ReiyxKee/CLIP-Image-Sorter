import argparse
import json
import os
import random
import re
import shutil
import tkinter as tk
from difflib import get_close_matches
from pathlib import Path

from clip_sorter import (
    ANIME, MODEL_ID, QUALIFIER, TRAINED_DIR, Gallery, base_name, collect_images, features, labeled_images,
    legalize, load_sus, log, original_stem, save_sus, pick_device, pretty, progress, remove_empty_dirs, title_name, transfer_unique,
)

import torch
import torch.nn.functional as F
from PIL import Image, ImageTk
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
    p.add_argument("--task", choices=("1", "2", "3", "4"), help="1 sort unclassed, 2 fix wrong category, 3 resolve suspicious, 4 train only")
    p.add_argument("--skip-labeling", action="store_true", help="Same as --task 4")
    p.add_argument("--corrections", type=Path, help="Text file of wrongly sorted characters, see README")
    return p.parse_args()


MULTI_FILE = ".multi_labels.json"


def load_multi(root):
    path = root / MULTI_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_multi(root, multi):
    (root / MULTI_FILE).write_text(json.dumps(multi, indent=1, ensure_ascii=False), encoding="utf-8")


def unlabeled_images(root):
    items, multi = [], load_multi(root)
    for p in collect_images(root, root):
        if p.relative_to(root).as_posix() in multi:
            continue
        parts = p.relative_to(root).parts
        if ANIME in parts:
            k = parts.index(ANIME)
            depth = len(parts) - k
            if depth in (2, 3):
                items.append((p, root.joinpath(*parts[:k + 1]), parts[k + 1] if depth == 3 else "", "", None))
    return items


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
                    transfer_unique(p, target, "move", f"{new[1]}_{original_stem(p)[:10]}")
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


def suspicious_images(root):
    items = []
    for entry in load_sus():
        path = Path(entry["path"])
        if not path.exists() or root not in path.parents:
            continue
        parts = path.relative_to(root).parts
        if ANIME not in parts:
            continue
        k = parts.index(ANIME)
        depth = len(parts) - k
        items.append((path, root.joinpath(*parts[:k + 1]), parts[k + 1] if depth >= 3 else "", parts[k + 2] if depth == 4 else "", entry))
    return items


def drop_sus(path):
    save_sus([e for e in load_sus() if e["path"] != str(path)])


def target_for(path, anime_dir, labels):
    if len(labels) == 1:
        title, slug = labels[0]
        return anime_dir / pretty(title) / pretty(slug), f"{slug}_{original_stem(path)[:10]}"
    titles = {t for t, _ in labels}
    return (anime_dir / pretty(titles.pop()) if len(titles) == 1 else anime_dir), original_stem(path)


def file_image(root, path, anime_dir, labels, multi):
    target, stem = target_for(path, anime_dir, labels)
    dest = transfer_unique(path, target, "move", stem)
    multi.pop(path.relative_to(root).as_posix(), None)
    if len(labels) > 1:
        multi[dest.relative_to(root).as_posix()] = [list(l) for l in labels]
    save_multi(root, multi)
    return dest


def to_slugs(char, series):
    return legalize(title_name(series)).lower(), legalize(base_name(char)).lower()


class LabelApp:
    def __init__(self, root, items):
        self.root, self.items, self.index = root, items, 0
        self.series_known, self.chars_known = known_names(root)
        self.all_chars = set().union(*self.chars_known.values()) if self.chars_known else set()
        self.multi, self.history, self.rows, self.last_series = load_multi(root), [], [], ""
        self.win = tk.Tk()
        self.win.title("CLIP Image Sorter - Label")
        self.image_label = tk.Label(self.win)
        self.image_label.pack(side="left", padx=8, pady=8)
        panel = tk.Frame(self.win)
        panel.pack(side="left", fill="y", padx=8, pady=8)
        self.info = tk.Label(panel, justify="left", anchor="w", wraplength=380)
        self.info.pack(fill="x")
        self.choices = tk.Frame(panel)
        self.choices.pack(fill="x")
        self.rows_frame = tk.Frame(panel)
        self.rows_frame.pack(fill="x", pady=8)
        tk.Button(panel, text="+ Character", command=self.add_row).pack(anchor="w")
        self.preview = tk.Label(panel, justify="left", anchor="w", wraplength=380)
        self.preview.pack(fill="x", pady=8)
        buttons = tk.Frame(panel)
        buttons.pack(fill="x")
        for text, command in (("Save (Enter)", self.save), ("Skip (Esc)", self.skip), ("Back (Ctrl+Z)", self.back), ("Stop & train", self.win.destroy)):
            tk.Button(buttons, text=text, command=command).pack(side="left", padx=2)
        self.status = tk.Label(panel, fg="gray", justify="left", anchor="w", wraplength=380)
        self.status.pack(fill="x", pady=8)
        self.win.bind("<Return>", lambda e: self.save())
        self.win.bind("<Escape>", lambda e: self.skip())
        self.win.bind("<Control-z>", lambda e: self.back())
        self.show()
        self.win.mainloop()

    def current(self):
        return self.items[self.index]

    def show(self):
        while self.index < len(self.items):
            path, _, series_folder, current, entry = self.current()
            try:
                image = Image.open(path).convert("RGB")
                image.thumbnail((900, 800))
                break
            except Exception as e:
                self.status["text"] = f"Skip {path.name}: {e}"
                self.index += 1
        else:
            self.win.destroy()
            return
        self.photo = ImageTk.PhotoImage(image)
        self.image_label.configure(image=self.photo)
        self.info["text"] = f"[{self.index + 1}/{len(self.items)}] {path.name}\nIn: {path.parent.relative_to(self.root)}"
        for row in self.rows:
            row["frame"].destroy()
        self.rows = []
        for widget in self.choices.winfo_children():
            widget.destroy()
        if entry:
            tk.Label(self.choices, text="Suspicious, models disagree:", fg="red").pack(anchor="w")
            for name, (title, slug) in (("Trained", entry["trained"]), (entry["source"] or "Other", entry["other"])):
                if slug or title:
                    tk.Button(self.choices, text=f"{name}: {pretty(slug) if slug else '?'} ({pretty(title) if title else '?'})",
                              command=lambda t=title, c=slug: self.use_choice(t, c)).pack(anchor="w")
        self.add_row(current, series_folder or self.last_series)

    def use_choice(self, title, slug):
        row = self.rows[0]
        row["char"].set(pretty(slug) if slug else "")
        row["series"].set(pretty(title) if title else "")

    def add_row(self, char="", series=None):
        frame = tk.Frame(self.rows_frame)
        frame.pack(fill="x", pady=4)
        char_var = tk.StringVar(value=char)
        series_var = tk.StringVar(value=self.last_series if series is None else series)
        row = {"frame": frame, "char": char_var, "series": series_var, "hints": tk.Frame(frame), "job": None}
        tk.Label(frame, text="Character").grid(row=0, column=0, sticky="w")
        char_entry = tk.Entry(frame, textvariable=char_var, width=28)
        char_entry.grid(row=0, column=1)
        tk.Label(frame, text="Series").grid(row=1, column=0, sticky="w")
        tk.Entry(frame, textvariable=series_var, width=28).grid(row=1, column=1)
        tk.Button(frame, text="x", command=lambda: self.remove_row(row)).grid(row=0, column=2, rowspan=2, padx=4)
        row["hints"].grid(row=2, column=0, columnspan=3, sticky="w")
        for var in (char_var, series_var):
            var.trace_add("write", lambda *_: self.schedule_hints(row))
        self.rows.append(row)
        self.schedule_hints(row)
        char_entry.focus_set()
        char_entry.icursor("end")

    def remove_row(self, row):
        row["frame"].destroy()
        self.rows.remove(row)
        self.update_preview()

    def schedule_hints(self, row):
        if row["job"]:
            self.win.after_cancel(row["job"])
        row["job"] = self.win.after(400, lambda: self.update_hints(row))

    def update_hints(self, row):
        row["job"] = None
        for widget in row["hints"].winfo_children():
            widget.destroy()
        char, series = row["char"].get().strip(), row["series"].get().strip()
        title, slug = to_slugs(char or "x", series or "x")
        local = self.chars_known.get(title, set())
        checks = [(row["series"], title, self.series_known, "series")] if series else []
        if char:
            checks.append((row["char"], slug, local if slug in local or get_close_matches(slug, local, 1, 0.75) else self.all_chars, "character"))
        for var, value, known, kind in checks:
            if value in known:
                continue
            matches = get_close_matches(value, known, n=3, cutoff=0.75)
            if not matches:
                tk.Label(row["hints"], text=f"new {kind}", fg="orange").pack(side="left")
            for match in matches:
                tk.Button(row["hints"], text=f"-> {pretty(match)}", command=lambda v=var, m=match: v.set(pretty(m))).pack(side="left")
        self.update_preview()

    def labels(self):
        labels = []
        for row in self.rows:
            char, series = row["char"].get().strip(), row["series"].get().strip()
            if char and series:
                labels.append(to_slugs(char, series))
            elif char:
                return None
        return list(dict.fromkeys(labels))

    def update_preview(self):
        labels = self.labels()
        if labels is None:
            self.preview["text"] = "Every character needs a series"
        elif labels:
            path, anime_dir, _, _, _ = self.current()
            self.preview["text"] = f"Will move to: {target_for(path, anime_dir, labels)[0].relative_to(self.root)}"
        else:
            self.preview["text"] = ""

    def save(self):
        labels = self.labels()
        if labels is None:
            self.status["text"] = "Every character needs a series"
            return
        if not labels:
            return self.skip()
        path, anime_dir, _, _, entry = self.current()
        try:
            dest = file_image(self.root, path, anime_dir, labels, self.multi)
            if entry:
                drop_sus(path)
        except Exception as e:
            self.status["text"] = f"Failed: {e}"
            return
        self.history.append((self.index, path, dest, entry))
        for title, slug in labels:
            self.series_known.add(title)
            self.chars_known.setdefault(title, set()).add(slug)
            self.all_chars.add(slug)
        self.last_series = pretty(labels[-1][0])
        self.status["text"] = f"Saved -> {dest.relative_to(self.root)}"
        self.index += 1
        self.show()

    def skip(self):
        self.status["text"] = f"Skipped {self.current()[0].name}"
        self.index += 1
        self.show()

    def back(self):
        if not self.history:
            self.status["text"] = "Nothing to undo"
            return
        index, original, dest, entry = self.history.pop()
        if original.exists() and original != dest:
            self.status["text"] = f"Cannot undo, {original.name} already exists"
            return
        try:
            original.parent.mkdir(parents=True, exist_ok=True)
            if original != dest:
                shutil.move(dest, original)
            self.multi.pop(dest.relative_to(self.root).as_posix(), None)
            save_multi(self.root, self.multi)
            if entry:
                save_sus(load_sus() + [entry])
        except Exception as e:
            self.status["text"] = f"Undo failed: {e}"
            return
        self.status["text"] = f"Undone, {original.name} moved back"
        self.index = index
        self.show()


def label_images(root, items):
    log(f"{len(items)} images to label, opening labeling window...")
    if items:
        LabelApp(root, items)
    remove_empty_dirs(root)


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

    labels = sorted({label for _, group in items for label in group})
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
    log(f"Training on {len(items)} images, {len(labels)} characters, last {args.unfreeze} layers unfrozen. Ctrl+C stops and saves.")
    try:
        run_epochs(model, processor, head, optimizer, augment, items, labels, index, args, device)
    except KeyboardInterrupt:
        log("Stopped early, saving progress so far...")
    save_trained(model, processor, head, labels)


def run_epochs(model, processor, head, optimizer, augment, items, labels, index, args, device):
    for epoch in range(1, args.epochs + 1):
        random.shuffle(items)
        model.train()
        seen = correct = 0
        total_loss = 0.0
        for i in range(0, len(items), args.batch_size):
            batch = [(load_image(p, augment), group) for p, group in items[i:i + args.batch_size]]
            batch = [(img, group) for img, group in batch if img is not None]
            if not batch:
                continue
            try:
                inputs = processor(images=[img for img, _ in batch], return_tensors="pt").to(device)
                target = torch.zeros(len(batch), len(labels), device=device)
                for row, (_, group) in enumerate(batch):
                    target[row, [index[label] for label in group]] = 1 / len(group)
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
            correct += (target.gather(1, logits.argmax(dim=-1, keepdim=True)) > 0).sum().item()
            total_loss += loss.item() * len(batch)
            progress(f"Epoch {epoch}/{args.epochs}", min(i + args.batch_size, len(items)), len(items))
        log(f"Epoch {epoch}: loss {total_loss / max(seen, 1):.3f}, train accuracy {correct * 100 // max(seen, 1)}%")


def save_trained(model, processor, head, labels):
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
    task = "4" if args.skip_labeling else "2" if args.corrections else args.task
    if task is None:
        print(f"1) Sort unclassed images ({len(unlabeled_images(root))})")
        print("2) Fix wrong category (corrections file)")
        print(f"3) Resolve suspicious ({len(suspicious_images(root))})")
        print("4) Train only")
        task = input("Choose [1-4]: ").strip()
    if task == "1":
        label_images(root, unlabeled_images(root))
    elif task == "2":
        corrections_file = args.corrections or Path(input("Corrections file: ").strip().strip("'\""))
        label_images(root, apply_corrections(root, parse_corrections(corrections_file.expanduser()), Gallery(None, args.model)))
    elif task == "3":
        label_images(root, suspicious_images(root))
    elif task != "4":
        raise SystemExit("Choose 1-4")
    if task != "4" and input("Train now? [Y/n]: ").strip().lower() not in ("", "y", "yes"):
        return
    items = [(p, [label]) for p, label in labeled_images(root)]
    items += [(root / rel, [tuple(l) for l in group]) for rel, group in load_multi(root).items() if (root / rel).exists()]
    if len({label for _, group in items for label in group}) < 2:
        raise SystemExit("Need at least 2 characters in <Series>/<Character>/ folders to train")
    train(items, args, pick_device())


if __name__ == "__main__":
    main()
