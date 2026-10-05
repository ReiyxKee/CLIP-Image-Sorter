"""Labeling window and suggestions."""
import json
import os
import queue
import shutil
import threading
import tkinter as tk
import torch
from PIL import Image
from PIL import ImageTk
from difflib import get_close_matches
from pathlib import Path

from .config import ANIME, FALLBACK_COUNT, MULTI_FILE, QUALIFIER, SERIES_COUNT, TAGGER_ID, TRAINED_DIR
from .corrections import reindex
from .files import collect_images, original_stem, remove_empty_dirs, transfer_unique
from .gallery import Gallery
from .logs import log
from .names import base_name, describe, describe_series, legalize, pretty, title_name
from .pools import Expander, build_pool
from .records import load_sus, save_sus
from .siglip import Sorter, pick_device
from .tagger import Tagger
from .trained import TrainedModel


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


class Suggester:
    def __init__(self, model_id):
        device = pick_device()
        log(f"Loading suggestion models on {device}...")
        self.lock = threading.Lock()
        self.sorter = Sorter(model_id, device)
        self.tagger = Tagger(TAGGER_ID, device)
        pool_file = Path(os.environ["HF_HOME"]) / "fallback_characters.pt"
        count = torch.load(pool_file).get("count", FALLBACK_COUNT) if pool_file.exists() else FALLBACK_COUNT
        pool = build_pool(self.sorter, model_id, "characters", 4, count, describe)
        series_pool = build_pool(self.sorter, model_id, "series", 3, SERIES_COUNT if pool else 0, describe_series)
        self.expander = Expander(self.sorter, model_id, series_pool, pool) if series_pool else None
        self.pool = (self.expander.pool_tags, self.expander.pool_embs) if self.expander else pool
        self.gallery = Gallery(self.sorter, model_id)
        trained_ready = all((TRAINED_DIR / f).exists() for f in ("config.json", "head.safetensors", "labels.json"))
        self.trained = TrainedModel(TRAINED_DIR, device) if trained_ready else None

    def suggest(self, path):
        with self.lock:
            _, embs = self.sorter.embed_images([path], 1)
            found = {}

            def add(label, score, source):
                if label[1] and score > found.get(label, (0, ""))[0]:
                    found[label] = (score, source)

            looks, top_tags = self.tagger.appearance_and_top(path, 3)
            for tag, score in top_tags:
                add(self.tagger.label(tag), score, "tagger")
            if self.trained:
                for label, score in self.trained.top(path, 3):
                    add(label, score, "trained")
            for label, score in self.gallery.top(embs[0], 3):
                add(label, score, "gallery")
            shortlist, series = [tag for tag, _ in top_tags], None
            if self.pool:
                tags, pool_embs = self.pool
                conf, idx = self.sorter.probs(embs, pool_embs)[0].topk(5)
                for score, j in zip(conf.tolist(), idx.tolist()):
                    add(self.tagger.label(tags[j]), score, "name")
                    shortlist.append(tags[j])
            if self.expander:
                _, series, top = self.expander.resolve(embs, [looks], 0.0, 2.0, self.tagger)[0]
                shortlist += top
            self.gallery.fetch_refs(shortlist, self.tagger)
            for label, score in self.gallery.top(embs[0], 3):
                add(label, score, "reference")
            ranked = sorted(found.items(), key=lambda item: -item[1][0])[:6]
            return [(label, score, source) for label, (score, source) in ranked], series

    def close(self):
        self.gallery.save()
        self.tagger.save_cache()
        if self.expander:
            self.expander.save()


def target_for(path, anime_dir, labels):
    if len(labels) == 1:
        title, slug = labels[0]
        return anime_dir / pretty(title) / pretty(slug), f"{slug}_{original_stem(path)[:10]}"
    titles = {t for t, _ in labels}
    return (anime_dir / pretty(titles.pop()) if len(titles) == 1 else anime_dir), original_stem(path)


def file_image(root, path, anime_dir, labels, multi):
    target, stem = target_for(path, anime_dir, labels)
    dest = transfer_unique(path, target, "move", stem)
    reindex(root, path, dest)
    multi.pop(path.relative_to(root).as_posix(), None)
    if len(labels) > 1:
        multi[dest.relative_to(root).as_posix()] = [list(l) for l in labels]
    save_multi(root, multi)
    return dest


def to_slugs(char, series):
    return legalize(title_name(series)).lower(), legalize(base_name(char)).lower()


class LabelApp:
    def __init__(self, root, items, suggester):
        self.root, self.items, self.index, self.suggester = root, items, 0, suggester
        self.results = queue.Queue()
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
        self.suggestions = tk.Frame(panel)
        self.suggestions.pack(fill="x", pady=4)
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
        self.win.after(200, self.poll_suggestions)
        self.win.mainloop()

    def request_suggestions(self, path):
        for widget in self.suggestions.winfo_children():
            widget.destroy()
        if not self.suggester:
            return
        tk.Label(self.suggestions, text="Looking up suggestions...", fg="gray").pack(anchor="w")
        token = self.index

        def work():
            try:
                self.results.put((token, self.suggester.suggest(path)))
            except Exception as e:
                self.results.put((token, e))

        threading.Thread(target=work, daemon=True).start()

    def poll_suggestions(self):
        while not self.results.empty():
            token, result = self.results.get()
            if token == self.index:
                self.show_suggestions(result)
        self.win.after(200, self.poll_suggestions)

    def show_suggestions(self, result):
        for widget in self.suggestions.winfo_children():
            widget.destroy()
        if isinstance(result, Exception):
            tk.Label(self.suggestions, text=f"Suggestions failed: {result}", fg="gray", wraplength=380, justify="left").pack(anchor="w")
            return
        found, series = result
        tk.Label(self.suggestions, text="Suggestions:" + (f"  (series guess: {pretty(series)})" if series else "")).pack(anchor="w")
        if not found:
            tk.Label(self.suggestions, text="none", fg="gray").pack(anchor="w")
        for (title, slug), score, source in found:
            tk.Button(self.suggestions, text=f"{pretty(slug)} ({pretty(title)})  {source} {score:.2f}",
                      command=lambda t=title, c=slug: self.use_choice(t, c)).pack(anchor="w")

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
        self.request_suggestions(path)

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
                reindex(self.root, dest, original)
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


def label_images(root, items, args):
    log(f"{len(items)} images to label, opening labeling window...")
    if items:
        suggester = None if args.no_suggest else Suggester(args.model)
        LabelApp(root, items, suggester)
        if suggester:
            suggester.close()
            del suggester
            torch.cuda.empty_cache()
    remove_empty_dirs(root)
