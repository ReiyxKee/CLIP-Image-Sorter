"""WD Tagger."""
import csv
import json
import os
import timm
import torch
from PIL import Image
from huggingface_hub import hf_hub_download
from pathlib import Path
from timm.data import create_transform
from timm.data import resolve_data_config

from .config import APPEARANCE_THRESHOLD, COLORS, QUALIFIER, UNKNOWN_TITLE
from .danbooru import danbooru_related
from .logs import log, progress
from .names import base_name, legalize, title_name
from .records import load_aliases, load_overrides


class Tagger:
    def __init__(self, repo_id, device):
        self.device = device
        self.model = timm.create_model(f"hf-hub:{repo_id}", pretrained=True).to(device).eval()
        self.transform = create_transform(**resolve_data_config(self.model.pretrained_cfg, model=self.model))
        with open(hf_hub_download(repo_id, "selected_tags.csv"), newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.char_idx = torch.tensor([i for i, r in enumerate(rows) if r["category"] == "4"])
        self.char_names = [rows[i]["name"] for i in self.char_idx.tolist()]
        self.cache_file = Path(os.environ["HF_HOME"]) / "series_cache.json"
        self.series_cache = json.loads(self.cache_file.read_text(encoding="utf-8")) if self.cache_file.exists() else {}
        self.overrides = load_overrides()
        self.aliases = load_aliases()
        self.explicit_idx = next(i for i, r in enumerate(rows) if r["category"] == "9" and r["name"] == "explicit")
        general = {r["name"]: i for i, r in enumerate(rows) if r["category"] == "0"}
        self.looks = [[(f"{c}_{part}", general[f"{c}_{part}"]) for c in COLORS if f"{c}_{part}" in general] for part in ("hair", "eyes")]

    def series(self, char):
        if char not in self.series_cache:
            try:
                related = danbooru_related(char, "copyright", 1)
            except Exception as e:
                log(f"Series lookup failed for {char}: {e}")
                return None
            self.series_cache[char] = related[0] if related else None
        return self.series_cache[char]

    def label(self, char):
        if char in self.aliases:
            return tuple(self.aliases[char])
        slug = legalize(base_name(char))
        if slug in self.overrides:
            return self.overrides[slug], slug
        qualifiers = QUALIFIER.findall(char)
        series = self.series(char) or (qualifiers[-1] if qualifiers else None)
        return legalize(title_name(series)) if series else UNKNOWN_TITLE, slug

    def save_cache(self):
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.cache_file.write_text(json.dumps(self.series_cache, indent=1, sort_keys=True), encoding="utf-8")

    @staticmethod
    def pad_square(image):
        side = max(image.size)
        canvas = Image.new("RGB", (side, side), (255, 255, 255))
        canvas.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
        return canvas

    @torch.no_grad()
    def probabilities(self, paths):
        batch = torch.stack([self.transform(self.pad_square(Image.open(p).convert("RGB"))) for p in paths])
        return self.model(batch[:, [2, 1, 0]].to(self.device)).sigmoid().cpu()

    @torch.no_grad()
    def appearance_and_top(self, path, k):
        row = self.probabilities([path])[0]
        best = [max(group, key=lambda t: row[t[1]]) for group in self.looks if group]
        looks = [name for name, idx in best if row[idx] >= APPEARANCE_THRESHOLD]
        conf, idx = row[self.char_idx].topk(k)
        return looks, [(self.char_names[j], c) for c, j in zip(conf.tolist(), idx.tolist())]

    @torch.no_grad()
    def tag(self, paths, batch_size, threshold, r18_threshold):
        labels, r18, appearance, multi = [], [], [], []
        for i in range(0, len(paths), batch_size):
            probs = self.probabilities(paths[i:i + batch_size])
            for row in probs[:, self.char_idx]:
                hits = sorted((row >= threshold).nonzero().flatten().tolist(), key=lambda j: -row[j])
                found = {}
                for j in hits:
                    title, char = self.label(self.char_names[j])
                    found.setdefault(char, title)
                titles = set(found.values())
                multi.append(len(found) > 1)
                if len(found) <= 1:
                    labels.append(next(((t, c) for c, t in found.items()), (None, None)))
                else:
                    labels.append((titles.pop(), None) if len(titles) == 1 else (None, None))
            r18 += (probs[:, self.explicit_idx] >= r18_threshold).tolist()
            for row in probs:
                best = [max(group, key=lambda t: row[t[1]]) for group in self.looks if group]
                appearance.append([name for name, idx in best if row[idx] >= APPEARANCE_THRESHOLD])
            progress("Tagging anime characters", min(i + batch_size, len(paths)), len(paths))
        return labels, r18, appearance, multi
