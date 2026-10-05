"""Reference gallery and learn mode."""
import os
import torch
from pathlib import Path

from .config import GALLERY_MARGIN, REF_IMAGES
from .danbooru import danbooru_reference_images
from .logs import log, progress
from .records import labeled_images


class Gallery:
    def __init__(self, sorter, model_id):
        self.sorter, self.model_id = sorter, model_id
        self.cache_file = Path(os.environ["HF_HOME"]) / "gallery.pt"
        cache = torch.load(self.cache_file) if self.cache_file.exists() else {}
        if cache.get("model") != model_id:
            cache = {}
        self.embs = cache.get("embs")
        self.labels = [tuple(l) for l in cache.get("labels", [])]
        self.fetched = set(cache.get("fetched", []))

    def add(self, embs, labels):
        if self.embs is not None and labels:
            keep = (embs @ self.embs.T).max(dim=-1).values < 0.995
            embs, labels = embs[keep], [l for l, k in zip(labels, keep.tolist()) if k]
        if labels:
            self.embs = embs if self.embs is None else torch.cat([self.embs, embs])
            self.labels += labels

    def top(self, emb, k):
        if self.embs is None:
            return []
        sims, idx = (self.embs @ emb).topk(min(50, len(self.labels)))
        found = {}
        for sim, j in zip(sims.tolist(), idx.tolist()):
            found.setdefault(self.labels[j], sim)
        return list(found.items())[:k]

    def match(self, embs, threshold, allowed=None):
        if self.embs is None:
            return [None] * len(embs)
        all_sims = embs @ self.embs.T
        if allowed is not None:
            mask = torch.tensor([l in allowed for l in self.labels])
            if not mask.any():
                return [None] * len(embs)
            all_sims = all_sims.masked_fill(~mask, -1.0)
        sims, idx = all_sims.topk(min(20, len(self.labels)), dim=-1)
        results = []
        for row_sims, row_idx in zip(sims.tolist(), idx.tolist()):
            best = self.labels[row_idx[0]]
            rival = next((s for s, j in zip(row_sims, row_idx) if self.labels[j] != best), 0.0)
            ok = row_sims[0] >= threshold and row_sims[0] - rival >= GALLERY_MARGIN
            results.append((best, row_sims[0]) if ok else None)
        return results

    def fetch_refs(self, tags, tagger):
        new = [t for t in dict.fromkeys(tags) if t not in self.fetched]
        for n, tag in enumerate(new, 1):
            progress("Fetching reference pictures", n, len(new))
            images = danbooru_reference_images(tag, REF_IMAGES)
            if images is None:
                continue
            self.fetched.add(tag)
            if images:
                self.add(self.sorter.embed_pil(images), [tagger.label(tag)] * len(images))

    def relabel_near(self, emb, new):
        if self.embs is None:
            return 0
        hits = ((self.embs @ emb) >= 0.995).tolist()
        if new:
            self.labels = [new if h else l for l, h in zip(self.labels, hits)]
        elif any(hits):
            keep = [not h for h in hits]
            self.embs = self.embs[torch.tensor(keep)] if any(keep) else None
            self.labels = [l for l, k in zip(self.labels, keep) if k]
        return sum(hits)

    def relabel(self, old, new):
        old = (old[0].lower(), old[1].lower())
        hits = [(l[0].lower(), l[1].lower()) == old for l in self.labels]
        if new:
            self.labels = [new if h else l for l, h in zip(self.labels, hits)]
        elif any(hits):
            keep = [not h for h in hits]
            self.embs = self.embs[torch.tensor(keep)] if any(keep) else None
            self.labels = [l for l, k in zip(self.labels, keep) if k]
        return sum(hits)

    def save(self):
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": self.model_id, "embs": self.embs, "labels": [list(l) for l in self.labels], "fetched": sorted(self.fetched)}, self.cache_file)


def learn_from_folder(root, sorter, gallery, batch_size):
    label_of = dict(labeled_images(root))
    paths, labels = list(label_of), list(label_of.values())
    kept, embs = sorter.embed_images(paths, batch_size)
    before = len(gallery.labels)
    gallery.add(embs, [label_of[p] for p in kept])
    gallery.save()
    log(f"Learned {len(gallery.labels) - before} new pictures across {len(set(labels))} characters (gallery: {len(gallery.labels)})")
