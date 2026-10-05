"""Fallback character pool and series-guided search."""
import os
import torch
from pathlib import Path

from .config import EXPAND_LIMIT, REF_SHORTLIST
from .danbooru import danbooru_related, danbooru_top_tags
from .logs import log, progress
from .names import describe, legalize, title_name


def build_pool(sorter, model_id, name, category, count, describe_fn):
    if count <= 0:
        return None
    cache_file = Path(os.environ["HF_HOME"]) / f"fallback_{name}.pt"
    if cache_file.exists():
        cache = torch.load(cache_file)
        if cache["model"] == model_id and cache.get("count") == count:
            return cache["tags"], cache["embs"]
    log(f"Fetching top {count} Danbooru {name}...")
    try:
        tags = danbooru_top_tags(category, count, f"Fetching Danbooru {name} pages")
    except Exception as e:
        log(f"Danbooru {name} fetch failed, skipping: {e}")
        return None
    unique = {}
    for tag in tags:
        unique.setdefault(describe_fn(tag), tag)
    prompts, tags = list(unique), list(unique.values())
    embs = []
    for i in range(0, len(prompts), 64):
        embs.append(sorter.embed_texts(prompts[i:i + 64]).cpu())
        progress(f"Embedding {name} (one-time)", min(i + 64, len(prompts)), len(prompts))
    embs = torch.cat(embs)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model_id, "count": count, "tags": tags, "embs": embs}, cache_file)
    return tags, embs


class Expander:
    def __init__(self, sorter, model_id, series_pool, fallback):
        self.sorter, self.model_id, self.series_pool = sorter, model_id, series_pool
        self.cache_file = Path(os.environ["HF_HOME"]) / "expand_cache.pt"
        cache = torch.load(self.cache_file) if self.cache_file.exists() else {}
        same = cache.get("model") == model_id
        self.queries = cache.get("queries", {})
        self.embs = cache.get("embs", {}) if same else {}
        self.pool_tags, self.pool_embs = list(fallback[0]), fallback[1]
        self.pool_desc = set(map(describe, self.pool_tags))
        self.grow(list(self.embs))

    def grow(self, tags):
        new = []
        for tag in tags:
            desc = describe(tag)
            if desc not in self.pool_desc:
                self.pool_desc.add(desc)
                new.append(tag)
        if new:
            self.pool_tags += new
            self.pool_embs = torch.cat([self.pool_embs, self.embed(new)])
            log(f"Fallback pool grew by {len(new)} to {len(self.pool_tags)} characters")

    def candidates(self, query):
        if query not in self.queries:
            try:
                self.queries[query] = danbooru_related(query, "character", EXPAND_LIMIT)
            except Exception as e:
                log(f"Series-guided search failed for {query}: {e}")
                return []
        return self.queries[query]

    def embed(self, tags):
        new = [t for t in tags if t not in self.embs]
        for i in range(0, len(new), 64):
            batch = new[i:i + 64]
            self.embs.update(zip(batch, self.sorter.embed_texts([describe(t) for t in batch]).cpu()))
        return torch.stack([self.embs[t] for t in tags])

    def resolve(self, image_embs, appearance, threshold, expand_threshold, tagger):
        series_tags, series_embs = self.series_pool
        conf, idx = self.sorter.probs(image_embs, series_embs).max(dim=-1)
        results, discovered = [], []
        for n, (emb, c, j, looks) in enumerate(zip(image_embs, conf.tolist(), idx.tolist(), appearance), 1):
            progress("Series-guided search", n, len(image_embs))
            if c < threshold:
                results.append((None, None, []))
                continue
            series = series_tags[j]
            tags = self.candidates(" ".join([series, *looks]))
            discovered += tags
            match, top = None, []
            if tags:
                probs = self.sorter.probs(emb[None], self.embed(tags))[0]
                pc, pj = probs.max(dim=-1)
                top = [tags[k] for k in probs.topk(min(REF_SHORTLIST, len(tags))).indices.tolist()]
                if pc.item() >= expand_threshold:
                    match = (tagger.label(tags[pj.item()]), pc.item())
            results.append((match, legalize(title_name(series)), top))
        self.grow(discovered)
        return results

    def save(self):
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": self.model_id, "queries": self.queries, "embs": self.embs}, self.cache_file)
