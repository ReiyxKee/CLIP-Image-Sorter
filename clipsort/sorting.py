"""Per-chunk sorting flow."""
from .config import ANIME, CATEGORY_PROMPTS, R18, REAL, REF_SHORTLIST, SERIES_THRESHOLD, SUS_FILE, UNKNOWN_TITLE
from .files import original_stem, transfer_unique
from .logs import log
from .names import pretty
from .nsfw import detect_nsfw
from .records import append_index, load_sus, save_sus
from .siglip import classify_categories


def conflicts(trained, other):
    titles = {t.lower() for t in (trained[0], other[0]) if t and t.lower() != UNKNOWN_TITLE.lower()}
    return bool(other[1] and other[1] != trained[1]) or len(titles) > 1


def resolve_unknown(unknown, embs, appearance, characters, sources, models, args, confident):
    sorter, _, _, tagger, fallback, expander, gallery, _ = models

    def accept(indices, matches, name):
        for i, m in zip(indices, matches):
            if m:
                characters[i], sources[i] = m[0], f"{name} {m[1]:.2f}"
        return [i for i in indices if characters[i][1] is None]

    if gallery and unknown:
        unknown = accept(unknown, gallery.match(embs[unknown], args.gallery_threshold), "gallery")
    pool = (expander.pool_tags, expander.pool_embs) if expander else fallback
    shortlist, series_titles = {i: [] for i in unknown}, {}
    if pool and unknown:
        pool_tags, pool_embs = pool
        probs = sorter.probs(embs[unknown], pool_embs)
        conf, idx = probs.max(dim=-1)
        for i, row in zip(unknown, probs.topk(min(REF_SHORTLIST, len(pool_tags)), dim=-1).indices.tolist()):
            shortlist[i] = [pool_tags[j] for j in row]
        unknown = accept(unknown, [(tagger.label(pool_tags[j]), c) if c >= args.fallback_threshold else None for c, j in zip(conf.tolist(), idx.tolist())], "name")

    unknown = [i for i in unknown if i not in confident]
    if expander and unknown:
        found = expander.resolve(embs[unknown], [appearance[i] for i in unknown], SERIES_THRESHOLD, args.expand_threshold, tagger)
        for i, (_, series, top) in zip(unknown, found):
            shortlist[i] += top
            if series:
                series_titles[i] = series
        unknown = accept(unknown, [m for m, _, _ in found], "series-search")

    if gallery and unknown:
        gallery.fetch_refs([t for i in unknown for t in shortlist[i]], tagger)
        unknown = accept(unknown, gallery.match(embs[unknown], args.gallery_threshold), "reference")

    for i in unknown:
        if i in series_titles:
            characters[i], sources[i] = (series_titles[i], None), "series only"


def process_chunk(paths, models, args, output, label, done_file):
    sorter, class_embs, nsfw, tagger, fallback, expander, gallery, trained = models
    paths, embs = sorter.embed_images(paths, args.batch_size)
    if not paths:
        return [], []
    categories, confs, probs = classify_categories(sorter, embs, class_embs, args.category_threshold)

    r18 = detect_nsfw(nsfw, paths, args.batch_size, args.nsfw_threshold) if nsfw else [False] * len(paths)
    names = list(CATEGORY_PROMPTS)
    real_i, anime_i = names.index(REAL), names.index(ANIME)
    for i in (i for i, flag in enumerate(r18) if flag):
        categories[i] = ANIME if probs[i, anime_i] > probs[i, real_i] else REAL

    characters = [(None, None)] * len(paths)
    sources = [""] * len(paths)
    suspicious = {}
    anime = [i for i, c in enumerate(categories) if c == ANIME]
    if anime:
        labels, flags, looks, multi = tagger.tag([paths[i] for i in anime], args.batch_size, args.character_threshold, args.nsfw_threshold)
        appearance = dict(zip(anime, looks))
        for i, char, flag in zip(anime, labels, flags):
            characters[i] = char
            sources[i] = "tagger" if char[1] else ""
            r18[i] = r18[i] or (flag and nsfw is not None)
        singles = [i for i, m in zip(anime, multi) if not m]
        confident = {}
        if trained and singles:
            for i, (char, c) in zip(singles, trained.predict([paths[i] for i in singles], args.batch_size)):
                if c >= args.trained_threshold:
                    confident[i] = (char, c)
        unknown = [i for i in singles if characters[i][1] is None]
        resolve_unknown(unknown, embs, appearance, characters, sources, models, args, confident)
        for i, (char, c) in confident.items():
            other, other_source = characters[i], sources[i]
            if conflicts(char, other):
                suspicious[i] = {"trained": list(char), "other": list(other), "source": other_source, "score": round(c, 3)}
                sources[i] = f"trained {c:.2f}, SUSPICIOUS vs {other_source}"
            else:
                sources[i] = f"trained {c:.2f}" + (f" + {other_source}" if other_source else "")
            characters[i] = char
        known = [i for i in singles if sources[i].startswith("tagger") or ("+ tagger" in sources[i])]
        if gallery and known:
            gallery.add(embs[known], [characters[i] for i in known])

    sus_entries = load_sus() if suspicious else []
    for n, (path, cat, conf, (title, char), flag, src) in enumerate(zip(paths, categories, confs, characters, r18, sources), 1):
        try:
            dest = ((output / R18 if flag else output) / cat).joinpath(*(pretty(x) for x in (title, char) if x))
            original = original_stem(path) if args.mode == "recategorize" else path.stem
            stem = f"{char}_{original[:10]}" if char else original
            dest = transfer_unique(path, dest, args.mode, stem)
            append_index(output, dest, path, (title, char))
            if done_file:
                with done_file.open("a", encoding="utf-8") as f:
                    f.write(f"{path}\n")
            if n - 1 in suspicious:
                sus_entries = [e for e in sus_entries if e["path"] != str(dest)] + [{"path": str(dest), **suspicious[n - 1]}]
            log(f"[{label} | {n}/{len(paths)}] {path.name} -> {dest.relative_to(output)} ({conf:.2f}{', ' + src if src else ''})")
        except Exception as e:
            log(f"[{label} | {n}/{len(paths)}] Skip {path.name}: {e}")
    if suspicious:
        save_sus(sus_entries)
        log(f"{len(suspicious)} suspicious images recorded in {SUS_FILE}")
    return categories, r18
