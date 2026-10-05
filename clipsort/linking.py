"""Danbooru linking for new characters."""
import json
import torch

from .config import LINKED_FILE, LINK_IMAGES, QUALIFIER, TRAINED_DIR
from .danbooru import danbooru_find_tags, danbooru_reference_images, danbooru_tag_exists
from .gallery import Gallery
from .logs import log
from .names import base_name, describe, legalize, pretty, title_name
from .records import labeled_images, load_aliases, save_aliases
from .siglip import Sorter, pick_device


def ask_tag(prompt, found, category):
    options = "  ".join(f"{i}) {name} ({count})" for i, (name, count) in enumerate(found[:5], 1)) or "no matches found"
    while True:
        answer = input(f"  {prompt}: {options}  [number / type a Danbooru tag / Enter = none]: ").strip()
        if not answer:
            return None
        if answer.isdigit() and 1 <= int(answer) <= min(5, len(found)):
            return found[int(answer) - 1][0]
        tag = answer.lower().replace(" ", "_")
        try:
            if danbooru_tag_exists(tag, category):
                return tag
            print(f"  '{tag}' is not a Danbooru {'series' if category == 3 else 'character'} tag, try again")
        except Exception as e:
            print(f"  Could not check '{tag}' on Danbooru ({e}), try again")


def choose_tag(title, char):
    found = danbooru_find_tags(char)
    exact = [name for name, _ in found if legalize(base_name(name)).lower() == char]
    same_series = [name for name in exact if any(legalize(title_name(q)).lower() == title for q in QUALIFIER.findall(name))]
    if same_series:
        return same_series[0]
    if len(exact) == 1:
        return exact[0]
    return ask_tag(f"Danbooru tag for {pretty(char)} ({pretty(title)})", found, 4)


def add_to_pool(sorter, model_id, tag):
    cache_file = TRAINED_DIR.parent / "expand_cache.pt"
    cache = torch.load(cache_file) if cache_file.exists() else {}
    if cache.get("model") != model_id:
        cache = {"model": model_id, "queries": cache.get("queries", {}), "embs": {}}
    cache["embs"][tag] = sorter.embed_texts([describe(tag)])[0].cpu()
    torch.save(cache, cache_file)


def link_new_characters(root, model_id, relink=False):
    linked = json.loads(LINKED_FILE.read_text(encoding="utf-8")) if LINKED_FILE.exists() else {}
    labels = sorted({label for _, label in labeled_images(root)})
    new = [label for label in labels if "|".join(label) not in linked or (relink and linked["|".join(label)] is None)]
    if not new:
        return
    log(f"Linking {len(new)} new characters to Danbooru...")
    sorter = Sorter(model_id, pick_device())
    gallery, aliases = Gallery(sorter, model_id), load_aliases()
    for n, (title, char) in enumerate(new, 1):
        try:
            tag = choose_tag(title, char)
            images = danbooru_reference_images(tag, LINK_IMAGES) if tag else []
        except Exception as e:
            log(f"[{n}/{len(new)}] {pretty(char)}: Danbooru lookup failed, will retry next time: {e}")
            continue
        linked["|".join((title, char))] = tag
        if not tag:
            log(f"[{n}/{len(new)}] {pretty(char)} ({pretty(title)}): not found on Danbooru, learning from your images only")
            continue
        aliases[tag] = [title, char]
        add_to_pool(sorter, model_id, tag)
        if images:
            gallery.add(sorter.embed_pil(images), [(title, char)] * len(images))
            gallery.fetched.add(tag)
        log(f"[{n}/{len(new)}] {pretty(char)} ({pretty(title)}) -> {tag}, {len(images or [])} reference pictures")
        LINKED_FILE.write_text(json.dumps(linked, indent=1, ensure_ascii=False), encoding="utf-8")
        save_aliases(aliases)
    LINKED_FILE.write_text(json.dumps(linked, indent=1, ensure_ascii=False), encoding="utf-8")
    save_aliases(aliases)
    gallery.save()
    del sorter
    torch.cuda.empty_cache()
