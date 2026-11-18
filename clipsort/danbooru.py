"""Danbooru API calls."""
import io
import json
from PIL import Image
from urllib.parse import urlencode
from urllib.request import Request
from urllib.request import urlopen

from .config import DANBOORU, DANBOORU_HEADERS, DANBOORU_MAX_PAGE
from .logs import log, progress


def danbooru_related(query, category, limit):
    params = urlencode({"query": query, "category": category, "limit": limit})
    with urlopen(Request(f"{DANBOORU}/related_tag.json?{params}", headers=DANBOORU_HEADERS), timeout=10) as r:
        return [t["tag"]["name"] for t in json.load(r)["related_tags"]]


def danbooru_top_tags(category, count, stage, min_posts=0):
    """Most-used tags of a category. Danbooru refuses pages past 1000, and the list
    ends long before that for characters, so stop at the first short page. A failed
    page keeps what was already fetched instead of throwing it all away."""
    tags, pages = [], min(-(-count // 1000), DANBOORU_MAX_PAGE)
    for page in range(1, pages + 1):
        progress(stage, page - 1, pages)
        params = urlencode({"search[category]": category, "search[order]": "count", "search[post_count]": f">={min_posts}",
                            "limit": 1000, "page": page, "only": "name"})
        try:
            with urlopen(Request(f"{DANBOORU}/tags.json?{params}", headers=DANBOORU_HEADERS), timeout=30) as r:
                batch = [t["name"] for t in json.load(r)]
        except Exception as e:
            if not tags:
                raise
            log(f"Danbooru page {page} failed, keeping the {len(tags)} tags fetched so far: {e}")
            break
        tags += batch
        if len(batch) < 1000:
            break
    progress(stage, pages, pages)
    return tags[:count]


def danbooru_find_tags(name, category=4):
    found = []
    for pattern in (f"{name}*", f"*{name}*"):
        params = urlencode({"search[category]": category, "search[name_matches]": pattern, "search[order]": "count", "limit": 10, "only": "name,post_count"})
        with urlopen(Request(f"{DANBOORU}/tags.json?{params}", headers=DANBOORU_HEADERS), timeout=15) as r:
            found = [(t["name"], t["post_count"]) for t in json.load(r)]
        if found:
            break
    return found


def danbooru_tag_exists(name, category):
    params = urlencode({"search[name]": name, "search[category]": category, "only": "name"})
    with urlopen(Request(f"{DANBOORU}/tags.json?{params}", headers=DANBOORU_HEADERS), timeout=15) as r:
        return bool(json.load(r))


def danbooru_reference_images(tag, count):
    try:
        params = urlencode({"tags": f"{tag} solo", "limit": 20})
        with urlopen(Request(f"{DANBOORU}/posts.json?{params}", headers=DANBOORU_HEADERS), timeout=15) as r:
            posts = json.load(r)
    except Exception as e:
        log(f"Reference lookup failed for {tag}: {e}")
        return None
    images = []
    for post in posts:
        if post.get("rating") not in ("g", "s"):
            continue
        variants = post.get("media_asset", {}).get("variants", [])
        url = next((v["url"] for v in variants if v["type"] == "360x360"), None) or post.get("preview_file_url")
        if not url:
            continue
        try:
            with urlopen(Request(url, headers=DANBOORU_HEADERS), timeout=15) as r:
                images.append(Image.open(io.BytesIO(r.read())).convert("RGB"))
        except Exception:
            continue
        if len(images) >= count:
            break
    return images
