import argparse
import csv
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

DEPENDENCIES = {
    "torch": "torch",
    "transformers": "transformers",
    "timm": "timm",
    "PIL": "pillow",
    "pillow_heif": "pillow-heif",
    "sentencepiece": "sentencepiece",
}
TORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu128"


def install_cuda_torch(*extra):
    subprocess.check_call([sys.executable, "-m", "pip", "install", *extra, "torch", "torchvision", "--index-url", TORCH_CUDA_INDEX])


def ensure_dependencies():
    missing = [pkg for mod, pkg in DEPENDENCIES.items() if importlib.util.find_spec(mod) is None]
    nvidia = shutil.which("nvidia-smi") is not None
    if missing:
        answer = input(f"Missing packages: {' '.join(missing)}. Install now? [y/N]: ").strip().lower()
        if answer not in ("y", "yes"):
            raise SystemExit(f"Install manually: {sys.executable} -m pip install -U {' '.join(missing)}")
        if nvidia and "torch" in missing:
            install_cuda_torch()
            missing.remove("torch")
        if missing:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", *missing])
    if not nvidia:
        return
    import torch
    if torch.version.cuda:
        return
    answer = input("NVIDIA GPU found but torch is CPU-only. Reinstall torch with CUDA? [y/N]: ").strip().lower()
    if answer in ("y", "yes"):
        install_cuda_torch("--force-reinstall", "--no-deps")
        raise SystemExit(subprocess.call([sys.executable, *sys.argv]))


sys.stdout.reconfigure(errors="replace")
ensure_dependencies()
HF_GLOBAL_HOME = Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")
os.environ.setdefault("HF_TOKEN_PATH", str(HF_GLOBAL_HOME / "token"))
GLOBAL_ENV = dict(os.environ)
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent / "models"))

import timm
import torch
import torch.nn.functional as F
from huggingface_hub import get_token, hf_hub_download
from PIL import Image
from pillow_heif import register_heif_opener
from timm.data import create_transform, resolve_data_config
from safetensors.torch import load_file
from transformers import AutoModel, AutoProcessor, pipeline

register_heif_opener()

MODEL_ID = "google/siglip2-so400m-patch14-384"
EXTENSIONS = {
    ".jpg", ".jpeg", ".jpe", ".jfif", ".png", ".webp", ".bmp", ".gif",
    ".tif", ".tiff", ".heic", ".heif", ".avif", ".ico", ".tga",
}
TAGGER_ID = "SmilingWolf/wd-eva02-large-tagger-v3"
NSFW_ID = "Falconsai/nsfw_image_detection"
CHUNK_SIZE = 500
NSFW_THRESHOLD = 0.85
R18 = "R18"
REAL = "Real_Photograph"
ANIME = "Anime_Handdrawn"
UNCATEGORIZABLE = "Uncategorizable"
AMBIGUOUS = "Ambiguous"
UNKNOWN_TITLE = "Unknown_Title"
DANBOORU = "https://danbooru.donmai.us"
DANBOORU_HEADERS = {"User-Agent": "clip-sorter/1.0"}
QUALIFIER = re.compile(r"\(([^()]*)\)")
FALLBACK_COUNT = 20000
SERIES_COUNT = 5000
EXPAND_LIMIT = 100
EXPAND_THRESHOLD = 0.7
APPEARANCE_THRESHOLD = 0.35
GALLERY_THRESHOLD = 0.85
TRAINED_THRESHOLD = 0.9
TRAINED_DIR = Path(os.environ["HF_HOME"]) / "finetuned"
SUS_FILE = Path(os.environ["HF_HOME"]) / "suspicious.json"
REF_SHORTLIST = 5
REF_IMAGES = 5
COLORS = ("aqua", "black", "blonde", "blue", "brown", "green", "grey", "orange", "pink", "purple", "red", "silver", "white", "yellow")
FALLBACK_THRESHOLD = 0.5

CATEGORY_PROMPTS = {
    REAL: [
        "a real photograph of a person taken with a camera",
        "a candid photo of a real human",
        "a professional portrait photograph of a real person",
        "a selfie of a real person",
        "a cosplay photograph of a real person in costume",
        "a realistic photo with natural skin texture and real lighting",
        "a film still of a real actor",
    ],
    ANIME: [
        "an anime illustration of a character",
        "a manga drawing of a character",
        "a hand-drawn sketch of a person",
        "a digital painting of a character",
        "a cartoon drawing of a person",
        "cel-shaded 2d anime art",
        "a 3d rendered anime style character",
        "fan art of a fictional character",
        "a watercolor or pencil drawing of a person",
    ],
    "Non_Human": [
        "a photo of an animal",
        "a photo of a landscape with no people",
        "a photo of a building or city with no people",
        "a photo of food",
        "a photo of an object or product",
        "a photo of a car or vehicle",
        "a drawing of scenery with no characters",
        "an illustration of an animal or creature",
    ],
    UNCATEGORIZABLE: [
        "a screenshot of text or a document",
        "a screenshot of a user interface",
        "an abstract pattern or texture",
        "a blank or solid color image",
        "a meme with mostly text",
        "a chart, diagram or graph",
        "a heavily blurred or corrupted image",
        "a logo or icon",
    ],
}


def legalize(name):
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_") or AMBIGUOUS


def pretty(slug):
    return slug.replace("_", " ").title()


def base_name(tag):
    return re.sub(r"_?\([^()]*\)", "", tag)


def title_name(tag):
    return base_name(tag.split("/")[0])


def danbooru_related(query, category, limit):
    params = urlencode({"query": query, "category": category, "limit": limit})
    with urlopen(Request(f"{DANBOORU}/related_tag.json?{params}", headers=DANBOORU_HEADERS), timeout=10) as r:
        return [t["tag"]["name"] for t in json.load(r)["related_tags"]]


def danbooru_top_tags(category, count, stage):
    tags, pages = [], -(-count // 1000)
    for page in range(1, pages + 1):
        progress(stage, page - 1, pages)
        params = urlencode({"search[category]": category, "search[order]": "count", "limit": 1000, "page": page, "only": "name"})
        with urlopen(Request(f"{DANBOORU}/tags.json?{params}", headers=DANBOORU_HEADERS), timeout=30) as r:
            tags += [t["name"] for t in json.load(r)]
    progress(stage, pages, pages)
    return tags[:count]


def describe(tag):
    qualifiers = QUALIFIER.findall(tag)
    name = pretty(legalize(base_name(tag)))
    return f"an illustration of {name} from {pretty(legalize(title_name(qualifiers[-1])))}" if qualifiers else f"an illustration of {name}"


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


def describe_series(tag):
    return f"an illustration of a character from {pretty(legalize(base_name(tag)))}"


def find_hf_cli():
    cli_dir = Path(GLOBAL_ENV["HF_HOME"]) / "cli" if "HF_HOME" in GLOBAL_ENV else Path.home() / ".hf-cli"
    candidates = [Path.home() / ".local" / "bin" / "hf", cli_dir / "venv" / "bin" / "hf", cli_dir / "venv" / "Scripts" / "hf.exe"]
    found = shutil.which("hf", path=GLOBAL_ENV.get("PATH"))
    return Path(found) if found else next((c for c in candidates if c.exists()), None)


def install_hf_cli():
    if sys.platform == "win32":
        cmd = ["powershell", "-ExecutionPolicy", "ByPass", "-c", "irm https://hf.co/cli/install.ps1 | iex"]
    else:
        cmd = ["bash", "-c", "curl -LsSf https://hf.co/cli/install.sh | bash"]
    subprocess.check_call(cmd, env=GLOBAL_ENV)


def ensure_hf_login():
    if get_token():
        log("Hugging Face login found, proceeding.")
        return
    if input("Log in to Hugging Face (needed for gated models like PixAI)? [y/N]: ").strip().lower() not in ("y", "yes"):
        return
    cli = find_hf_cli()
    if cli is None:
        if input("Hugging Face CLI not found. Install it globally? [y/N]: ").strip().lower() not in ("y", "yes"):
            return
        install_hf_cli()
        cli = find_hf_cli()
        if cli is None:
            log("Hugging Face CLI install not found, skipping login.")
            return
    subprocess.call([str(cli), "auth", "login"], env=GLOBAL_ENV)


def log(msg):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}")


def progress(stage, done, total):
    print(f"\r[{datetime.now():%Y-%m-%d %H:%M:%S}] {stage} {done}/{total} ({done * 100 // max(total, 1)}%)", end="\n" if done >= total else "", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description="Sort images by style and character using SigLIP2 and WD Tagger.")
    p.add_argument("--target-path", type=Path, help="Folder (or single image) to categorize, or sorted folder to recategorize")
    p.add_argument("--output-path", type=Path, help="Folder to copy sorted images into")
    p.add_argument("--model", default=MODEL_ID)
    p.add_argument("--tagger", default=TAGGER_ID)
    p.add_argument("--nsfw-model", default=NSFW_ID)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--chunk-size", type=int, help="Images to fully sort and copy per cycle")
    p.add_argument("--fallback-count", type=int, help="Top Danbooru characters in the fallback pool, 0 disables")
    p.add_argument("--fallback-threshold", type=float, help="Fallback match cutoff 0-1, higher = stricter")
    p.add_argument("--expand-threshold", type=float, help="Series-guided search match cutoff 0-1, higher = stricter")
    p.add_argument("--trained-threshold", type=float, help="Trained model match cutoff 0-1, higher = stricter, -1 disables")
    p.add_argument("--gallery-threshold", type=float, help="Reference picture match cutoff 0-1, higher = stricter, -1 disables")
    p.add_argument("--mode", choices=("copy", "move", "recategorize", "learn"), help="Copy (default), move, re-sort an output folder in place, or learn from a sorted folder")
    p.add_argument("--category-threshold", type=float, default=0.45)
    p.add_argument("--character-threshold", type=float, default=0.85)
    p.add_argument("--nsfw-threshold", type=float, help="R18 cutoff 0-1, higher flags fewer images, -1 disables")
    return p.parse_args()


def resolve_paths(args):
    if args.mode is None:
        answer = input("Copy, move, recategorize, or learn from a sorted folder? [C/m/r/l]: ").strip().lower()
        args.mode = {"m": "move", "r": "recategorize", "l": "learn"}.get(answer, answer if answer in ("move", "recategorize", "learn") else "copy")
    recategorize = args.mode in ("recategorize", "learn")
    prompt = f"Sorted folder to {args.mode}: " if recategorize else "Folder to categorize: "
    target = args.target_path or (args.output_path if recategorize else None) or Path(input(prompt).strip().strip("'\""))
    target = target.expanduser().resolve()
    if not target.exists():
        raise SystemExit(f"Target not found: {target}")
    output = target if recategorize else args.output_path
    if args.mode == "learn":
        return target, target
    if output is None:
        default_out = (target.parent if target.is_dir() else target.parent.parent) / f"{target.stem}_sorted"
        answer = input(f"Output folder [{default_out}]: ").strip().strip("'\"")
        output = Path(answer) if answer else default_out
    if args.chunk_size is None:
        answer = input(f"Images per batch cycle [{CHUNK_SIZE}]: ").strip()
        args.chunk_size = int(answer) if answer else CHUNK_SIZE
    if args.nsfw_threshold is None:
        answer = input(f"R18 threshold 0-1, higher = stricter, -1 = off [{NSFW_THRESHOLD}]: ").strip()
        args.nsfw_threshold = float(answer) if answer else NSFW_THRESHOLD
    if args.fallback_count is None:
        answer = input(f"Fallback character pool size, 0 = off [{FALLBACK_COUNT}]: ").strip()
        args.fallback_count = int(answer) if answer else FALLBACK_COUNT
    if args.fallback_count > 0 and args.fallback_threshold is None:
        answer = input(f"Fallback match threshold 0-1, higher = stricter [{FALLBACK_THRESHOLD}]: ").strip()
        args.fallback_threshold = float(answer) if answer else FALLBACK_THRESHOLD
    if args.fallback_count > 0 and args.expand_threshold is None:
        answer = input(f"Series-guided search threshold 0-1, higher = stricter [{EXPAND_THRESHOLD}]: ").strip()
        args.expand_threshold = float(answer) if answer else EXPAND_THRESHOLD
    if args.gallery_threshold is None:
        answer = input(f"Reference picture match threshold 0-1, higher = stricter, -1 = off [{GALLERY_THRESHOLD}]: ").strip()
        args.gallery_threshold = float(answer) if answer else GALLERY_THRESHOLD
    if args.trained_threshold is None:
        args.trained_threshold = -1
        if (TRAINED_DIR / "head.safetensors").exists():
            answer = input(f"Trained model match threshold 0-1, higher = stricter, -1 = off [{TRAINED_THRESHOLD}]: ").strip()
            args.trained_threshold = float(answer) if answer else TRAINED_THRESHOLD
    return target, output.expanduser().resolve()


def collect_images(target, output):
    if target.is_file():
        return [target]
    return sorted(
        p for p in target.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTENSIONS and (target == output or output not in p.parents)
    )


def original_stem(path):
    prefix = f"{legalize(path.parent.name).lower()}_"
    return path.stem[len(prefix):] if path.stem.startswith(prefix) else path.stem


def remove_empty_dirs(root):
    for d in sorted((d for d in root.rglob("*") if d.is_dir()), key=lambda d: len(d.parts), reverse=True):
        if not any(d.iterdir()):
            d.rmdir()


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def features(out):
    return F.normalize(out if torch.is_tensor(out) else out.pooler_output, dim=-1)


class Sorter:
    def __init__(self, model_id, device):
        self.device = device
        self.model = AutoModel.from_pretrained(model_id).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.scale = self.model.logit_scale.exp().item()

    @torch.no_grad()
    def embed_texts(self, texts):
        inputs = self.processor(text=texts, padding="max_length", max_length=64, return_tensors="pt").to(self.device)
        return features(self.model.get_text_features(**inputs))

    def embed_groups(self, groups):
        return torch.stack([F.normalize(self.embed_texts(g).mean(0), dim=0) for g in groups])

    @torch.no_grad()
    def embed_pil(self, images):
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        return features(self.model.get_image_features(**inputs)).cpu()

    def embed_images(self, paths, batch_size):
        kept, embs = [], []
        for i in range(0, len(paths), batch_size):
            batch, images = [], []
            for p in paths[i:i + batch_size]:
                try:
                    images.append(Image.open(p).convert("RGB"))
                    batch.append(p)
                except Exception as e:
                    log(f"Skip {p.name}: {e}")
            if not images:
                continue
            embs.append(self.embed_pil(images))
            kept.extend(batch)
            progress("Analyzing images", min(i + batch_size, len(paths)), len(paths))
        return kept, torch.cat(embs) if embs else torch.empty(0)

    def probs(self, image_embs, text_embs):
        return (self.scale * image_embs @ text_embs.cpu().T).softmax(dim=-1)


def classify_categories(sorter, embs, class_embs, threshold):
    names = list(CATEGORY_PROMPTS)
    probs = sorter.probs(embs, class_embs)
    conf, idx = probs.max(dim=-1)
    categories = [names[i] if c >= threshold else UNCATEGORIZABLE for c, i in zip(conf.tolist(), idx.tolist())]
    return categories, conf.tolist(), probs


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
        qualifiers = QUALIFIER.findall(char)
        series = self.series(char) or (qualifiers[-1] if qualifiers else None)
        return legalize(title_name(series)) if series else UNKNOWN_TITLE, legalize(base_name(char))

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
    def tag(self, paths, batch_size, threshold, r18_threshold):
        labels, r18, appearance, multi = [], [], [], []
        for i in range(0, len(paths), batch_size):
            batch = torch.stack([self.transform(self.pad_square(Image.open(p).convert("RGB"))) for p in paths[i:i + batch_size]])
            probs = self.model(batch[:, [2, 1, 0]].to(self.device)).sigmoid().cpu()
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

    def match(self, embs, threshold):
        if self.embs is None:
            return [None] * len(embs)
        sims, idx = (embs @ self.embs.T).max(dim=-1)
        return [(self.labels[j], s) if s >= threshold else None for s, j in zip(sims.tolist(), idx.tolist())]

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


def load_sus():
    return json.loads(SUS_FILE.read_text(encoding="utf-8")) if SUS_FILE.exists() else []


def save_sus(entries):
    SUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SUS_FILE.write_text(json.dumps(entries, indent=1, ensure_ascii=False), encoding="utf-8")


def conflicts(trained, other):
    titles = {t.lower() for t in (trained[0], other[0]) if t and t.lower() != UNKNOWN_TITLE.lower()}
    return bool(other[1] and other[1] != trained[1]) or len(titles) > 1


def labeled_images(root):
    items = []
    for p in collect_images(root, root):
        parts = p.relative_to(root).parts
        if ANIME in parts:
            k = parts.index(ANIME)
            if len(parts) == k + 4:
                items.append((p, (legalize(parts[k + 1]).lower(), legalize(parts[k + 2]).lower())))
    if not items:
        raise SystemExit(f"No images in {ANIME}/<Title>/<Character>/ folders under {root}")
    return items


def learn_from_folder(root, sorter, gallery, batch_size):
    label_of = dict(labeled_images(root))
    paths, labels = list(label_of), list(label_of.values())
    kept, embs = sorter.embed_images(paths, batch_size)
    before = len(gallery.labels)
    gallery.add(embs, [label_of[p] for p in kept])
    gallery.save()
    log(f"Learned {len(gallery.labels) - before} new pictures across {len(set(labels))} characters (gallery: {len(gallery.labels)})")


class TrainedModel:
    def __init__(self, path, device):
        self.device = device
        self.model = AutoModel.from_pretrained(path).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(path)
        head = load_file(path / "head.safetensors")
        self.weight, self.bias = head["weight"].to(device), head["bias"].to(device)
        self.labels = [tuple(l) for l in json.loads((path / "labels.json").read_text(encoding="utf-8"))]

    @torch.no_grad()
    def predict(self, paths, batch_size):
        results = []
        for i in range(0, len(paths), batch_size):
            images = [Image.open(p).convert("RGB") for p in paths[i:i + batch_size]]
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            logits = features(self.model.get_image_features(**inputs)) @ self.weight.T + self.bias
            conf, idx = logits.softmax(dim=-1).max(dim=-1)
            results += [(self.labels[j], c) for c, j in zip(conf.tolist(), idx.tolist())]
            progress("Trained model check", min(i + batch_size, len(paths)), len(paths))
        return results


def detect_nsfw(clf, paths, batch_size, threshold):
    flags = []
    for n, res in enumerate(clf((Image.open(p).convert("RGB") for p in paths), batch_size=batch_size), 1):
        flags.append(next(r["score"] for r in res if r["label"] == "nsfw") >= threshold)
        progress("Checking R18", n, len(paths))
    return flags


def transfer_unique(src, dest_dir, mode, stem):
    dest = dest_dir / f"{stem}{src.suffix}"
    n = 1
    while dest.exists() and dest != src:
        dest = dest_dir / f"{stem}_{n}{src.suffix}"
        n += 1
    if dest == src:
        return src
    dest_dir.mkdir(parents=True, exist_ok=True)
    (shutil.copy2 if mode == "copy" else shutil.move)(src, dest)
    return dest


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
        found = expander.resolve(embs[unknown], [appearance[i] for i in unknown], args.fallback_threshold, args.expand_threshold, tagger)
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


def main():
    args = parse_args()
    ensure_hf_login()
    target, output = resolve_paths(args)
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

    output.mkdir(parents=True, exist_ok=True)
    totals, r18_total = Counter(), 0
    chunk_count = -(-len(paths) // args.chunk_size)
    for start in range(0, len(paths), args.chunk_size):
        chunk = paths[start:start + args.chunk_size]
        label = f"Chunk {start // args.chunk_size + 1}/{chunk_count}"
        log(f"{label}: images {start + 1}-{start + len(chunk)} of {len(paths)}")
        try:
            categories, r18 = process_chunk(chunk, models, args, output, label, None if recategorize else done_file)
        except Exception as e:
            log(f"{label} failed, skipping: {e}")
            continue
        tagger.save_cache()
        if expander:
            expander.save()
        if gallery:
            gallery.save()
        totals.update(categories)
        r18_total += sum(r18)

    if recategorize:
        remove_empty_dirs(output)
    log(f"Done. Summary: {dict(totals)}, {R18}: {r18_total}")


if __name__ == "__main__":
    main()
