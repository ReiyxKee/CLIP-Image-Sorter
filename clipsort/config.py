"""Constants and file paths."""
import os
import re
from pathlib import Path


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
EXPAND_THRESHOLD = 0.9
SERIES_THRESHOLD = 0.5
APPEARANCE_THRESHOLD = 0.35
GALLERY_THRESHOLD = 0.9
GALLERY_MARGIN = 0.03
TRAINED_THRESHOLD = 0.9
KEEP_VERSIONS = 3
TRAINED_DIR = Path(os.environ["HF_HOME"]) / "finetuned"
SUS_FILE = Path(os.environ["HF_HOME"]) / "suspicious.json"
OVERRIDES_FILE = Path(os.environ["HF_HOME"]) / "series_overrides.json"
ALIASES_FILE = Path(os.environ["HF_HOME"]) / "character_aliases.json"
INDEX_FILE = ".sorted_index.jsonl"
SESSION_FILE = Path(os.environ["HF_HOME"]) / "correction_session.json"
REF_SHORTLIST = 5
REF_IMAGES = 5
COLORS = ("aqua", "black", "blonde", "blue", "brown", "green", "grey", "orange", "pink", "purple", "red", "silver", "white", "yellow")
FALLBACK_THRESHOLD = 0.85
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
MULTI_FILE = ".multi_labels.json"
RETRAIN_FILE = TRAINED_DIR.parent / "retrain_labels.json"
LINKED_FILE = TRAINED_DIR.parent / "linked_characters.json"
LINK_IMAGES = 10
DISCOVER_TAGGER = 0.5
DISCOVER_GALLERY = 0.85
CAST_LIMIT = 100
