"""User settings file (clip_config.ini): folder names, sorter and trainer options.

Created with defaults on first run. Command-line flags override it for one run; an empty value
means "ask when the script starts". Imported by config.py, so it must not import config itself.
"""
import configparser
import os
import re
import sys
from pathlib import Path

from .setup import PROJECT_DIR


def _config_path():
    for i, arg in enumerate(sys.argv):
        if arg == "--config" and i + 1 < len(sys.argv):
            return Path(sys.argv[i + 1]).expanduser().resolve()
        if arg.startswith("--config="):
            return Path(arg.split("=", 1)[1]).expanduser().resolve()
    return Path(os.environ.get("CLIP_CONFIG") or PROJECT_DIR / "clip_config.ini")


CONFIG_FILE = _config_path()

DEFAULT_FOLDERS = {
    "anime": "Anime_Handdrawn",
    "real": "Real_Photograph",
    "r18": "R18",
    "non_human": "Non_Human",
    "uncategorizable": "Uncategorizable",
}

TEMPLATE = """\
# CLIP Image Sorter settings
# Edit and save, then just run: python clip_sorter.py / python clip_sorter_train.py
# Command-line flags override these for one run (python clip_sorter.py --help).
# Leave a value empty to be asked when the script starts.
# Write paths as they are, no quotes needed: D:\\Pictures\\Images
# Delete this file to get the defaults back.

[folders]
# Top-level folder names in the sorted output. After changing them, the next sorter or
# trainer run offers to rename the existing folders (and their records) to match.
anime = Anime_Handdrawn
real = Real_Photograph
# parent folder for adult images (holds its own anime/real folders), see [sorter] r18
r18 = R18
non_human = Non_Human
uncategorizable = Uncategorizable

[files]
# rename sorted files to <character>_<first 10 letters of the original name> (sorter and trainer): on / off
# off keeps the file's current name (a _1, _2 suffix is added only when the name is taken)
rename = on

[sorter]
# folder (or single image) to sort
target_path = D:\\Pictures\\Images\\
# where sorted images go (also the trainer's folder unless [trainer] target_path is set)
output_path = D:\\Pictures\\auto_sorted
# copy, move, recategorize (re-sort the output folder in place) or learn (add a hand-fixed folder to the gallery)
mode = move
# split adult images into the [folders] r18 folder: on / off
r18 = off
# R18 cutoff 0-1, higher = flags fewer images
nsfw_threshold = 0.85
# images fully sorted and copied per cycle (progress is saved after each)
chunk_size = 6000
batch_size = 8
# minimum confidence for anime / real / non-human, below it goes to uncategorizable
category_threshold = 0.45
# WD Tagger character cutoff 0-1
character_threshold = 0.95
# top Danbooru characters used for name matching (0 = off); about 68000 is all of them
fallback_count = 10000000
fallback_threshold = 0.95
expand_threshold = 0.95
# reference picture match cutoff, -1 = off
gallery_threshold = 0.95
# your trained model's cutoff, -1 = off (skipped anyway until you train one)
trained_threshold = 0.95
# move "anime" images the tagger rates realistic to the real folder, -1 = off
realistic_threshold = 0.5
# duplicates: off, exact (same file content) or similar (same picture resized / re-encoded)
# compared with everything already in the output folder and earlier in the run; the first copy is kept
dedupe = off
# where duplicates go (copied in copy mode, moved otherwise), empty = <output_path>/duplicates
duplicates_path =
# similar only: how different two pictures may be, 0-64 bits (0 = identical, 4 = safe, 10 = loose)
similar_distance = 4
model = google/siglip2-so400m-patch14-384
tagger = SmilingWolf/wd-eva02-large-tagger-v3
nsfw_model = Falconsai/nsfw_image_detection

[trainer]
# sorted folder to train on, empty = [sorter] output_path
target_path =
# base model when no trained copy exists yet, empty = [sorter] model
model =
epochs = 5
batch_size = 8
# last vision layers to fine-tune, 0 = classifier only (fast)
unfreeze = 2
lr = 1e-5
# 1 sort unclassed, 2 fix wrong category, 3 resolve suspicious, 4 train only, 5 roll back; empty = show the menu
task =
# unattended: record unclear Danbooru matches as pending, auto-recover and auto-train (on / off)
no_ask = off
# do not look up new characters on Danbooru (on / off)
no_link = off
# do not load models for label suggestions (on / off)
no_suggest = off
"""

TRUE, FALSE = {"on", "yes", "true", "1"}, {"off", "no", "false", "0"}
BAD_NAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def _load():
    if not CONFIG_FILE.exists():
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_FILE.write_text(TEMPLATE, encoding="utf-8")
        print(f"Created default settings: {CONFIG_FILE}")
    parser = configparser.ConfigParser(interpolation=None, comment_prefixes=("#", ";"), inline_comment_prefixes=None)
    try:
        parser.read(CONFIG_FILE, encoding="utf-8")
    except configparser.Error as e:
        raise SystemExit(f"Cannot read {CONFIG_FILE}: {e}")
    return parser


PARSER = _load()


def raw(section, key):
    """Value as written, '' when missing or empty. Surrounding quotes are dropped."""
    value = PARSER.get(section, key, fallback="").strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1]
    return value


def convert(section, key, kind, text):
    try:
        if kind is bool:
            if text.lower() in TRUE:
                return True
            if text.lower() in FALSE:
                return False
            raise ValueError("use on or off")
        if kind is Path:
            return Path(text).expanduser()
        return kind(text)
    except ValueError as e:
        raise SystemExit(f"{CONFIG_FILE}: [{section}] {key} = {text!r} is not valid ({e})")


def apply(args, section, spec):
    """Fill every option the command line left as None: from the settings file, else the built-in default.
    spec: {key: (type, default)}; default None leaves it None so the script asks."""
    for key, (kind, default) in spec.items():
        if getattr(args, key, None) is not None:
            continue
        text = raw(section, key)
        setattr(args, key, convert(section, key, kind, text) if text else default)


def save(section, values):
    """Write key = value lines into the settings file, keeping comments and layout."""
    lines = CONFIG_FILE.read_text(encoding="utf-8").splitlines()
    start = next((i for i, l in enumerate(lines) if l.strip().lower() == f"[{section}]"), None)
    if start is None:
        lines += ["", f"[{section}]"]
        start = len(lines) - 1
    end = next((i for i in range(start + 1, len(lines)) if lines[i].lstrip().startswith("[")), len(lines))
    for key, value in values.items():
        text = "on" if value is True else "off" if value is False else str(value)
        line = f"{key} = {text}"
        hit = next((i for i in range(start + 1, end) if re.match(rf"\s*{re.escape(key)}\s*=", lines[i])), None)
        if hit is None:
            at = end
            while at > start + 1 and not lines[at - 1].strip():
                at -= 1
            lines.insert(at, line)
            end += 1
        else:
            lines[hit] = line
        if not PARSER.has_section(section):
            PARSER.add_section(section)
        PARSER.set(section, key, text)
    CONFIG_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def offer_save(section, values):
    values = {k: v for k, v in values.items() if v is not None}
    if not values:
        return
    shown = ", ".join(f"{k} = {v}" for k, v in values.items())
    if input(f"Save to {CONFIG_FILE.name} [{section}] so you are not asked next time ({shown})? [Y/n]: ").strip().lower() in ("", "y", "yes"):
        save(section, values)
        print(f"Saved to {CONFIG_FILE}")


def _folders():
    names = {}
    for key, default in DEFAULT_FOLDERS.items():
        name = raw("folders", key) or default
        if BAD_NAME.search(name) or name in (".", "..") or name != name.rstrip(" .") or name != name.strip():
            raise SystemExit(f'{CONFIG_FILE}: [folders] {key} = "{name}" is not a usable folder name '
                             '(no < > : " / \\ | ? *, no leading/trailing spaces, no trailing dot)')
        names[key] = name
    lowered = [n.lower() for n in names.values()]
    if len(set(lowered)) != len(lowered):
        raise SystemExit(f"{CONFIG_FILE}: [folders] names must all be different: {names}")
    return names


FOLDERS = _folders()
RENAME_FILES = convert("files", "rename", bool, raw("files", "rename")) if raw("files", "rename") else True
