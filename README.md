# CLIP Image Sorter

Sort a folder of images into real photos, anime/hand-drawn art, non-human, and uncategorizable, then sort anime by series and character.

## Requirements

- Python 3.10–3.13
- NVIDIA GPU (CUDA) or Apple Silicon recommended; CPU works but is slow
- ~10 GB disk for models (stored in `./models`)

Missing packages are installed on first run after you confirm.

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux
```

## Sort

```bash
python clip_sorter.py
python clip_sorter.py --target-path ./pics --output-path ./pics_sorted
```

Anything not passed as a flag is asked at start. Modes: `copy` (default), `move`, `recategorize` (re-sort a sorted folder in place), `learn` (add a hand-fixed sorted folder to the reference gallery).

Output:

```
pics_sorted/
├── R18/{Real_Photograph, Anime_Handdrawn/...}
├── Real_Photograph/
├── Anime_Handdrawn/<Series>/<Character>/
├── Non_Human/
└── Uncategorizable/
```

Runs in chunks and resumes where it stopped (`.sorted_sources.txt` in the output folder).

Each run's log (command, every sorted file, errors with tracebacks) is saved to `./Log/<script>_<date_time>.log`.

## How anime characters are found

Tried in order until one is confident:

1. Your trained model (see below) runs alongside the steps below. When it is confident it wins; if the others disagree on character or series, the image is logged in `models/suspicious.json`
2. WD Tagger (SmilingWolf/wd-eva02-large-tagger-v3)
3. Reference gallery of already-known pictures
4. SigLIP2 name match against top Danbooru characters
5. Series-guided Danbooru search by hair/eye color
6. Danbooru reference pictures for the top candidates
7. Series folder only, or `Anime_Handdrawn` root

Series names come from Danbooru and are cached in `./models`.

## Train

```bash
python clip_sorter_train.py --target-path ./pics_sorted
```

Menu: `1` sort unclassed images, `2` fix wrong category, `3` resolve suspicious, `4` train only (or `--task 1-4`). Training follows.

Fix wrong category: move misplaced images (or whole character folders) to the right `<Series>/<Character>/` folder yourself, then run option `2`. It compares the folders with `.sorted_index.jsonl` (written by the sorter), relabels the moved images in the gallery, renames them, and remembers series fixes (e.g. Kirito → Sword Art Online) in `models/series_overrides.json` for future sorts. The first run on an older sorted folder only records a baseline.

Opens a labeling window for each anime image without a character: fill in character and series (click a suggestion to fix typos, `+ Character` for group images), Save (Enter), Skip (Esc) or Back (Ctrl+Z). It moves the image, then fine-tunes a SigLIP2 copy into `./models/finetuned`. `clip_sorter.py` checks it first.

Each image gets clickable suggestions from the tagger, your trained model, the gallery, the name pool, and Danbooru (series guess + reference pictures), loaded in the background. `--no-suggest` turns them off.

For images with several characters, enter them comma separated (`alsace, taihou`) and give each a series. They are filed like the sorter does (same series → series folder, mixed → `Anime_Handdrawn` root) and remembered in `.multi_labels.json` for training.

Wrongly sorted characters can be listed in a corrections file (`--corrections fixes.txt`):

```
# review every image in this folder
Azur Lane/Vanguard
# move the whole folder
Unknown Title/Kirito -> Sword Art Online/Kirito
```

Corrections also fix the reference gallery, so wrong pictures stop spreading.

Retraining continues from the last trained model, drops characters that no longer have images, and relearns any character touched by a correction from scratch. `--fresh` retrains everything from the original model.

Ctrl+C during training stops early and still saves what was learned.

Useful flags: `--skip-labeling`, `--epochs`, `--unfreeze 0` (classifier only, fast).

## Models

- [google/siglip2-so400m-patch14-384](https://huggingface.co/google/siglip2-so400m-patch14-384)
- [SmilingWolf/wd-eva02-large-tagger-v3](https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3)
- [Falconsai/nsfw_image_detection](https://huggingface.co/Falconsai/nsfw_image_detection)

R18 detection is a first pass, not moderation.

## License

[Apache-2.0](LICENSE). Third-party packages and models keep their own licenses; they are downloaded at runtime, not bundled. Danbooru data is fetched live and subject to Danbooru's terms.
