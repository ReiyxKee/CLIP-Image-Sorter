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

Menu: `1` sort unclassed images, `2` fix wrong category, `3` resolve suspicious, `4` train only, `5` roll back the trained model (or `--task 1-5`). Training follows.

The last 3 trained models are kept: `models/finetuned` (current), `finetuned.1` and `finetuned.2` (older). Each training shifts them back one; option `5` restores the previous one (~1.6 GB each).

Fix wrong category: move misplaced images (or whole character folders) to the right `<Series>/<Character>/` folder yourself, then run option `2`. You can also drop uncategorized images of one series straight into a `<Series>/` folder: option `2` finds the series and its cast on Danbooru and sorts them into character folders, leaving unsure ones for option `1`.

Option `2` runs in stages (moves, link, discover, finalize, train) and records progress in `models/correction_session.json`. If it stops midway, both scripts warn at start and the trainer offers to recover from the last stage. It compares the folders with `.sorted_index.jsonl` (written by the sorter), relabels the moved images in the gallery, renames them, and remembers series fixes (e.g. Kirito → Sword Art Online) in `models/series_overrides.json` for future sorts. The first run on an older sorted folder only records a baseline.

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

New character folders you create are linked to Danbooru before training: the matching tag is found (you pick when unclear), 10 reference pictures go into the gallery, the name joins the fallback pool, and future detections of that tag are filed under your folder names. When unsure it lists matches; pick a number, type the exact Danbooru tag yourself (checked before use), or press Enter for none. Unknown characters (e.g. OCs) are learned from your images only. `--relink` asks again for names earlier marked as not on Danbooru; `--no-link` skips linking.

Retraining continues from the last trained model, drops characters that no longer have images, and relearns any character touched by a correction from scratch. `--fresh` retrains everything from the original model.

Ctrl+C during training stops early and still saves what was learned.

Useful flags: `--skip-labeling`, `--epochs`, `--unfreeze 0` (classifier only, fast).

## Project layout

`clip_sorter.py` and `clip_sorter_train.py` are thin entry points; each feature lives in `clipsort/`:

| Module | Feature |
|---|---|
| `setup.py` | Dependency check, CUDA torch, Hugging Face paths |
| `hf_login.py` | Hugging Face CLI and login |
| `config.py` | Constants and file paths |
| `logs.py` | Console output and `Log/` files |
| `names.py` | Name cleanup and prompt text |
| `danbooru.py` | Danbooru API |
| `files.py` | Image listing, moving, cleanup |
| `records.py` | Index, suspicious list, aliases, series corrections |
| `siglip.py` | SigLIP2 and main category |
| `tagger.py` | WD Tagger |
| `nsfw.py` | R18 check |
| `pools.py` | Fallback pool and series-guided search |
| `gallery.py` | Reference gallery and learn mode |
| `trained.py` | Fine-tuned model |
| `sorting.py` | Per-chunk sorting flow |
| `labeling.py` | Labeling window and suggestions |
| `corrections.py` | Fix wrong category stages and recovery |
| `linking.py` | Danbooru linking for new characters |
| `finetune.py` | Training |

## Models

- [google/siglip2-so400m-patch14-384](https://huggingface.co/google/siglip2-so400m-patch14-384)
- [SmilingWolf/wd-eva02-large-tagger-v3](https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3)
- [Falconsai/nsfw_image_detection](https://huggingface.co/Falconsai/nsfw_image_detection)

R18 detection is a first pass, not moderation.

## License

[Apache-2.0](LICENSE). Third-party packages and models keep their own licenses; they are downloaded at runtime, not bundled. Danbooru data is fetched live and subject to Danbooru's terms.
