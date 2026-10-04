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

## How anime characters are found

Tried in order until one is confident:

1. WD Tagger (SmilingWolf/wd-eva02-large-tagger-v3)
2. Reference gallery of already-known pictures
3. SigLIP2 name match against top Danbooru characters
4. Series-guided Danbooru search by hair/eye color
5. Danbooru reference pictures for the top candidates
6. Your trained model (see below)
7. Series folder only, or `Anime_Handdrawn` root

Series names come from Danbooru and are cached in `./models`.

## Train

```bash
python clip_sorter_train.py --target-path ./pics_sorted
```

Shows each anime image without a character, asks for character and series (with typo suggestions), moves it, then fine-tunes a SigLIP2 copy into `./models/finetuned`. `clip_sorter.py` uses it as the last fallback.

Useful flags: `--skip-labeling`, `--epochs`, `--unfreeze 0` (classifier only, fast).

## Models

- [google/siglip2-so400m-patch14-384](https://huggingface.co/google/siglip2-so400m-patch14-384)
- [SmilingWolf/wd-eva02-large-tagger-v3](https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3)
- [Falconsai/nsfw_image_detection](https://huggingface.co/Falconsai/nsfw_image_detection)

R18 detection is a first pass, not moderation.

## License

[Apache-2.0](LICENSE). Third-party packages and models keep their own licenses; they are downloaded at runtime, not bundled. Danbooru data is fetched live and subject to Danbooru's terms.
