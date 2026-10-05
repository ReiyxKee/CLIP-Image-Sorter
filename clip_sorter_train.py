import clipsort
import argparse
import json
from pathlib import Path

from clipsort.config import MODEL_ID, SESSION_FILE
from clipsort.corrections import apply_corrections, parse_corrections, run_corrections
from clipsort.finetune import rollback, train
from clipsort.gallery import Gallery
from clipsort.labeling import label_images, load_multi, suspicious_images, unlabeled_images
from clipsort.linking import link_new_characters
from clipsort.logs import log
from clipsort.records import labeled_images
from clipsort.siglip import pick_device


def parse_args():
    p = argparse.ArgumentParser(description="Label unknown anime images and fine-tune a SigLIP2 copy as the last fallback.")
    p.add_argument("--target-path", type=Path, help="Sorted output folder")
    p.add_argument("--model", default=MODEL_ID, help="Base model when no trained copy exists yet")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--unfreeze", type=int, default=2, help="Last vision layers to fine-tune, 0 = classifier only")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--task", choices=("1", "2", "3", "4", "5"), help="1 sort unclassed, 2 fix wrong category, 3 resolve suspicious, 4 train only, 5 roll back model")
    p.add_argument("--skip-labeling", action="store_true", help="Same as --task 4")
    p.add_argument("--fresh", action="store_true", help="Retrain from the original model instead of the last trained copy")
    p.add_argument("--no-ask", action="store_true", help="Skip unclear Danbooru matches instead of asking; they are asked next time")
    p.add_argument("--relink", action="store_true", help="Ask again for characters and series earlier marked as not on Danbooru")
    p.add_argument("--no-link", action="store_true", help="Do not look up new characters on Danbooru")
    p.add_argument("--no-suggest", action="store_true", help="Do not load models for label suggestions")
    p.add_argument("--corrections", type=Path, help="Text file of wrongly sorted characters, see README")
    return p.parse_args()


def main():
    args = parse_args()
    root = (args.target_path or Path(input("Sorted output folder: ").strip().strip("'\""))).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Folder not found: {root}")
    task = "4" if args.skip_labeling else "2" if args.corrections else args.task
    resume = False
    if SESSION_FILE.exists():
        pending = json.loads(SESSION_FILE.read_text(encoding="utf-8"))
        log(f"Warning: unfinished correction session found for {pending.get('root')} (stopped at stage '{pending.get('stage')}')")
        if pending.get("root") != str(root):
            log(f"It belongs to another folder. Run with --target-path \"{pending.get('root')}\" to recover it.")
        elif input("Recover it now? [Y/n]: ").strip().lower() in ("", "y", "yes"):
            task, resume = "2", True
        elif input("Discard it? [y/N]: ").strip().lower() in ("y", "yes"):
            SESSION_FILE.unlink()
    if task is None:
        print(f"1) Sort unclassed images ({len(unlabeled_images(root))})")
        print("2) Fix wrong category (detect images you moved by hand)")
        print(f"3) Resolve suspicious ({len(suspicious_images(root))})")
        print("4) Train only")
        print("5) Roll back trained model")
        task = input("Choose [1-5]: ").strip()
    if task == "1":
        label_images(root, unlabeled_images(root), args)
    elif task == "2" and args.corrections:
        label_images(root, apply_corrections(root, parse_corrections(args.corrections.expanduser()), Gallery(None, args.model)), args)
    elif task == "2":
        run_corrections(root, args, resume)
    elif task == "3":
        label_images(root, suspicious_images(root), args)
    elif task == "5":
        return rollback()
    elif task != "4":
        raise SystemExit("Choose 1-5")
    if not args.no_link and task != "2":
        link_new_characters(root, args.model, args.relink, not args.no_ask)
    if task != "4" and input("Train now? [Y/n]: ").strip().lower() not in ("", "y", "yes"):
        return
    items = [(p, [label]) for p, label in labeled_images(root)]
    items += [(root / rel, [tuple(l) for l in group]) for rel, group in load_multi(root).items() if (root / rel).exists()]
    if len({label for _, group in items for label in group}) < 2:
        raise SystemExit("Need at least 2 characters in <Series>/<Character>/ folders to train")
    train(items, args, pick_device())
    if task == "2":
        SESSION_FILE.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
