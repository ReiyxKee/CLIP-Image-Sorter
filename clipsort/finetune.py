"""Fine-tuning."""
import json
import random
import shutil
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file
from safetensors.torch import save_file
from torchvision import transforms
from transformers import AutoModel
from transformers import AutoProcessor

from .config import RETRAIN_FILE, TRAINED_DIR
from .logs import log, progress
from .siglip import features


def load_image(path, augment):
    try:
        return augment(Image.open(path).convert("RGB"))
    except Exception as e:
        log(f"Skip {path.name}: {e}")
        return None


def train(items, args, device):
    resume = (TRAINED_DIR / "head.safetensors").exists() and not args.fresh
    source = TRAINED_DIR if resume else args.model
    log(f"Loading {source} on {device}...")
    model = AutoModel.from_pretrained(source).to(device)
    processor = AutoProcessor.from_pretrained(source)

    labels = sorted({label for _, group in items for label in group})
    index = {label: i for i, label in enumerate(labels)}
    head = torch.nn.Linear(model.config.vision_config.hidden_size, len(labels)).to(device)
    reset = {tuple(l) for l in json.loads(RETRAIN_FILE.read_text(encoding="utf-8"))} if RETRAIN_FILE.exists() else set()
    if resume:
        old = load_file(TRAINED_DIR / "head.safetensors")
        old_labels = [tuple(l) for l in json.loads((TRAINED_DIR / "labels.json").read_text(encoding="utf-8"))]
        dropped = len(set(old_labels) - set(index))
        log(f"Continuing from last trained model: {dropped} characters dropped, {len(reset & set(index))} corrected characters relearned from scratch")
        with torch.no_grad():
            for j, label in enumerate(old_labels):
                if label in index and label not in reset:
                    head.weight[index[label]] = old["weight"][j].to(device)
                    head.bias[index[label]] = old["bias"][j].to(device)

    for param in model.parameters():
        param.requires_grad = False
    vision = model.vision_model
    tuned = [*vision.encoder.layers[-args.unfreeze:], vision.post_layernorm, getattr(vision, "head", None)] if args.unfreeze > 0 else []
    backbone = [param for module in tuned if module is not None for param in module.parameters()]
    for param in backbone:
        param.requires_grad = True
    optimizer = torch.optim.AdamW([{"params": backbone, "lr": args.lr}, {"params": head.parameters(), "lr": 1e-3}])
    augment = transforms.Compose([transforms.RandomResizedCrop(384, scale=(0.6, 1.0)), transforms.RandomHorizontalFlip()])
    log(f"Training on {len(items)} images, {len(labels)} characters, last {args.unfreeze} layers unfrozen. Ctrl+C stops and saves.")
    try:
        run_epochs(model, processor, head, optimizer, augment, items, labels, index, args, device)
    except KeyboardInterrupt:
        log("Stopped early, saving progress so far...")
    save_trained(model, processor, head, labels)


def run_epochs(model, processor, head, optimizer, augment, items, labels, index, args, device):
    for epoch in range(1, args.epochs + 1):
        random.shuffle(items)
        model.train()
        seen = correct = 0
        total_loss = 0.0
        for i in range(0, len(items), args.batch_size):
            batch = [(load_image(p, augment), group) for p, group in items[i:i + args.batch_size]]
            batch = [(img, group) for img, group in batch if img is not None]
            if not batch:
                continue
            try:
                inputs = processor(images=[img for img, _ in batch], return_tensors="pt").to(device)
                target = torch.zeros(len(batch), len(labels), device=device)
                for row, (_, group) in enumerate(batch):
                    target[row, [index[label] for label in group]] = 1 / len(group)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                    logits = head(features(model.get_image_features(**inputs)).float())
                loss = F.cross_entropy(logits.float(), target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            except Exception as e:
                log(f"Batch skipped: {e}")
                continue
            seen += len(batch)
            correct += (target.gather(1, logits.argmax(dim=-1, keepdim=True)) > 0).sum().item()
            total_loss += loss.item() * len(batch)
            progress(f"Epoch {epoch}/{args.epochs}", min(i + args.batch_size, len(items)), len(items))
        log(f"Epoch {epoch}: loss {total_loss / max(seen, 1):.3f}, train accuracy {correct * 100 // max(seen, 1)}%")


def save_trained(model, processor, head, labels):
    staging = TRAINED_DIR.with_name(TRAINED_DIR.name + "_new")
    shutil.rmtree(staging, ignore_errors=True)
    model.save_pretrained(staging)
    processor.save_pretrained(staging)
    save_file({"weight": head.weight.detach().cpu().contiguous(), "bias": head.bias.detach().cpu().contiguous()}, staging / "head.safetensors")
    (staging / "labels.json").write_text(json.dumps([list(l) for l in labels]), encoding="utf-8")
    del model
    shutil.rmtree(TRAINED_DIR, ignore_errors=True)
    staging.rename(TRAINED_DIR)
    RETRAIN_FILE.unlink(missing_ok=True)
    log(f"Saved trained model to {TRAINED_DIR}")
