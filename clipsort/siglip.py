"""SigLIP2 model and main category."""
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel
from transformers import AutoProcessor

from .config import CATEGORY_PROMPTS, UNCATEGORIZABLE
from .logs import log, progress


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
