"""Fine-tuned character model."""
import json
import torch
from PIL import Image
from safetensors.torch import load_file
from transformers import AutoModel
from transformers import AutoProcessor

from .logs import progress
from .siglip import features


class TrainedModel:
    def __init__(self, path, device):
        self.device = device
        self.model = AutoModel.from_pretrained(path).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(path)
        head = load_file(path / "head.safetensors")
        self.weight, self.bias = head["weight"].to(device), head["bias"].to(device)
        self.labels = [tuple(l) for l in json.loads((path / "labels.json").read_text(encoding="utf-8"))]

    @torch.no_grad()
    def top(self, path, k):
        inputs = self.processor(images=[Image.open(path).convert("RGB")], return_tensors="pt").to(self.device)
        probs = (features(self.model.get_image_features(**inputs)) @ self.weight.T + self.bias).softmax(dim=-1)[0]
        conf, idx = probs.topk(min(k, len(self.labels)))
        return [(self.labels[j], c) for c, j in zip(conf.tolist(), idx.tolist())]

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
