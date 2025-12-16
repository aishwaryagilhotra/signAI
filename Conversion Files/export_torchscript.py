import torch
import torch.nn as nn
from torchvision import models
import json

MODEL_PATH = "asl_model.pth"
CLASS_NAMES_PATH = "class_names.json"
OUTPUT_PATH = "asl_model.ptl"
IMG_SIZE = 128

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features, len(class_names)
)

model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

example = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)

scripted_model = torch.jit.trace(model, example)
scripted_model.save(OUTPUT_PATH)

print(f"Saved TorchScript model → {OUTPUT_PATH}")
