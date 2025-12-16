import torch
import torch.nn as nn
from torchvision import models
import json

MODEL_PATH = "asl_model.pth"
CLASS_NAMES_PATH = "class_names.json"
ONNX_PATH = "asl_model.onnx"
IMG_SIZE = 128

# load class names
with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

# load model
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)

torch.onnx.export(
    model,
    dummy,
    ONNX_PATH,
    input_names=["input"],
    output_names=["output"],
    opset_version=12,
)

print(f"Exported ONNX model → {ONNX_PATH}")
