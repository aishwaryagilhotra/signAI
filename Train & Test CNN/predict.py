import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import json
import sys
import os

MODEL_PATH = "asl_model.pth"          # your saved model
CLASS_NAMES_PATH = "class_names.json" # saved during training
IMG_SIZE = 128                        # same as training

# ===== LOAD CLASS NAMES =====
with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

# ===== IMAGE TRANSFORMS =====
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# ===== LOAD MODEL =====
def load_model():
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    
    # replace classifier (must match training!)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))

    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

# ===== PREDICT FUNCTION =====
def predict(image_path):
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    img = Image.open(image_path).convert("RGB")
    img_t = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_t)
        _, predicted = torch.max(outputs, 1)
        label = class_names[predicted.item()]

    print(f"Predicted ASL Letter: {label}")

# ===== CLI USAGE =====
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: py predict.py <image_path>")
        sys.exit(1)

    img_path = sys.argv[1]
    predict(img_path)
