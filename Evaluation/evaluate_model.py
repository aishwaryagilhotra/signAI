import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import json
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# -------------------------
# CONFIG
# -------------------------
MODEL_PATH = "asl_model.pth"
CLASS_NAMES_PATH = "class_names.json"
TEST_DIR = "asl-alphabet/asl_alphabet_test/asl_alphabet_test"
IMG_SIZE = 128

# -------------------------
# LOAD CLASS NAMES
# -------------------------
with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

# Some images (like 'del') missing in test set → ensure mapping still valid
label_to_index = {name: idx for idx, name in enumerate(class_names)}

# -------------------------
# MODEL TRANSFORMS
# -------------------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# -------------------------
# LOAD MODEL
# -------------------------
def load_model():
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

model = load_model()

# -------------------------
# RUN EVALUATION
# -------------------------
true_labels = []
pred_labels = []

for file in os.listdir(TEST_DIR):
    if not file.endswith(".jpg"):
        continue

    # extract label from filename → e.g., A_test.jpg → A
    raw_label = file.replace("_test.jpg", "")
    true_label = raw_label

    if true_label not in class_names:
        print(f"⚠ Skipping unknown label file: {file}")
        continue

    img_path = os.path.join(TEST_DIR, file)
    img = Image.open(img_path).convert("RGB")
    img_t = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_t)
        _, pred = torch.max(outputs, 1)
        pred_label = class_names[pred.item()]

    true_labels.append(true_label)
    pred_labels.append(pred_label)

# -------------------------
# PRINT CLASSIFICATION REPORT
# -------------------------
print("\nClassification Report:\n")
print(classification_report(true_labels, pred_labels))

# -------------------------
# CONFUSION MATRIX
# -------------------------
unique_labels = sorted(set(true_labels))

cm = confusion_matrix(true_labels, pred_labels, labels=unique_labels)

plt.figure(figsize=(15, 15))
sns.heatmap(cm, annot=True, cmap="Blues", xticklabels=unique_labels, yticklabels=unique_labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix — Custom ASL Evaluation")
plt.tight_layout()
plt.savefig("confusion_matrix.png")

print("\nSaved confusion_matrix.png")

# -------------------------
# PER-CLASS ACCURACY
# -------------------------
accuracies = {}
for i, label in enumerate(unique_labels):
    correct = cm[i][i]
    total = cm[i].sum()
    accuracies[label] = float(correct / total)

with open("per_class_accuracy.json", "w") as f:
    json.dump(accuracies, f, indent=4)

print("\nSaved per_class_accuracy.json")
print("\nEvaluation complete!")
