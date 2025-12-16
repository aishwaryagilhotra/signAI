import os, copy, json
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from sklearn.model_selection import train_test_split

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

DATA_DIR = "asl-alphabet/asl_alphabet_train/asl_alphabet_train"

MODEL_OUT = "asl_model.pth"

IMG_SIZE = 128          # smaller = faster locally
BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-3

# transforms
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# dataset
dataset = datasets.ImageFolder(DATA_DIR)
classes = dataset.classes
print("Classes:", classes)

with open("class_names.json", "w") as f:
    json.dump(classes, f)

indices = list(range(len(dataset)))
train_idx, val_idx = train_test_split(
    indices, test_size=0.15, stratify=dataset.targets
)

train_ds = torch.utils.data.Subset(dataset, train_idx)
val_ds = torch.utils.data.Subset(dataset, val_idx)

train_ds.dataset.transform = train_tf
val_ds.dataset.transform = val_tf

train_loader = torch.utils.data.DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True
)
val_loader = torch.utils.data.DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False
)

# model (FAST)
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
for p in model.parameters():
    p.requires_grad = False

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features, len(classes)
)

model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=LR)

best_acc = 0
best_weights = copy.deepcopy(model.state_dict())

for epoch in range(1, EPOCHS+1):
    print(f"\nEpoch {epoch}/{EPOCHS}")
    for phase in ["train", "val"]:
        model.train() if phase=="train" else model.eval()
        loader = train_loader if phase=="train" else val_loader

        correct, total, loss_sum = 0, 0, 0
        for imgs, labels in tqdm(loader):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            with torch.set_grad_enabled(phase=="train"):
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                preds = outputs.argmax(1)

                if phase=="train":
                    loss.backward()
                    optimizer.step()

            loss_sum += loss.item()*imgs.size(0)
            correct += (preds==labels).sum().item()
            total += imgs.size(0)

        acc = correct/total
        print(f"{phase.upper()} Acc: {acc:.4f}")

        if phase=="val" and acc>best_acc:
            best_acc = acc
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, MODEL_OUT)
            print("Saved best model")

print("Training done. Best acc:", best_acc)
