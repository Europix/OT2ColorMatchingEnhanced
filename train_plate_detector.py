import os
import random

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.model_selection import train_test_split

DATA_ROOT = "dataset/train"

classes = ["nothing", "1", "6", "12", "24", "48", "96", "384"]
class_to_idx = {c: i for i, c in enumerate(classes)}

IMG_SIZE = 400
BATCH_SIZE = 16
LR = 1e-4
EPOCHS = 40

# 固定随机种子，训练更稳定
torch.manual_seed(42)
random.seed(42)

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]


class PlateDataset(Dataset):
    """Simple image classification dataset."""

    def __init__(self, files, labels, transform):
        self.files = list(files)
        self.labels = list(labels)
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        img = self.transform(img)
        label = self.labels[idx]
        return img, label


# ------------------------- load all files -------------------------
all_files = []
all_labels = []

for cls in classes:
    folder = os.path.join(DATA_ROOT, cls)
    if not os.path.isdir(folder):
        continue
    for fname in os.listdir(folder):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            all_files.append(os.path.join(folder, fname))
            all_labels.append(class_to_idx[cls])

# 洗牌
combined = list(zip(all_files, all_labels))
random.shuffle(combined)
all_files, all_labels = zip(*combined)

# train / val
train_files, val_files, train_labels, val_labels = train_test_split(
    all_files,
    all_labels,
    test_size=0.15,
    stratify=all_labels,
    random_state=42,
)

# ------------------------- transforms -------------------------
train_tf = transforms.Compose([
    # 随机裁到中间区域，尽量让网络关注 plate 而不是左右金属背景
    transforms.RandomResizedCrop(
        IMG_SIZE,
        scale=(0.75, 1.0),
        ratio=(0.9, 1.1),
    ),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
    ),
    transforms.RandomRotation(5),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
])

val_tf = transforms.Compose([
    # 先放大一点再中心裁剪，和训练时的 “关注中间区域” 保持一致
    transforms.Resize(int(IMG_SIZE * 1.2)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
])

train_ds = PlateDataset(train_files, train_labels, train_tf)
val_ds   = PlateDataset(val_files, val_labels, val_tf)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# ------------------------- model -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

in_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(in_features, len(classes))
)

model = model.to(device)

# label_smoothing 可以稍微缓解过拟合
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
optimizer = optim.Adam(model.parameters(), lr=LR)

# ------------------------- train loop -------------------------
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for imgs, labels in train_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # ---- validation ----
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    print(
        f"Epoch {epoch+1}/{EPOCHS} "
        f"| Train Loss {running_loss/len(train_loader):.4f} "
        f"| Val Loss {val_loss/len(val_loader):.4f} "
        f"| Val Acc {correct/total:.3f}"
    )

torch.save(model.state_dict(), "plate_model_resnet18_colorRing.pth")
print("Model saved.")
