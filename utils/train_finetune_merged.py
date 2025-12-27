import os
import time
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torchvision import datasets, transforms, models


# -----------------------------
# 基本配置
# -----------------------------
OLD_DATA_ROOT = "dataset"      # 旧数据集，有 train/val
NEW_DATA_ROOT = "dataset2"     # 新数据集，只有一个层
OLD_CKPT_PATH = "plate_classifier_resnet18.pth"  # 你以前训好的模型

BATCH_SIZE = 16
NUM_EPOCHS = 20
LR = 1e-4
VAL_RATIO_NEW = 0.2            # 新数据里再切一点出来做 val
OUT_CKPT = "plate_classifier_resnet18_merged_finetuned.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# -----------------------------
# 数据增强 & 预处理
# -----------------------------
train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5]),
])

val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5]),
])


# -----------------------------
# 1. 载入旧数据集 (dataset/train, dataset/val)
# -----------------------------
old_train_ds = datasets.ImageFolder(os.path.join(OLD_DATA_ROOT, "train"),
                                    transform=train_tf)
old_val_ds = datasets.ImageFolder(os.path.join(OLD_DATA_ROOT, "val"),
                                  transform=val_tf)

old_classes = old_train_ds.classes
print("Old dataset classes:", old_classes)

# -----------------------------
# 2. 载入新数据集 (dataset2)
#    注意：这里要求 dataset2 的子文件夹名字和 old_classes 完全一样
# -----------------------------
new_full_ds = datasets.ImageFolder(NEW_DATA_ROOT, transform=train_tf)
new_classes = new_full_ds.classes
print("New dataset2 classes:", new_classes)

if new_classes != old_classes:
    raise ValueError(
        f"❌ Class mismatch between dataset and dataset2.\n"
        f"dataset/train classes: {old_classes}\n"
        f"dataset2 classes:      {new_classes}\n"
        f"请把 dataset2 里的子文件夹改名，使两边完全一致，再重新跑。"
    )

num_classes = len(old_classes)

# 把 dataset2 再切一部分出来做 val
indices = list(range(len(new_full_ds)))
random.seed(42)
random.shuffle(indices)
val_size_new = max(1, int(len(indices) * VAL_RATIO_NEW))
new_val_idx = indices[:val_size_new]
new_train_idx = indices[val_size_new:]

new_train_ds = Subset(new_full_ds, new_train_idx)
new_val_ds = Subset(
    datasets.ImageFolder(NEW_DATA_ROOT, transform=val_tf),
    new_val_idx
)

print(f"Old train: {len(old_train_ds)}, old val: {len(old_val_ds)}")
print(f"New  train: {len(new_train_ds)}, new  val: {len(new_val_ds)}")

# -----------------------------
# 3. 拼成最终的 train / val
# -----------------------------
train_ds = ConcatDataset([old_train_ds, new_train_ds])
val_ds = ConcatDataset([old_val_ds, new_val_ds])

print(f"Total train samples: {len(train_ds)}, total val: {len(val_ds)}")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=0)


# -----------------------------
# 4. 构建模型 & 从旧 ckpt 继续训练
# -----------------------------
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# 加载旧权重
ckpt = torch.load(OLD_CKPT_PATH, map_location=device)
state_dict = ckpt.get("model_state_dict", ckpt)

# 如果之前最后一层名字一样，直接 strict=True 就行；
# 如果有维度不一致/名字对不上，可以只加载 backbone 部分：
backbone_state = {k: v for k, v in state_dict.items()
                  if not k.startswith("fc.")}
missing, unexpected = model.load_state_dict(backbone_state, strict=False)
print("Loaded backbone from old checkpoint.")
print("Missing keys:", missing)
print("Unexpected keys:", unexpected)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


# -----------------------------
# 5. 训练循环
# -----------------------------
def run_epoch(loader, train=True):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_num = 0

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            logits = model(imgs)
            loss = criterion(logits, labels)
            if train:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(logits, 1)
        total_correct += (preds == labels).sum().item()
        total_num += imgs.size(0)

    avg_loss = total_loss / total_num
    acc = total_correct / total_num
    return avg_loss, acc


best_val_acc = 0.0
best_state = None

for epoch in range(1, NUM_EPOCHS + 1):
    t0 = time.time()

    train_loss, train_acc = run_epoch(train_loader, train=True)
    val_loss, val_acc = run_epoch(val_loader, train=False)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state = {
            "model_state_dict": model.state_dict(),
            "class_names": old_classes,
        }

    dt = time.time() - t0
    print(
        f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
        f"Train Loss: {train_loss:.3f}, Acc: {train_acc:.3f} | "
        f"Val Loss: {val_loss:.3f}, Acc: {val_acc:.3f} | "
        f"{dt:.1f}s"
    )

# -----------------------------
# 6. 保存新的 finetuned 模型
# -----------------------------
if best_state is None:
    best_state = {
        "model_state_dict": model.state_dict(),
        "class_names": old_classes,
    }

torch.save(best_state, OUT_CKPT)
print(f"✅ Finetuned model saved to {OUT_CKPT}")
print("Classes order:", old_classes)
