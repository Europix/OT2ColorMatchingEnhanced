# train_plate_detector.py
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

CSV_PATH = "dataset_labels.csv"
IMG_DIR   = Path("PlateImages")
SAVE_PATH = "plate_model_resnet18.pth"
BATCH_SIZE, LR, EPOCHS = 8, 1e-4, 30
# 统一网络输入尺寸（与你的相机比例一致）
IN_H, IN_W = 1080, 1920
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class PlateDataset(Dataset):
    def __init__(self, df, img_dir, class_names, transform=None):
        self.df = df.reset_index(drop=True)
        self.dir = img_dir
        self.transform = transform
        self.class_names = class_names
        self.cls2idx = {c:i for i,c in enumerate(class_names)}

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        r   = self.df.iloc[i]
        img = Image.open(self.dir / r.file_name).convert("RGB")
        w0, h0 = img.size

        # 先 resize 到网络输入尺寸
        img_resized = img.resize((IN_W, IN_H), Image.BILINEAR)

        # 同步缩放角点到 (IN_W, IN_H)
        sx, sy = IN_W / w0, IN_H / h0
        coords = np.array([r.x1, r.y1, r.x2, r.y2, r.x3, r.y3, r.x4, r.y4], np.float32)
        coords[0::2] *= sx
        coords[1::2] *= sy
        # 再归一化到 0~1（相对于网络输入尺寸）
        coords_norm = coords / np.array([IN_W, IN_H, IN_W, IN_H, IN_W, IN_H, IN_W, IN_H], np.float32)

        if self.transform:
            img_tensor = self.transform(img_resized)
        else:
            img_tensor = transforms.ToTensor()(img_resized)

        y_cls = self.cls2idx[str(r.plate_type)]
        return img_tensor, torch.tensor(y_cls), torch.tensor(coords_norm, dtype=torch.float32), (w0, h0)

train_tf = transforms.Compose([
    transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.25, hue=0.08),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])
val_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

class PlateNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_f = m.fc.in_features
        m.fc = nn.Identity()
        self.backbone = m
        self.cls = nn.Linear(in_f, num_classes)
        self.reg = nn.Linear(in_f, 8)

    def forward(self, x):
        f = self.backbone(x)
        return self.cls(f), torch.sigmoid(self.reg(f))

def train():
    df = pd.read_csv(CSV_PATH)
    # 用训练集里的 plate_type 列构造“确定顺序”的类别表，并保存到模型
    class_names = sorted(df['plate_type'].astype(str).unique().tolist())

    tr_df, va_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df['plate_type'])
    tr_ds = PlateDataset(tr_df, IMG_DIR, class_names, train_tf)
    va_ds = PlateDataset(va_df, IMG_DIR, class_names, val_tf)
    tr_dl = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    va_dl = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = PlateNet(num_classes=len(class_names)).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    ce    = nn.CrossEntropyLoss()
    l1    = nn.SmoothL1Loss()

    print(f"Classes: {class_names}")
    for e in range(1, EPOCHS+1):
        model.train(); tc=tr=0.0
        for x,y,xy,_ in tr_dl:
            x,y,xy = x.to(DEVICE), y.to(DEVICE), xy.to(DEVICE)
            opt.zero_grad()
            c,p = model(x)
            lc, lr = ce(c,y), l1(p, xy)
            (lc + 3.0*lr).backward(); opt.step()
            tc += lc.item(); tr += lr.item()
        model.eval(); vc=vr=0.0
        with torch.no_grad():
            for x,y,xy,_ in va_dl:
                x,y,xy = x.to(DEVICE), y.to(DEVICE), xy.to(DEVICE)
                c,p = model(x); vc += ce(c,y).item(); vr += l1(p,xy).item()
        print(f"Epoch {e:02d}/{EPOCHS} | Train(C,R): {tc/len(tr_dl):.3f},{tr/len(tr_dl):.3f} | "
              f"Val(C,R): {vc/len(va_dl):.3f},{vr/len(va_dl):.3f}")

    # 保存模型 + 类别表 + 输入尺寸
    torch.save({
        "state_dict": model.state_dict(),
        "class_names": class_names,
        "input_size": (IN_H, IN_W),
    }, SAVE_PATH)
    print(f"✅ saved -> {SAVE_PATH}")

if __name__ == "__main__":
    train()
