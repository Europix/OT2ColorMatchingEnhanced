# testmodel.py

import sys
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ---------------------------------------------------------
# 类别顺序：必须和训练脚本完全一致！！
# ---------------------------------------------------------
classes = ["nothing", "1", "6", "12", "24", "48", "96", "384"]
num_classes = len(classes)
print("Using classes:", classes)

# ---------------------------------------------------------
# 设备
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------------------------------------------------
# 构建和训练时一致的模型结构
# ---------------------------------------------------------
model = models.resnet18(weights=None)
in_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(in_features, num_classes)
)

state_path = "plate_model_resnet18_colorRing.pth"  # <<< 改成你的权重文件名
state = torch.load(state_path, map_location=device)
model.load_state_dict(state)
model.to(device)
model.eval()

# ---------------------------------------------------------
# 图像预处理（和 val_tf 一致）
# ---------------------------------------------------------
IMG_SIZE = 400
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.2)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
])

def predict_one(img_path: str):
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)  # [1,3,H,W]

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = torch.argmax(probs).item()

    pred_class = classes[pred_idx]
    print(f"Image: {img_path}")
    print(f"Pred class: {pred_class}")
    print("Probabilities:")
    for i, c in enumerate(classes):
        print(f"  {c:>7}: {probs[i].item():.4f}")


if __name__ == "__main__":
    # 用法：python testmodel.py path/to/image.jpg
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        # 不想用命令行就直接在这里写死一张图的路径
        img_path = r"C:\Users\mercu\Desktop\ipys\ASRfinal\OT2ColorMatchingEnhanced\dataset\train\48\WIN_20251121_16_08_42_Pro.jpg"  # <<< 改成你的图片

    predict_one(img_path)
