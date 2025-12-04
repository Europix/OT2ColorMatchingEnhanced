import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import pandas as pd

# ============ CONFIG ============
MODEL_PATH = "plate_model_resnet18.pth"
IMG_PATH = r"PlateImages\24.jpg"  # 你要测试的图片路径
IMG_SIZE = (1080, 1920)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = ['1', '6', '12', '24', '48', '96', '384', 'nothing']
DEBUG_VISUALIZE = True
# ================================


# ============ MODEL ============
class PlateNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.cls_head = nn.Linear(in_feats, num_classes)
        self.reg_head = nn.Linear(in_feats, 8)

    def forward(self, x):
        feats = self.backbone(x)
        cls_logits = self.cls_head(feats)
        coords = torch.sigmoid(self.reg_head(feats))
        return cls_logits, coords


# ============ LOAD MODEL ============
def load_model():
    model = PlateNet(num_classes=len(CLASS_NAMES))
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


# ============ TRANSFORMS ============
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
])


# ============ PREDICT FUNCTION ============
def predict_plate(img_path):
    model = load_model()
    img_pil = Image.open(img_path).convert("RGB")
    img_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        cls_logits, coords = model(img_tensor)
        probs = torch.softmax(cls_logits, dim=1)[0].cpu().numpy()
        pred_class = CLASS_NAMES[np.argmax(probs)]
        coords = coords[0].cpu().numpy()

    w, h = img_pil.size
    coords_denorm = (coords * np.array([w, h, w, h, w, h, w, h])).reshape(-1, 2)

    print("Predicted Plate Type:", pred_class)
    print("Class Probabilities:")
    print(pd.DataFrame([probs], columns=CLASS_NAMES, index=["Probabilities"]))
    print("Corner Coordinates (pixels):")
    print(coords_denorm)

    if DEBUG_VISUALIZE:
        img_cv = np.array(img_pil)
        pts = coords_denorm.astype(int)
        cv2.polylines(img_cv, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.imwrite("prediction_debug.jpg", cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR))
        print("✅ Saved visualization to prediction_debug.jpg")

    return pred_class, coords_denorm


if __name__ == "__main__":
    predict_plate(IMG_PATH)
