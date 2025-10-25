import torch
from torchvision import transforms, models, datasets
from PIL import Image
import torch.nn.functional as F
import pandas as pd
import numpy as np


data_dir = "dataset"               # same folder used in training
model_path = "plate_classifier_resnet18.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================
# 2️⃣ Load class names automatically
# =============================
train_data = datasets.ImageFolder(f"{data_dir}/train")
classes = train_data.classes
print("Detected classes:", classes)

# =============================
# 3️⃣ Load model
# =============================
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# =============================
# 4️⃣ Define transform
# =============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

# =============================
# 5️⃣ Prediction function
# =============================
def predict_image(img_path):
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()
    return probs


img_path = "PlateImages/96.jpg"  # replace with your test image
probs = predict_image(img_path)

df = pd.DataFrame([probs], columns=classes)
df["Predicted"] = [classes[int(np.argmax(probs))]]
print(df.T)
