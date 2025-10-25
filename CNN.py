import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm
import pandas as pd
import numpy as np

# ==============================
# 1. Configuration
# ==============================
data_dir = "dataset"
batch_size = 8
num_epochs = 15
learning_rate = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================
# 2. Data pipeline
# ==============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

train_data = datasets.ImageFolder(f"{data_dir}/train", transform=transform)
val_data = datasets.ImageFolder(f"{data_dir}/val", transform=transform)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size)

classes = train_data.classes
print("Classes:", classes)

# ==============================
# 3. Model setup
# ==============================
model = models.resnet18(weights="IMAGENET1K_V1")
model.fc = nn.Linear(model.fc.in_features, len(classes))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ==============================
# 4. Training loop
# ==============================
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    print(f"Epoch {epoch+1}/{num_epochs} | "
          f"Loss: {total_loss/len(train_loader):.4f} | "
          f"Val Acc: {100*correct/total:.2f}%")

# ==============================
# 5. Save model
# ==============================
torch.save(model.state_dict(), "plate_classifier_resnet18.pth")
print("✅ Model saved as plate_classifier_resnet18.pth")

# ==============================
# 6. Inference: probability matrix
# ==============================
def predict_probs(model, img_paths):
    model.eval()
    probs_list = []
    transform_eval = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])
    for path in tqdm(img_paths, desc="Predicting"):
        from PIL import Image
        img = Image.open(path).convert("RGB")
        x = transform_eval(img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = torch.softmax(model(x), dim=1)
        probs = outputs.cpu().numpy().flatten()
        probs_list.append(probs)
    return np.array(probs_list)

# Example usage:
# test_images = ["PlatelImages/24.jpg", "PlatelImages/48.jpg"]
# prob_matrix = predict_probs(model, test_images)
# df = pd.DataFrame(prob_matrix, columns=classes, index=test_images)
# print(df)
# df.to_csv("prediction_matrix.csv")
