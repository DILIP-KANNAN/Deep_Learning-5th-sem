# ============================================================
# 📦 Imports
# ============================================================
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# ============================================================
# ⚙️ Step 1: Setup Device
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# ============================================================
# 📁 Step 2: Dataset and Transformations
# ============================================================
# 👉 Change this to your dataset path
base_dir = r"C:\Users\dilip\datasets\Deep Learning 5th sem\lab expts\PetImages"
   
IMG_SIZE = 224
BATCH_SIZE = 32
VAL_SPLIT = 0.2

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Standard ImageNet normalization
                         std=[0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(root=base_dir, transform=transform)
num_classes = len(dataset.classes)
print(f"✅ Classes found: {dataset.classes}")

# ============================================================
# 📊 Step 3: Split Train and Validation (No manual files)
# ============================================================
train_size = int((1 - VAL_SPLIT) * len(dataset))
val_size = len(dataset) - train_size
train_data, val_data = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

# ============================================================
# 🧩 Step 4: Model Builder Function
# ============================================================
def build_model(model_name):
    if model_name == "vgg16":
        base_model = models.vgg16(weights="IMAGENET1K_V1")
        base_model.classifier[6] = nn.Linear(4096, 1)

    elif model_name == "resnet50":
        base_model = models.resnet50(weights="IMAGENET1K_V1")
        base_model.fc = nn.Linear(base_model.fc.in_features, 1)

    elif model_name == "googlenet":
        base_model = models.googlenet(weights="IMAGENET1K_V1")
        base_model.fc = nn.Linear(base_model.fc.in_features, 1)
    
    else:
        raise ValueError("Unknown model name")

    return base_model.to(device)

# ============================================================
# 🧠 Step 5: Training and Evaluation Utilities
# ============================================================
def train_model(model, optimizer, criterion, train_loader, val_loader, epochs=5):
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = torch.round(torch.sigmoid(outputs))
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        history["train_loss"].append(running_loss / len(train_loader))
        history["val_loss"].append(val_loss / len(val_loader))
        history["val_acc"].append(val_acc)

        print(f"\n📈 Epoch {epoch+1}: Train Loss={running_loss/len(train_loader):.4f} | "
              f"Val Loss={val_loss/len(val_loader):.4f} | Val Acc={val_acc*100:.2f}%\n")

    return history

# ============================================================
# 🧪 Step 6: Train All Models
# ============================================================
models_to_train = ["vgg16", "resnet50", "googlenet"]
histories = {}

for name in models_to_train:
    print(f"\n🚀 Training {name.upper()} ...\n")
    model = build_model(name)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    history = train_model(model, optimizer, criterion, train_loader, val_loader, epochs=5)
    histories[name] = history

# ============================================================
# 📊 Step 7: Plot Results
# ============================================================
def plot_history(histories, metric):
    plt.figure(figsize=(10, 5))
    for name, hist in histories.items():
        plt.plot(hist[metric], label=f"{name.upper()}")
    plt.xlabel("Epochs")
    plt.ylabel(metric.replace("_", " ").title())
    plt.title(f"{metric.replace('_', ' ').title()} Comparison")
    plt.legend()
    plt.grid(True)
    plt.show()

plot_history(histories, "val_acc")
plot_history(histories, "val_loss")

# ============================================================
# 🧾 Step 8: Final Evaluation
# ============================================================
for name, hist in histories.items():
    best_acc = max(hist["val_acc"])
    print(f"{name.upper()} Best Validation Accuracy: {best_acc*100:.2f}%")
