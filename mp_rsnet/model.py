import os
import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

# ---------------- Parameters ---------------- #
DATA_DIR = "/kaggle/input/telugu-dataset"
CSV_PATH = os.path.join(DATA_DIR, "train.csv")
IMG_DIR = os.path.join(DATA_DIR, "data")
BATCH_SIZE = 32
NUM_EPOCHS = 10
IMG_SIZE = 224
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------- Dataset Class ---------------- #
class TeluguDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.data = dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_filename = self.data.iloc[idx, 0]
        label = int(self.data.iloc[idx, 2])
        img_path = os.path.join(self.img_dir, img_filename.replace("data/", ""))
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# ---------------- Load CSV ---------------- #
df = pd.read_csv('/kaggle/input/train-csv/train1.csv')
df[df.columns[0]] = df[df.columns[0]].str.replace("\\", "/", regex=False)

le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label'])

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label_encoded'])

# ---------------- Transforms (with Augmentation) ---------------- #
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# ---------------- Datasets & Loaders ---------------- #
train_dataset = TeluguDataset(train_df, IMG_DIR, train_transform)
val_dataset = TeluguDataset(val_df, IMG_DIR, val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# ---------------- Model ---------------- #
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.fc.in_features, len(le.classes_))
)
model = model.to(DEVICE)

# ---------------- Loss and Optimizer ---------------- #
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# ---------------- Validation Function ---------------- #
def validate(model, val_loader, criterion):
    model.eval()
    total_val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE).long()

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    avg_loss = total_val_loss / len(val_loader)
    return accuracy, avg_loss

# ---------------- Training Loop ---------------- #
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    for images, labels in loop:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE).long()

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        loop.set_postfix(loss=loss.item(), accuracy=100 * correct / total)

    train_acc = 100 * correct / total
    train_loss = total_loss / len(train_loader)

    val_acc, val_loss = validate(model, val_loader, criterion)

    print(f"\n✅ Epoch {epoch+1} completed.")
    print(f"   🔹 Train     — Accuracy: {train_acc:.2f}%, Loss: {train_loss:.4f}")
    print(f"   🔹 Validation — Accuracy: {val_acc:.2f}%, Loss: {val_loss:.4f}")

# ---------------- Save Model ---------------- #
torch.save(model.state_dict(), "resnet18_telugu.pth")
print("✅ Model saved as resnet18_telugu.pth")
