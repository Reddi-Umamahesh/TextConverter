# import torch
# from torchvision import transforms, models
# from PIL import Image
# import pandas as pd
# import os
# import torch.nn as nn

# # --- CONFIGURATION ---
# MODEL_PATH = 'resnet18_telugu.pth'  # Path to trained model
# TRAIN_CSV_PATH = r'C:\mp_rsnet\data\train.csv'  # Path to train.csv
# IMAGE_PATH = 'Img/pa1.jpeg'  # Path to image for prediction

# # --- LOAD LABELS FROM TRAIN.CSV ---
# train_df = pd.read_csv(TRAIN_CSV_PATH, header=None, names=["image_path", "label"])
# train_df = train_df[train_df['label'] != 'label']  # remove header row if mistakenly present
# train_df['label'] = train_df['label'].astype(str).str.strip()

# label_list = sorted(train_df['label'].unique())
# idx_to_label = {i: label for i, label in enumerate(label_list)}

# print(f"✅ Labels loaded from train.csv: {len(label_list)} (should be 557)")

# # --- LOAD MODEL ---
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model = models.resnet18()
# model.fc = nn.Sequential(
#     nn.Dropout(0.3),
#     nn.Linear(model.fc.in_features, len(label_list))  # this must match trained model
# )
# model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
# model = model.to(device)
# model.eval()

# # --- IMAGE PREPROCESSING ---
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
# ])

# def predict_image(image_path):
#     image = Image.open(image_path).convert('RGB')

#     input_tensor = transform(image).unsqueeze(0).to(device)

#     with torch.no_grad():
#         output = model(input_tensor)
#         predicted_idx = torch.argmax(output, dim=1).item()
#         predicted_label = idx_to_label[predicted_idx]
    
#     return predicted_label

# # --- RUN PREDICTION ---
# predicted_char = predict_image(IMAGE_PATH)
# print(f"📝 Predicted Telugu character: {predicted_char}")


import sys
import torch
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import torch.nn as nn

# Receive image path dynamically
image_path = sys.argv[1]  # <<<<<<<<<<image_path = sys.argv[1]
MODEL_PATH = 'C:/TextConverter/mp_rsnet/resnet18_telugu.pth'
TRAIN_CSV_PATH = 'C:/TextConverter/mp_rsnet/data/train.csv'

# --- Load labels ---
train_df = pd.read_csv(TRAIN_CSV_PATH, header=None, names=["image_path", "label"])
train_df = train_df[train_df['label'] != 'label']
train_df['label'] = train_df['label'].astype(str).str.strip()
label_list = sorted(train_df['label'].unique())
idx_to_label = {i: label for i, label in enumerate(label_list)}

# --- Load model ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet18()
model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.fc.in_features, len(label_list))
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# --- Preprocessing ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# --- Prediction ---
def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        predicted_idx = torch.argmax(output, dim=1).item()
        predicted_label = idx_to_label[predicted_idx]
    return predicted_label

# predicted_char = predict_image(image_path)
# print(predicted_char)  # <<<<<< Only output the prediction
predicted_char = predict_image(image_path)

# Encode output correctly
sys.stdout.buffer.write(predicted_char.encode('utf-8'))