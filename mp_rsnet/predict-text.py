import os
from PIL import Image
import torch
from torchvision import transforms, models
import torch.nn as nn
import pandas as pd
import cv2
import numpy as np

# ---------------- Config ---------------- #
MODEL_PATH = 'C:\TextConverter\mp_rsnet\resnet18_telugu.pth'
LABEL_CSV = r'C:\TextConverter\mp_rsnet\data\train.csv'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 224

# ---------------- Load Labels ---------------- #
train_df = pd.read_csv(LABEL_CSV, header=None, names=["image_path", "label"])
train_df = train_df[train_df['label'] != 'label']
train_df['label'] = train_df['label'].astype(str).str.strip()
label_list = sorted(train_df['label'].unique())
idx_to_label = {i: label for i, label in enumerate(label_list)}

# ---------------- Load Model ---------------- #
def load_model(model_path, device):
    model = models.resnet18()
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, len(label_list))
    )
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

model = load_model(MODEL_PATH, DEVICE)

# ---------------- Preprocessing Function ---------------- #
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def preprocess_char(img):
    img_pil = Image.fromarray(img).convert("RGB")
    tensor = transform(img_pil).unsqueeze(0).to(DEVICE)
    return tensor

# ---------------- Predict Single Character ---------------- #
def predict_character(model, img_tensor):
    with torch.no_grad():
        output = model(img_tensor)
        pred_idx = torch.argmax(output, dim=1).item()
        return idx_to_label[pred_idx]

# ---------------- Segment Image Into Characters ---------------- #
def segment_characters(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find external contours
    contours, _ = cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    chars = []

    for cnt in sorted(contours, key=lambda c: cv2.boundingRect(c)[0]):
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 10 and h > 10:  # filter out noise
            char_img = threshed[y:y+h, x:x+w]
            
            # Pad to make square
            pad_size = abs(h - w) // 2
            if h > w:
                char_img = cv2.copyMakeBorder(char_img, 0, 0, pad_size, pad_size, cv2.BORDER_CONSTANT, value=0)
            else:
                char_img = cv2.copyMakeBorder(char_img, pad_size, pad_size, 0, 0, cv2.BORDER_CONSTANT, value=0)
            
            # Resize
            char_img = cv2.resize(char_img, (224, 224), interpolation=cv2.INTER_AREA)
            chars.append(char_img)

    return chars[::-1]  # sometimes left-right gets flipped, try reversing


# ---------------- Recognize Telugu Text ---------------- #
def recognize_telugu_text(image_path):
    characters = segment_characters(image_path)
    recognized = ""
    for char_img in characters:
        tensor = preprocess_char(char_img)
        label = predict_character(model, tensor)
        recognized += label
    return recognized

# ---------------- Run ---------------- #
if __name__ == "__main__":
    input_image = "Img/5.jpeg"  # replace with your image path
    result = recognize_telugu_text(input_image)
    print(f"\n📝 Recognized Telugu Text: {result}")
