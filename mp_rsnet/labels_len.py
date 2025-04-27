import pandas as pd

# --- CONFIGURATION ---
TRAIN_CSV_PATH = r'C:\mp_rsnet\data\train1.csv'   # Path to your train.csv
LABELS_CSV_PATH = r'C:\mp_rsnet\data\labels.csv' # Path to your labels.csv

# --- LOAD LABELS ---
train_df = pd.read_csv(TRAIN_CSV_PATH, header=None, names=["image_path", "label"])
train_labels = set(train_df['label'].astype(str).str.strip())

labels_df = pd.read_csv(LABELS_CSV_PATH, header=None, names=["label"])
all_labels = set(labels_df['label'].astype(str).str.strip())

# --- ANALYSIS ---
print(f"✅ Total unique labels in train.csv: {len(train_labels)}")
print(f"✅ Total unique labels in labels.csv: {len(all_labels)}")

# Labels in labels.csv but not in train.csv (extra label(s))
extra_labels = all_labels - train_labels
# Labels in train.csv but not in labels.csv (possible errors)
missing_labels = train_labels - all_labels

# --- RESULTS ---
print("\n🔍 Extra label(s) in labels.csv but not in train.csv:")
print(extra_labels if extra_labels else "None")

print("\n❌ Missing label(s) in labels.csv (present in train.csv but not in labels.csv):")
print(missing_labels if missing_labels else "None")
