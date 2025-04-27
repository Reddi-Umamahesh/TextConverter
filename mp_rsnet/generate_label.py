import os
import pandas as pd

# Define base directory
base_dir = r'C:\mp_rsnet\data'

# Load labels.csv
labels_df = pd.read_csv(os.path.join(base_dir, 'labels.csv'))
if len(labels_df.columns) == 2:
    labels_df.columns = ['path_pattern', 'telugu_label']
else:
    raise ValueError("labels.csv must have exactly 2 columns")

# Normalize label mapping (case-insensitive keys)
label_mapping = {k.strip().lower(): v for k, v in zip(labels_df['path_pattern'], labels_df['telugu_label'])}

# Initialize lists
image_paths = []
labels = []
warned_paths = set()

# Traverse dataset
for root, _, files in os.walk(base_dir):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            full_path = os.path.join(root, file)
            relative_path = os.path.relpath(full_path, base_dir).replace("\\", "/")  # image path like achulu/a/1.jpg

            parts = relative_path.split('/')
            path_pattern = ""

            # Detect pattern based on folder structure
            if parts[0].lower() == 'guninthamulu' and len(parts) >= 3:
                # guninthamulu/ka/ana/1.jpg → guninthamulu/ka/ana
                path_pattern = f"{parts[0]}/{parts[1]}/{parts[2]}"
            elif len(parts) >= 2:
                # achulu/a/1.jpg → achulu/a
                path_pattern = f"{parts[0]}/{parts[1]}"
            else:
                continue  # skip malformed paths

            # Lookup label
            telugu_char = label_mapping.get(path_pattern.lower())

            if telugu_char:
                image_paths.append(f"data/{relative_path}")
                labels.append(telugu_char)
            else:
                if path_pattern not in warned_paths:
                    print(f"⚠️ Missing label for path pattern: {path_pattern}")
                    warned_paths.add(path_pattern)

# Save train.csv
df = pd.DataFrame({'image_path': image_paths, 'label': labels})
output_csv = os.path.join(base_dir, 'train1.csv')
df.to_csv(output_csv, index=False)

print(f"\n✅ train.csv generated at: {output_csv}")
print(f"Total images: {len(df)}")
print(df.head())
