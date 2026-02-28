import os
import shutil
import random

RAW_DIR =r"C:\Users\DELL\OneDrive\Desktop\CNN_PROJECT\data\raw"
OUT_DIR =r"C:\Users\DELL\OneDrive\Desktop\CNN_PROJECT\data\processed"

train_ratio = 0.7
val_ratio = 0.2

classes = os.listdir(RAW_DIR)

for cls in classes:
    cls_path = os.path.join(RAW_DIR, cls)
    images = os.listdir(cls_path)
    random.shuffle(images)

    total = len(images)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))

    splits = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    for split in splits:
        split_path = os.path.join(OUT_DIR, split, cls)
        os.makedirs(split_path, exist_ok=True)

        for img in splits[split]:
            src = os.path.join(cls_path, img)
            dst = os.path.join(split_path, img)
            shutil.copy(src, dst)

print("Dataset split done successfully")
