import os
import shutil
import random
import yaml

# Configuration
SOURCE_IMG_DIR = "output/images"
SOURCE_LBL_DIR = "output/labels"
DATASET_DIR = "dataset"
SPLIT_RATIO = 0.8  # 80% Training, 20% Validation


def prepare_yolo_dataset():
    # 1. Create Folder Structure
    for split in ['train', 'val']:
        os.makedirs(f"{DATASET_DIR}/{split}/images", exist_ok=True)
        os.makedirs(f"{DATASET_DIR}/{split}/labels", exist_ok=True)

    # 2. Get list of files
    files = [f for f in os.listdir(SOURCE_IMG_DIR) if f.endswith(('.jpg', 'jpeg'))]
    random.shuffle(files)

    split_index = int(len(files) * SPLIT_RATIO)
    train_files = files[:split_index]
    val_files = files[split_index:]

    print(f"Dataset: {len(train_files)} Train, {len(val_files)} Val")

    # 3. Move Files
    for filename in files:
        # Determine if it goes to 'train' or 'val'
        split = 'train' if filename in train_files else 'val'

        # Move Image
        src_img = os.path.join(SOURCE_IMG_DIR, filename)
        dst_img = os.path.join(DATASET_DIR, split, "images", filename)
        shutil.copy(src_img, dst_img)

        # Move Label
        name_without_ext = os.path.splitext(filename)[0]
        lbl_name = name_without_ext + ".txt"
        src_lbl = os.path.join(SOURCE_LBL_DIR, lbl_name)
        dst_lbl = os.path.join(DATASET_DIR, split, "labels", lbl_name)

        if os.path.exists(src_lbl):
            shutil.copy(src_lbl, dst_lbl)

    # 4. Create data.yaml
    yaml_content = {
        'path': os.path.abspath(DATASET_DIR),  # Absolute path
        'train': 'train/images',
        'val': 'val/images',
        'names': {0: 'stop_sign'}
    }

    with open(f"{DATASET_DIR}/data.yaml", 'w') as f:
        yaml.dump(yaml_content, f)

    print(f"Dataset ready, check {os.path.abspath(DATASET_DIR)}")
    print("data.yaml created successfully")


if __name__ == "__main__":
    prepare_yolo_dataset()
