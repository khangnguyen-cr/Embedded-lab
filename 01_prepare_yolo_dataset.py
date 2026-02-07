import os
import random
import shutil
from pathlib import Path

# =========================
# INPUT từ lab trước
# =========================
SRC_IMG_DIR = "face_detection_dataset/images"
SRC_LBL_DIR = "face_detection_dataset/labels_yolo"  # tạo ở bước convert_to_yolo

# =========================
# OUTPUT chuẩn YOLO
# =========================
OUT_DIR = "dataset_yolo"
SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

CLASS_NAMES = ["face"]  # 1 lớp

def ensure_dirs():
    for p in [
        f"{OUT_DIR}/images/train", f"{OUT_DIR}/images/val", f"{OUT_DIR}/images/test",
        f"{OUT_DIR}/labels/train", f"{OUT_DIR}/labels/val", f"{OUT_DIR}/labels/test",
    ]:
        os.makedirs(p, exist_ok=True)

def list_images():
    exts = (".jpg", ".jpeg", ".png")
    imgs = [f for f in os.listdir(SRC_IMG_DIR) if f.lower().endswith(exts)]
    imgs.sort()
    return imgs

def copy_pair(img_name, split):
    src_img = Path(SRC_IMG_DIR) / img_name
    lbl_name = Path(img_name).stem + ".txt"
    src_lbl = Path(SRC_LBL_DIR) / lbl_name

    dst_img = Path(OUT_DIR) / "images" / split / img_name
    dst_lbl = Path(OUT_DIR) / "labels" / split / lbl_name

    shutil.copy2(src_img, dst_img)
    # nếu label không tồn tại, tạo file rỗng để đồng bộ
    if src_lbl.exists():
        shutil.copy2(src_lbl, dst_lbl)
    else:
        dst_lbl.write_text("", encoding="utf-8")

def write_yaml():
    yaml_path = Path(OUT_DIR) / "data.yaml"
    # đường dẫn tương đối, Ultralytics sẽ resolve theo nơi chạy
    content = f"""path: {OUT_DIR}
train: images/train
val: images/val
test: images/test

names:
  0: {CLASS_NAMES[0]}
"""
    yaml_path.write_text(content, encoding="utf-8")
    print("Saved:", yaml_path)

def main():
    assert abs((TRAIN_RATIO + VAL_RATIO + TEST_RATIO) - 1.0) < 1e-6
    ensure_dirs()

    imgs = list_images()
    if len(imgs) == 0:
        raise RuntimeError("Khong tim thay anh trong SRC_IMG_DIR")

    random.seed(SEED)
    random.shuffle(imgs)

    n = len(imgs)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    n_test = n - n_train - n_val

    train_list = imgs[:n_train]
    val_list = imgs[n_train:n_train + n_val]
    test_list = imgs[n_train + n_val:]

    for name in train_list:
        copy_pair(name, "train")
    for name in val_list:
        copy_pair(name, "val")
    for name in test_list:
        copy_pair(name, "test")

    write_yaml()
    print("Total images:", n)
    print("Train/Val/Test:", len(train_list), len(val_list), len(test_list))
    print("Output:", OUT_DIR)

if __name__ == "__main__":
    main()
