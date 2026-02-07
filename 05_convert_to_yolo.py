import cv2
import os

IMG_DIR = "face_detection_dataset/images"
LBL_DIR = "face_detection_dataset/labels"
YOLO_DIR = "face_detection_dataset/labels_yolo"
os.makedirs(YOLO_DIR, exist_ok=True)

def convert_box_to_yolo(x, y, w, h, img_w, img_h):
    x_center = (x + w / 2.0) / img_w
    y_center = (y + h / 2.0) / img_h
    bw = w / img_w
    bh = h / img_h
    return x_center, y_center, bw, bh

for img_name in sorted(os.listdir(IMG_DIR)):
    if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    img_path = os.path.join(IMG_DIR, img_name)
    base = os.path.splitext(img_name)[0]
    lbl_path = os.path.join(LBL_DIR, base + ".txt")

    img = cv2.imread(img_path)
    if img is None:
        continue
    img_h, img_w = img.shape[:2]

    yolo_path = os.path.join(YOLO_DIR, base + ".txt")
    if not os.path.exists(lbl_path):
        open(yolo_path, "w", encoding="utf-8").close()
        continue

    out_lines = []
    with open(lbl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            x, y, w, h = map(int, line.split())
            xc, yc, bw, bh = convert_box_to_yolo(x, y, w, h, img_w, img_h)
            out_lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

    with open(yolo_path, "w", encoding="utf-8") as f:
        for ln in out_lines:
            f.write(ln + "\n")

print("Da chuyen nhan YOLO vao:", YOLO_DIR)
