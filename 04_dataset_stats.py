import os

IMG_DIR = "face_detection_dataset/images"
LBL_DIR = "face_detection_dataset/labels"

num_images = 0
num_images_with_face = 0
total_boxes = 0

for img_name in os.listdir(IMG_DIR):
    if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    num_images += 1
    base = os.path.splitext(img_name)[0]
    lbl_path = os.path.join(LBL_DIR, base + ".txt")
    if not os.path.exists(lbl_path):
        continue

    with open(lbl_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        if len(lines) > 0:
            num_images_with_face += 1
            total_boxes += len(lines)

avg_boxes = (total_boxes / num_images_with_face) if num_images_with_face > 0 else 0.0

print("Tong so anh:", num_images)
print("So anh co khuon mat:", num_images_with_face)
print("Tong so bounding box:", total_boxes)
print("Trung binh box/anh (chi tinh anh co mat):", round(avg_boxes, 2))
