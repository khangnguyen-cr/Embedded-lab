import os
import cv2

DATA_DIR = "dataset_yolo"
IMG_DIR = os.path.join(DATA_DIR, "images", "train")
LBL_DIR = os.path.join(DATA_DIR, "labels", "train")

def read_yolo(lbl_path):
    boxes = []
    if not os.path.exists(lbl_path):
        return boxes
    with open(lbl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cid, xc, yc, w, h = line.split()
            boxes.append((int(cid), float(xc), float(yc), float(w), float(h)))
    return boxes

def yolo_to_xyxy(xc, yc, w, h, img_w, img_h):
    x1 = int((xc - w/2) * img_w)
    y1 = int((yc - h/2) * img_h)
    x2 = int((xc + w/2) * img_w)
    y2 = int((yc + h/2) * img_h)
    return x1, y1, x2, y2

for name in sorted(os.listdir(IMG_DIR)):
    if not name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    img_path = os.path.join(IMG_DIR, name)
    lbl_path = os.path.join(LBL_DIR, os.path.splitext(name)[0] + ".txt")

    img = cv2.imread(img_path)
    if img is None:
        continue
    h, w = img.shape[:2]

    boxes = read_yolo(lbl_path)
    for (cid, xc, yc, bw, bh) in boxes:
        x1, y1, x2, y2 = yolo_to_xyxy(xc, yc, bw, bh, w, h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"id={cid}", (x1, max(0, y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("YOLO Labels", img)
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()

