import cv2
import os

IMG_DIR = "face_detection_dataset/images"
LBL_DIR = "face_detection_dataset/labels"

def read_boxes(lbl_path):
    boxes = []
    if not os.path.exists(lbl_path):
        return boxes
    with open(lbl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            x, y, w, h = map(int, line.split())
            boxes.append((x, y, w, h))
    return boxes

for img_name in sorted(os.listdir(IMG_DIR)):
    if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    img_path = os.path.join(IMG_DIR, img_name)
    base = os.path.splitext(img_name)[0]
    lbl_path = os.path.join(LBL_DIR, base + ".txt")

    frame = cv2.imread(img_path)
    if frame is None:
        continue

    boxes = read_boxes(lbl_path)
    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.putText(frame, f"{img_name} | faces: {len(boxes)}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Visualize Labels", frame)
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()
