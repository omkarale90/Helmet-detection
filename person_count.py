import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
image_path = "group.jpg"
img = cv2.imread(image_path)
results = model(img)

count = 0
for r in results:
    for box in r.boxes:
        cls = int(box.cls[0])
        label = model.names[cls]
        conf = float(box.conf[0])

        if label == "person" and conf > 0.5:
            count += 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


cv2.putText(img, f"People count: {count}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

cv2.imshow("Detected People", img)
cv2.waitKey(0)
cv2.destroyAllWindows()







