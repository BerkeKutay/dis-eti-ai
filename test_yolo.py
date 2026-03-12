import torch
import ultralytics.nn.tasks
from ultralytics import YOLO
import cv2

model_path = "runs/detect/dental_strong_m_v4/weights/best.pt"
model = YOLO(model_path)

print("Model loaded successfully.")
print(model.names)

# Test image
image_path = "Dataset/test/images/00929.jpg"  # buraya bir örnek gingiva foto koy

# Inference
results = model(image_path)

# Sonuçları yazdır
for r in results:
    boxes = r.boxes
    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        print(f"Class ID: {cls_id}, Confidence: {conf:.3f}")

# Görselleştir
annotated_frame = results[0].plot()
cv2.imshow("Detection", annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()