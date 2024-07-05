import sys
sys.path.insert(0, "/home/irman/Documents/zed-sdk/object detection/custom detector/python/pytorch_yolov8/YOLOv8-multi-task/ultralytics")

from ultralytics import YOLO
import cv2

img = cv2.imread('/home/irman/Documents/zed-sdk/object detection/custom detector/python/pytorch_yolov8/videos/jalan_tol_new.png')


number = 3 #input how many tasks in your work
model = YOLO('/home/irman/Documents/zed-sdk/object detection/custom detector/python/pytorch_yolov8/YOLOv8-multi-task/runs/multi/yolopm/weights/best.pt', task='multi')  # Validate the model
results = model.predict(source=img, imgsz=(384,672), device=[0],name='v4_daytime', save=True, conf=0.25, iou=0.45, show_labels=False, stream=True, save_txt=True)
print('Pred results: ', results)

# for r in results[0]:
#     boxes = r.boxes  # Boxes object for bbox outputs
#     masks = r.masks  # Masks object for segment masks outputs
#     probs = r.probs  # Class probabilities for classification outputs
#     print('boxes:', boxes)
#     print('masks:', masks)
#     print('probs:', probs)

# results[0].save_txt()  # Save results to /inference/output




