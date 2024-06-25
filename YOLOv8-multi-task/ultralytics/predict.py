import sys
sys.path.insert(0, "/home/irman/Documents/magang/Level2-FSD/YOLOv8-multi-task/ultralytics")

from ultralytics import YOLO


number = 3 #input how many tasks in your work
model = YOLO('/home/irman/Documents/magang/Level2-FSD/YOLOv8-multi-task/ultralytics/v4s.pt', task='multi')  # Validate the model
model.predict(source='/home/irman/Documents/magang/Level2-FSD/YOLOv8-multi-task/videos/jalan_tol_new.png', imgsz=(645,645), device=0,name='detection_', save=True, conf=0.25, iou=0.45, show_labels=False)
