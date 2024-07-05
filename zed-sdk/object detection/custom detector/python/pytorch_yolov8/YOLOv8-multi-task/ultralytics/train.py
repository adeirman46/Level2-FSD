import sys
sys.path.insert(0, "/home/irman/Documents/zed-sdk/object detection/custom detector/python/pytorch_yolov8/YOLOv8-multi-task/ultralytics")
# 现在就可以导入Yolo类了
from ultralytics import YOLO

# Load a model
model = YOLO('/home/irman/Documents/zed-sdk/object detection/custom detector/python/pytorch_yolov8/YOLOv8-multi-task/ultralytics/v4s.pt', task='multi')  # build a new model from YAML
# model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
# model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

# Train the model
model.train(data='/home/irman/Documents/zed-sdk/object detection/custom detector/python/pytorch_yolov8/YOLOv8-multi-task/ultralytics/datasets/bdd-multi.yaml', batch=8, epochs=300, imgsz=(640,640), device=[0], name='yolopm', val=True, task='multi',classes=[0,2,3,4,5,6,9,10,11],combine_class=[0,2,3,4,5,6,9],single_cls=True)
