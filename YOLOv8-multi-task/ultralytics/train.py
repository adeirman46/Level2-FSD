import sys
sys.path.insert(0, "/home/irman/Documents/YOLOv8-multi-task/ultralytics")
# 现在就可以导入Yolo类了
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8-bdd-v4-one-dropout-individual-n.yaml', task='multi')  # build a new model from YAML
# .load('/home/irman/Documents/YOLOv8-multi-task/ultralytics/v4s.pt')
# model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
# model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights
# model = YOLO('ultralytics/models/v8/yolov8-bdd.yaml', task='multi').load('/home/irman/Documents/YOLOv8-multi-task/ultralytics/v4s.pt')

# Train the model
model.train(data='/home/irman/Documents/YOLOv8-multi-task/ultralytics/datasets/bdd-multi.yaml', batch=2, epochs=5, imgsz=(640,640), device=0, name='yolopm', val=True, task='multi', classes=[0,2,3,4,5,6,9,10,11], single_cls=False)
