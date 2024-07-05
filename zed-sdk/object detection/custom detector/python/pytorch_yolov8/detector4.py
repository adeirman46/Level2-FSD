import pyzed.sl as sl
import numpy as np
import cv2
import argparse
from ultralytics import YOLO

# Parse command line arguments
parser = argparse.ArgumentParser(description="ZED YOLOv8 Object Detection")
parser.add_argument('--model', type=str, default='yolov8s.pt', help='Path to the YOLOv8 model')
parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold for object detection')
args = parser.parse_args()

# Load the YOLOv8 model
model = YOLO(args.model)

# Create a Camera object
zed = sl.Camera()

# Create an InitParameters object and set configuration parameters
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720  # Use HD720 video mode
init_params.camera_fps = 30  # Set FPS at 30
init_params.coordinate_units = sl.UNIT.METER  # Set the units to meters

# Open the camera
if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
    print("Failed to open ZED camera")
    exit(1)

# Create Mat objects to hold the frames and depth
image = sl.Mat()
depth = sl.Mat()

while True:
    # Grab an image
    if zed.grab() == sl.ERROR_CODE.SUCCESS:
        # Retrieve the left image and depth map
        zed.retrieve_image(image, sl.VIEW.LEFT)
        zed.retrieve_measure(depth, sl.MEASURE.DEPTH)

        # Convert to OpenCV format
        frame = image.get_data()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)

        # Run YOLOv8 inference
        results = model.predict(frame_rgb, conf=args.conf, iou=0.45)

        # Process YOLOv8 results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = box.conf[0]
                cls = int(box.cls[0])

                if conf >= args.conf:
                    # Get the center of the bounding box
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    # Get the depth value at the center of the bounding box
                    depth_value = depth.get_value(cx, cy)[1]

                    # Display depth information
                    label = f'{model.names[cls]} {depth_value:.2f}m'
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Display the image
        cv2.imshow("ZED + YOLOv8", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Close the camera
zed.close()
cv2.destroyAllWindows()
