import pyzed.sl as sl
import numpy as np
import cv2
import argparse
from ultralytics import YOLO
import sys

sys.path.insert(0, "/home/irman/Documents/zed-sdk/object detection/custom detector/python/pytorch_yolov8/YOLOv8-multi-task/ultralytics")

number = 3  # input how many tasks in your work

# Parse command line arguments
parser = argparse.ArgumentParser(description="ZED YOLOv8 Object Detection")
parser.add_argument('--model', type=str, default='/home/irman/Documents/zed-sdk/object detection/custom detector/python/pytorch_yolov8/YOLOv8-multi-task/ultralytics/v4s.pt', help='Path to the YOLOv8 model')
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

def birdseye_transform(frame, src_points, dst_points, size):
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    birdseye = cv2.warpPerspective(frame, matrix, size)
    return birdseye

while True:
    # Grab an image
    if zed.grab() == sl.ERROR_CODE.SUCCESS:
        # Retrieve the left image and depth map
        zed.retrieve_image(image, sl.VIEW.LEFT)
        zed.retrieve_measure(depth, sl.MEASURE.DEPTH)

        # Convert to OpenCV format
        frame = image.get_data()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)

        # Get the dimensions of the frame
        height, width = frame.shape[:2]

        # Calculate the coordinates for the rectangular mask
        rect_width = 500  # Adjust as needed for your specific centering
        rect_height = height  # Adjust as needed for your specific centering
        rect_x = (width - rect_width) // 2
        rect_y = (height - rect_height)

        # Create a black mask image
        mask = np.zeros_like(frame_rgb)

        # Draw a white filled rectangle on the mask
        cv2.rectangle(mask, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (255, 255, 255), -1)

        # Perform bitwise AND operation between the frame and the mask
        bitwise_frame = cv2.bitwise_and(frame_rgb, mask)
        # Convert to RGB
        bitwise_frame = cv2.cvtColor(bitwise_frame, cv2.COLOR_BGR2RGB)
        # Resize to 1280x720
        bitwise_frame = cv2.resize(bitwise_frame, (1280, 720))
        
        # Run YOLOv8 inference
        results = model.predict(bitwise_frame, conf=args.conf, iou=0.45, device=[0], imgsz=(384, 672), show_labels=False, save=True, stream=True)
        
        # Check if results is not None
        if results is not None:
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
                        cv2.putText(frame_rgb, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Bird's-eye view transformation parameters
        src_points = np.float32([[rect_x, rect_y], [rect_x + rect_width, rect_y], [rect_x, rect_y + rect_height], [rect_x + rect_width, rect_y + rect_height]])
        dst_points = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
        size = (width, height)
        
        # Generate bird's-eye view
        birdseye_view = birdseye_transform(bitwise_frame, src_points, dst_points, size)
        
        # Display the images
        cv2.imshow("ZED + YOLOv8", cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB))
        cv2.imshow("Bird's Eye View", birdseye_view)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Close the camera
zed.close()
cv2.destroyAllWindows()
