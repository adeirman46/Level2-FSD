#!/usr/bin/env python3

import sys
import numpy as np
import argparse
import torch
import cv2
import pyzed.sl as sl
from ultralytics import YOLO

from threading import Lock, Thread
from time import sleep

import ogl_viewer.viewer as gl
import cv_viewer.tracking_viewer as cv_viewer

lock = Lock()
run_signal = False
exit_signal = False

def xywh2abcd(xywh, im_shape):
    output = np.zeros((4, 2))
    x_min = (xywh[0] - 0.5*xywh[2]) * im_shape[1]
    x_max = (xywh[0] + 0.5*xywh[2]) * im_shape[1]
    y_min = (xywh[1] - 0.5*xywh[3]) * im_shape[0]
    y_max = (xywh[1] + 0.5*xywh[3]) * im_shape[0]
    output[0][0] = x_min
    output[0][1] = y_min
    output[1][0] = x_max
    output[1][1] = y_min
    output[2][0] = x_max
    output[2][1] = y_max
    output[3][0] = x_min
    output[3][1] = y_max
    return output

def detections_to_custom_box(detections, im_shape):
    output = []
    for i, det in enumerate(detections):
        xywh = det.xywh[0]
        obj = sl.CustomBoxObjectData()
        obj.bounding_box_2d = xywh2abcd(xywh, im_shape)
        obj.label = int(det.cls)
        obj.probability = float(det.conf)
        obj.is_grounded = False
        obj.unique_object_id = sl.generate_unique_id()  # Generate a unique ID for each object
        output.append(obj)
    return output

def torch_thread(weights, img_size, conf_thres=0.2, iou_thres=0.45):
    global image_net, exit_signal, run_signal, detections

    print("Initializing Network...")
    model = YOLO(weights)

    while not exit_signal:
        if run_signal:
            lock.acquire()
            img = cv2.cvtColor(image_net, cv2.COLOR_BGRA2RGB)
            det = model.predict(img, save=False, imgsz=img_size, conf=conf_thres, iou=iou_thres)[0].cpu().numpy().boxes
            detections = detections_to_custom_box(det, img.shape)
            lock.release()
            run_signal = False
        sleep(0.01)

def main():
    global image_net, exit_signal, run_signal, detections

    capture_thread = Thread(target=torch_thread, kwargs={'weights': opt.weights, 'img_size': opt.img_size, "conf_thres": opt.conf_thres})
    capture_thread.start()

    print("Initializing Camera...")
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.coordinate_units = sl.UNIT.METER
    init_params.sdk_verbose = True
    status = zed.open(init_params)

    if status != sl.ERROR_CODE.SUCCESS:
        print(f"Camera Open : {repr(status)}. Exit program.")
        exit()

    image_left_tmp = sl.Mat()

    print("Enabling Positional Tracking...")
    positional_tracking_parameters = sl.PositionalTrackingParameters()
    zed.enable_positional_tracking(positional_tracking_parameters)

    print("Enabling Object Detection...")
    obj_param = sl.ObjectDetectionParameters()
    obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.CUSTOM_BOX_OBJECTS
    obj_param.enable_tracking = True
    zed.enable_object_detection(obj_param)

    objects = sl.Objects()
    obj_runtime_param = sl.ObjectDetectionRuntimeParameters()

    # Display setup
    camera_info = zed.get_camera_information()
    display_resolution = sl.Resolution(min(camera_info.camera_configuration.resolution.width, 1280),
                                       min(camera_info.camera_configuration.resolution.height, 720))
    image_scale = [display_resolution.width / camera_info.camera_configuration.resolution.width,
                   display_resolution.height / camera_info.camera_configuration.resolution.height]
    image_left_ocv = np.zeros((display_resolution.height, display_resolution.width, 4), dtype=np.uint8)

    while not exit_signal:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image_left_tmp, sl.VIEW.LEFT)
            image_net = image_left_tmp.get_data()
            run_signal = True

            while run_signal:
                sleep(0.001)

            zed.ingest_custom_box_objects(detections)
            zed.retrieve_objects(objects, obj_runtime_param)

            zed.retrieve_image(image_left_tmp, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
            np.copyto(image_left_ocv, image_left_tmp.get_data())

            # Display detections and distances
            for obj in objects.object_list:
                if obj.tracking_state == sl.OBJECT_TRACKING_STATE.OK:
                    bbox = obj.bounding_box_2d
                    cv2.rectangle(image_left_ocv, 
                                  (int(bbox[0][0] * image_scale[0]), int(bbox[0][1] * image_scale[1])),
                                  (int(bbox[2][0] * image_scale[0]), int(bbox[2][1] * image_scale[1])),
                                  (0, 255, 0), 2)
                    distance = np.linalg.norm(obj.position)
                    cv2.putText(image_left_ocv, f"Dist: {distance:.2f}m", 
                                (int(bbox[0][0] * image_scale[0]), int(bbox[0][1] * image_scale[1] - 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow("ZED Object Detection", image_left_ocv)
            key = cv2.waitKey(10)
            if key == 27:  # ESC key
                exit_signal = True

    zed.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov8m.pt', help='model.pt path(s)')
    parser.add_argument('--img_size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf_thres', type=float, default=0.4, help='object confidence threshold')
    opt = parser.parse_args()

    with torch.no_grad():
        main()