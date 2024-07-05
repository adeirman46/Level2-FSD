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
import signal
import time

lock = Lock()
run_signal = False
exit_signal = False

zed_list = []
left_list = []
depth_list = []
timestamp_list = []
thread_list = []
stop_signal = False

def signal_handler(signal, frame):
    global stop_signal
    stop_signal = True
    time.sleep(0.5)
    exit()

def xywh2abcd(xywh, im_shape):
    output = np.zeros((4, 2))

    x_min = (xywh[0] - 0.5*xywh[2])
    x_max = (xywh[0] + 0.5*xywh[2])
    y_min = (xywh[1] - 0.5*xywh[3])
    y_max = (xywh[1] + 0.5*xywh[3])

    output[0][0] = x_min
    output[0][1] = y_min

    output[1][0] = x_max
    output[1][1] = y_min

    output[2][0] = x_max
    output[2][1] = y_max

    output[3][0] = x_min
    output[3][1] = y_max
    return output

def detections_to_custom_box(detections, im0):
    output = []
    for i, det in enumerate(detections):
        xywh = det.xywh[0]

        obj = sl.CustomBoxObjectData()
        obj.bounding_box_2d = xywh2abcd(xywh, im0.shape)
        obj.label = det.cls
        obj.probability = det.conf
        obj.is_grounded = False
        output.append((obj, xywh))  # Return both object and original xywh for drawing

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

            detections = detections_to_custom_box(det, image_net)
            lock.release()
            run_signal = False
        sleep(0.01)

def grab_run(index):
    global stop_signal
    global zed_list
    global timestamp_list
    global left_list
    global depth_list

    runtime = sl.RuntimeParameters()
    while not stop_signal:
        err = zed_list[index].grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS:
            zed_list[index].retrieve_image(left_list[index], sl.VIEW.LEFT)
            zed_list[index].retrieve_measure(depth_list[index], sl.MEASURE.DEPTH)
            timestamp_list[index] = zed_list[index].get_timestamp(sl.TIME_REFERENCE.CURRENT).data_ns
        time.sleep(0.001)
    zed_list[index].close()

def main():
    global image_net, exit_signal, run_signal, detections
    global stop_signal, zed_list, left_list, depth_list, timestamp_list, thread_list

    capture_thread = Thread(target=torch_thread, kwargs={'weights': opt.weights, 'img_size': opt.img_size, "conf_thres": opt.conf_thres})
    capture_thread.start()

    signal.signal(signal.SIGINT, signal_handler)

    print("Running...")
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD720
    init.camera_fps = 60

    name_list = []
    last_ts_list = []
    cameras = sl.Camera.get_device_list()
    index = 0
    for cam in cameras:
        init.set_from_serial_number(cam.serial_number)
        name_list.append("ZED {}".format(cam.serial_number))
        print("Opening {}".format(name_list[index]))
        zed_list.append(sl.Camera())
        left_list.append(sl.Mat())
        depth_list.append(sl.Mat())
        timestamp_list.append(0)
        last_ts_list.append(0)
        status = zed_list[index].open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            print(repr(status))
            zed_list[index].close()
        index += 1

    for index in range(len(zed_list)):
        if zed_list[index].is_opened():
            thread_list.append(Thread(target=grab_run, args=(index,)))
            thread_list[index].start()

    key = ''
    while key != 'q':
        for index in range(len(zed_list)):
            if zed_list[index].is_opened():
                if timestamp_list[index] > last_ts_list[index]:
                    lock.acquire()
                    image_net = left_list[index].get_data()
                    run_signal = True
                    lock.release()

                    while run_signal:
                        sleep(0.001)

                    lock.acquire()
                    for det, xywh in detections:
                        bbox = det.bounding_box_2d
                        x_min = int(bbox[0][0])
                        y_min = int(bbox[0][1])
                        x_max = int(bbox[2][0])
                        y_max = int(bbox[2][1])
                        cv2.rectangle(image_net, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                        x_center = int((bbox[0][0] + bbox[2][0]) / 2)
                        y_center = int((bbox[0][1] + bbox[2][1]) / 2)
                        err, depth_value = depth_list[index].get_value(x_center, y_center)
                        if np.isfinite(depth_value):
                            print("Depth of object {} at ({}, {}): {}MM".format(det.label, x_center, y_center, round(depth_value)))
                    lock.release()

                    cv2.imshow(name_list[index], image_net)
                    last_ts_list[index] = timestamp_list[index]
        key = cv2.waitKey(1) & 0xFF
    cv2.destroyAllWindows()

    stop_signal = True
    for index in range(len(thread_list)):
        thread_list[index].join()

    print("\nFINISH")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov8m.pt', help='model.pt path(s)')
    parser.add_argument('--svo', type=str, default=None, help='optional svo file')
    parser.add_argument('--img_size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf_thres', type=float, default=0.4, help='object confidence threshold')
    opt = parser.parse_args()

    with torch.no_grad():
        main()
