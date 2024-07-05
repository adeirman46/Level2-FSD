import argparse
import cv2
import numpy as np
import platform
import torch
import torch.nn as nn
from pathlib import Path
from ultralytics import YOLO
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.yolo.cfg import get_cfg
from ultralytics.yolo.data import load_inference_source
from ultralytics.yolo.data.augment import LetterBox, classify_transforms
from ultralytics.yolo.utils import DEFAULT_CFG, LOGGER, SETTINGS, callbacks, colorstr, ops
from ultralytics.yolo.utils.checks import check_imgsz, check_imshow
from ultralytics.yolo.utils.files import increment_path
from ultralytics.yolo.utils.torch_utils import select_device, smart_inference_mode
import pyzed.sl as sl

STREAM_WARNING = """
    WARNING ⚠️ stream/video/webcam/dir predict source will accumulate results in RAM unless `stream=True` is passed,
    causing potential out-of-memory errors for large sources or long-running streams/videos.

    Usage:
        results = model(source=..., stream=True)  # generator of Results objects
        for r in results:
            boxes = r.boxes  # Boxes object for bbox outputs
            masks = r.masks  # Masks object for segment masks outputs
            probs = r.probs  # Class probabilities for classification outputs
"""


class BasePredictor:
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        self.args = get_cfg(cfg, overrides)
        project = self.args.project or Path(SETTINGS['runs_dir']) / self.args.task
        name = self.args.name or f'{self.args.mode}'
        self.save_dir = increment_path(Path(project) / name, exist_ok=self.args.exist_ok)
        if self.args.conf is None:
            self.args.conf = 0.25
        self.done_warmup = False
        if self.args.show:
            self.args.show = check_imshow(warn=True)

        self.model = None
        self.data = self.args.data
        self.imgsz = None
        self.device = None
        self.dataset = None
        self.vid_path, self.vid_writer = None, None
        self.plotted_img = None
        self.data_path = None
        self.source_type = None
        self.batch = None
        self.sigmoid = nn.Sigmoid()
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        callbacks.add_integration_callbacks(self)

    def preprocess(self, im):
        if not isinstance(im, torch.Tensor):
            im = np.stack(self.pre_transform(im))
            im = im[..., ::-1].transpose((0, 3, 1, 2))
            im = np.ascontiguousarray(im)
            im = torch.from_numpy(im)
        img = im.to(self.device)
        img = img.half() if self.model.fp16 else img.float()
        img /= 255
        return img

    def pre_transform(self, im):
        same_shapes = all(x.shape == im[0].shape for x in im)
        auto = same_shapes and self.model.pt
        return [LetterBox(self.imgsz, auto=auto, stride=self.model.stride)(image=x) for x in im]

    def write_results(self, idx, results_list, batch):
        p, im, _ = batch
        log_string = ''
        if len(im.shape) == 3:
            im = im[None]
        self.seen += 1
        if self.source_type.webcam or self.source_type.from_img:
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)
        self.data_path = p
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]
        plotted_img = []
        for i, results in enumerate(results_list):
            if isinstance(results, list):
                result = results[idx]
                try:
                    log_string += result.verbose()
                except:
                    pass

                if self.args.save or self.args.show:
                    plot_args = dict(line_width=self.args.line_width,
                                     boxes=self.args.boxes,
                                     conf=self.args.show_conf,
                                     labels=self.args.show_labels)
                    if not self.args.retina_masks:
                        plot_args['im_gpu'] = im[idx]
                        plotted_img.append(result.plot(**plot_args))
            else:
                plotted_img.append(results)
        self.plotted_img = plotted_img
        return log_string

    def postprocess(self, preds, img, orig_img):
        return preds

    def __call__(self, source=None, model=None, stream=False):
        self.stream = stream
        if stream:
            return self.stream_inference(source, model)
        else:
            self.stream_inference(source, model)

    def predict_cli(self, source=None, model=None):
        gen = self.stream_inference(source, model)
        for _ in gen:
            pass

    def setup_source(self, source):
        self.imgsz = check_imgsz(self.args.imgsz, stride=self.model.stride, min_dim=2)
        self.transforms = getattr(self.model.model, 'transforms', classify_transforms(
            self.imgsz[0])) if self.args.task == 'classify' else None
        self.dataset = load_inference_source(source=source, imgsz=self.imgsz, vid_stride=self.args.vid_stride)
        self.source_type = self.dataset.source_type
        if not getattr(self, 'stream', True) and (self.dataset.mode == 'stream' or
                                                  len(self.dataset) > 1000 or
                                                  any(getattr(self.dataset, 'video_flag', [False]))):
            LOGGER.warning(STREAM_WARNING)
        self.vid_path, self.vid_writer = [None] * self.dataset.bs, [None] * self.dataset.bs

    @smart_inference_mode()
    def stream_inference(self, source=None, model=None):
        if self.args.verbose:
            LOGGER.info('')

        if not self.model:
            self.setup_model(model)
        self.setup_source(source if source is not None else self.args.source)

        if self.args.save or self.args.save_txt:
            (self.save_dir / 'labels' if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)
        if not self.done_warmup:
            self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else self.dataset.bs, 3, *self.imgsz))
            self.done_warmup = True

        self.seen, self.windows, self.batch, profilers = 0, [], None, (ops.Profile(), ops.Profile(), ops.Profile())
        self.run_callbacks('on_predict_start')
        for batch in self.dataset:
            self.batch = batch
            path, im0s, vid_cap, s = batch
            visualize = increment_path(self.save_dir / Path(path[0]).stem,
                                       mkdir=True) if self.args.visualize and (not self.source_type.tensor) else False

            with profilers[0]:
                im = self.preprocess(im0s)

            with profilers[1]:
                preds = self.model(im, augment=self.args.augment, visualize=visualize)

            with profilers[2]:
                if self.args.task == 'multi':
                    self.results = []
                    for i, pred in enumerate(preds):
                        if isinstance(pred, tuple):
                            pred = self.postprocess_det(pred, im, im0s)
                            self.results.append(pred)
                        else:
                            pred = self.postprocess_seg(pred)
                            self.results.append(pred)
                else:
                    self.results = self.postprocess(preds, im, im0s)

            n = len(im0s)
            for i in range(n):
                if self.source_type.tensor:
                    continue
                p, im0 = path[i], im0s[i].copy()
                p = Path(p)

                if self.args.verbose or self.args.save or self.args.save_txt or self.args.show:
                    s += self.write_results(i, self.results, (p, im, im0))

                if self.args.save and self.plotted_img is not None:
                    self.save_preds(vid_cap, i, str(self.save_dir / p.name))

            if self.args.verbose:
                LOGGER.info(f'{s}{profilers[1].dt * 1E3:.1f}ms')

    def setup_model(self, model, verbose=True):
        device = select_device(self.args.device, verbose=verbose)
        model = model or self.args.model
        self.args.half &= device.type != 'cpu'
        self.model = AutoBackend(model,
                                 device=device,
                                 dnn=self.args.dnn,
                                 data=self.args.data,
                                 fp16=self.args.half,
                                 fuse=True,
                                 verbose=verbose)
        self.device = device
        self.model.eval()

    def show(self, p):
        im0 = self.plotted_img
        if platform.system() == 'Linux' and p not in self.windows:
            self.windows.append(p)
            cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
        cv2.imshow(str(p), im0)
        cv2.waitKey(500 if self.batch[3].startswith('image') else 1)

    def save_preds(self, vid_cap, idx, save_path):
        if self.source_type.tensor or self.source_type.stream:
            for si, im0 in enumerate(self.plotted_img):
                self.save_pred(im0, save_path, si)
            return

        vid_path, vid_writer = self.vid_path[idx], self.vid_writer[idx]
        if self.dataset.mode == 'image':
            cv2.imwrite(save_path, self.plotted_img[idx])
        else:
            if vid_path != save_path:
                vid_path = save_path
                if isinstance(vid_writer, cv2.VideoWriter):
                    vid_writer.release()
                if vid_cap:
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                else:
                    fps, w, h = 30, self.plotted_img[idx].shape[1], self.plotted_img[idx].shape[0]
                vid_writer = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            vid_writer.write(self.plotted_img[idx])
        self.vid_path[idx], self.vid_writer[idx] = vid_path, vid_writer

    def run_callbacks(self, name):
        for callback in self.callbacks[name]:
            callback(self)


class DetectionPredictor(BasePredictor):
    def postprocess_det(self, preds, img, orig_img):
        if not hasattr(self, 'names'):
            self.names = {k: v for k, v in enumerate(self.model.names)}
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        classes=self.args.classes,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det)
        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if isinstance(orig_img, list) else orig_img.shape
            pred = ops.scale_boxes(img.shape[2:], pred, shape).round()
            if self.args.task == 'multi':
                preds[i] = (pred, preds[1][i])
            else:
                preds[i] = pred
        return preds


class SegmentationPredictor(BasePredictor):
    def postprocess_det(self, preds, img, orig_img):
        if not hasattr(self, 'names'):
            self.names = {k: v for k, v in enumerate(self.model.names)}
        dets, segs = preds
        dets = ops.non_max_suppression(dets,
                                       self.args.conf,
                                       self.args.iou,
                                       classes=self.args.classes,
                                       agnostic=self.args.agnostic_nms,
                                       max_det=self.args.max_det,
                                       nm=segs)
        for i, (det, seg) in enumerate(zip(dets, segs)):
            if len(det):
                shape = orig_img[i].shape if isinstance(orig_img, list) else orig_img.shape
                det[:, :4] = ops.scale_boxes(img.shape[2:], det[:, :4], shape).round()
                seg[:, :4] = ops.scale_boxes(img.shape[2:], seg[:, :4], shape, ratio_pad=0).round()
                segments = ops.process_mask_native(seg, det[:, :4], img.shape[2:], upsample=True)
                segments[:, 1:] = ops.scale_segments(img.shape[2:], segments[:, 1:], shape, normalize=True)
                dets[i], segs[i] = det, segments
        return dets, segs

    def postprocess(self, preds, img, orig_img):
        return self.postprocess_det(preds, img, orig_img)


if __name__ == "__main__":
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

    # Load the predictor
    predictor = SegmentationPredictor(cfg=args.model)

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

            # Run YOLOv8 inference
            results = predictor(bitwise_frame, model=model, stream=True)

            # Check if results is not None
            if results is not None:
                for result in results:
                    boxes = result[0]
                    masks = result[1]
                    for box in boxes:
                        x1, y1, x2, y2, conf, cls = box.tolist()
                        if conf >= args.conf:
                            # Get the center of the bounding box
                            cx = (x1 + x2) // 2
                            cy = (y1 + y2) // 2

                            # Get the depth value at the center of the bounding box
                            depth_value = depth.get_value(int(cx), int(cy))[1]

                            # Display depth information
                            label = f'{predictor.names[int(cls)]} {depth_value:.2f}m'
                            cv2.putText(bitwise_frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                            cv2.rectangle(bitwise_frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

                    # Overlay segmentation masks
                    if masks is not None:
                        for mask in masks:
                            mask = mask[0].cpu().numpy()
                            bitwise_frame[mask == 1] = [0, 255, 0]

            # Display the image
            cv2.imshow("ZED + YOLOv8", bitwise_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Close the camera
    zed.close()
    cv2.destroyAllWindows()
