# Internship-SelfDrivingCar
Self driving car level 2 (longitudinal and lateral) from depth camera, GNSS, IMU, and wheel odometry

## baseline_line_finding_yolov5
```python
conda create -n fsd_env python=3.8
conda activate fsd_env
cd baseline_lane_finding_yolov5/
pip install -r requirements.txt
python track.py --source close_calls.mp4 --show-vid # from video
python track.py --source close_calls.mp4 --show-vid # from webcam
```

## yolop
```python
conda create --name fsd --clone fsd_env_3
conda activate fsd
cd yolop/
pip install -r requirements.txt
python tools/demo.py --source videos/jalan_tol.mp4 --device 0 # from video, --device 0 is cuda
python tools/demo.py --source 0 --device 0 # from webcam and using cuda
```
