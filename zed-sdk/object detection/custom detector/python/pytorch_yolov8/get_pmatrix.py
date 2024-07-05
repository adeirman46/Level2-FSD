import pyzed.sl as sl
import numpy as np

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

# Enable positional tracking
positional_tracking_params = sl.PositionalTrackingParameters()
if zed.enable_positional_tracking(positional_tracking_params) != sl.ERROR_CODE.SUCCESS:
    print("Failed to enable positional tracking")
    exit(1)

# Get camera parameters
cam_params = zed.get_camera_information().camera_configuration.calibration_parameters
left_cam_params = cam_params.left_cam
right_cam_params = cam_params.right_cam

# Create the intrinsic matrix K
Kl = np.array([[left_cam_params.fx, 0, left_cam_params.cx],
              [0, left_cam_params.fy, left_cam_params.cy],
              [0, 0, 1]])
Kr = np.array([[right_cam_params.fx, 0, right_cam_params.cx],
              [0, right_cam_params.fy, right_cam_params.cy],
              [0, 0, 1]])


# Capture pose data
pose = sl.Pose()

# Get the current pose
if zed.get_position(pose) != sl.POSITIONAL_TRACKING_STATE.OK:
    print("Failed to get camera position")
    exit(1)

# Extract rotation (R) and translation (t)
R = pose.get_rotation_matrix().r.T
t = pose.get_translation().get()

# Baseline between left and right cameras
baseline = 120.0 / 1000  # 120 mm to meters

# Construct the world to camera transformation
world2cam = np.hstack((R, np.dot(-R, t).reshape(3, -1)))
world2cam_right = np.hstack((R, np.dot(-R, t - np.array([baseline, 0, 0])).reshape(3, -1)))

# Construct the projection matrix P
P_left = np.dot(Kl, world2cam)
# Construct the projection matrix P
P_right = np.dot(Kr, world2cam_right)

# Print the projection matrix
print("Projection Matrix P_left:\n", P_left)
print("Projection Matrix P_right:\n", P_right)

# Close the camera
zed.close()
