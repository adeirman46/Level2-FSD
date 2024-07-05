import matplotlib.image as mpimg
import matplotlib.pyplot as plt
# Uncomment the next line for use in a Jupyter notebook
# This enables the interactive matplotlib window
# %matplotlib inline
image = mpimg.imread('/home/irman/Documents/zed-sdk/object detection/custom detector/python/pytorch_yolov8/videos/jalan_tol_new.png')
plt.imshow(image)
plt.show() 