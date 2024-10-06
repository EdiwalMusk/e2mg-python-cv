import cv2 as cv
import numpy as np

from main.utils.common import cv_show

# 读取图像
image = cv.imread('../../img/malong.jpg')
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# 初始化ORB检测器
orb = cv.ORB_create()

# 检测关键点
keypoints, descriptors = orb.detectAndCompute(gray, None)
print(np.array(keypoints).shape)
print(np.array(descriptors).shape)

# 绘制关键点
output_image = cv.drawKeypoints(gray, keypoints, None, color=(0, 255, 0), flags=0)

# 显示图像
cv_show('output', output_image)

# 计算特征
kp, des = orb.compute(gray, keypoints)
print(np.array_equal(keypoints, kp))
print(np.array_equal(descriptors, des))