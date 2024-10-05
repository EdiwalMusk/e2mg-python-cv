import cv2 as cv
import numpy as np

from main.utils.common import cv_show

img = cv.imread("../img/niupi.png")
cv_show("niupi.png", img)

# canny边缘检测
v1 = cv.Canny(img, 50, 150)
v2 = cv.Canny(img, 50, 100)

res = np.hstack((v1, v2))
cv_show("res", res)