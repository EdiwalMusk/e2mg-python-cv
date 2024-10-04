import cv2 as cv
import numpy as np

from common import cv_show

img = cv.imread("../img/chongfu.jpeg")
img_face = cv.imread("../img/chongfu_unit.jpeg")

h, w = img_face.shape[:2]
print(h, w)

# 模版可匹配多次
res = cv.matchTemplate(img, img_face, cv.TM_CCOEFF_NORMED)
print(res.shape)

threshold = 0.8
loc = np.where(res >= threshold)
for pt in zip(*loc[::-1]):
    bottom_right = pt[0] + w, pt[1] + h
    cv.rectangle(img, pt, bottom_right, (0, 0, 255), 2)

cv_show("img", img)