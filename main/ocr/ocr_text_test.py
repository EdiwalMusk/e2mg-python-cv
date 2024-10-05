import argparse

import numpy as np
import cv2 as cv

from main.utils.common import resize, cv_show, four_point_transform

# 设置参数
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
args = vars(ap.parse_args())

height = 800

image = cv.imread(args["image"])
ratio = image.shape[0] / height
orig = image.copy()

image = resize(orig, height=height)

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
gray = cv.GaussianBlur(gray, (5, 5), 0)
edged = cv.Canny(gray, 20, 100)

# 展示预处理结果
print("STEP1:边缘检测")
cv_show('image', gray)
cv_show('edged', edged)

cnts = cv.findContours(edged.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)[0]
cnts = sorted(cnts, key=cv.contourArea, reverse=True)[:5]

screenCnt = None
# 遍历轮廓
for c in cnts:
    peri = cv.arcLength(c, True)
    approx = cv.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:
        screenCnt = approx
        break

cv.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
cv_show('image', image)

# 透视变换
warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
ref = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
cv_show('ref', resize(ref, height=500))
ref = cv.threshold(ref, 150, 255, cv.THRESH_BINARY)[1]

cv_show('ref', resize(ref, height=500))