import cv2 as cv

from main.utils.common import cv_show

# 角点检测
img = cv.imread("../../img/tingchechang.png")
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
dst = cv.cornerHarris(gray, 2, 3, 0.04)

img[dst > 0.01 * dst.max()] = [0, 0, 255]
cv_show('img', img)