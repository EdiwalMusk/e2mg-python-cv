import cv2 as cv

from main.utils.common import cv_show

# 轮廓检测
# 读取图像
img = cv.imread("../img/tingchechang.png")
print(img.shape)
cv_show('img', img)

# 灰度处理
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
print(img.shape)
cv_show('gray', gray)

ret, threshold = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
print(threshold.shape)
cv_show('threshold', threshold)

contours, hierarchy = cv.findContours(threshold, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
print(contours)

draw_img = img.copy()
res = cv.drawContours(draw_img, contours, -1, (0, 0, 255), 2)
cv_show('res', res)

cnt = contours[0]
print(cv.contourArea(cnt))
print(cv.arcLength(cnt, True))
