import cv2 as cv

from common import cv_show

# 轮廓近似
# 读取图像
img = cv.imread("../img/lunkuo.png")
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
draw_img = img.copy()
res = cv.drawContours(draw_img, cnt, -1, (0, 0, 255), 2)
cv_show('cnt0', res)

# 轮廓近似
epsilon = 0.1 * cv.arcLength(cnt, True)
approx = cv.approxPolyDP(cnt, epsilon, True)

draw_img = img.copy()
res = cv.drawContours(draw_img, [approx], -1, (0, 0, 255), 2)
cv_show('res', res)