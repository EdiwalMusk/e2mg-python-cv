import cv2 as cv

from main.utils.common import cv_show

# 金字塔测试
# 读取图像
img = cv.imread("../img/cat.jpg")
cv_show('cat', img)
print(img.shape)

# 上采样
up = cv.pyrUp(img)
cv_show("up", up)
print(up.shape)

# 下采样
down = cv.pyrDown(img)
cv_show("down", down)
print(down.shape)

# 拉普拉斯金字塔
down1 = cv.pyrDown(img)
up1 = cv.pyrUp(down1)
img1 = img - up1
cv_show('img1', img1)