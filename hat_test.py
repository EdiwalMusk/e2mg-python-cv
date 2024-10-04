import cv2 as cv
import numpy as np

# 腐蚀测试
# 读取图像
img = cv.imread("img/niupi.png")
cv.imshow('fushi', img)
cv.waitKey(0)
cv.destroyAllWindows()

kernel = np.ones((5,5),np.uint8)

# 礼帽
tophat = cv.morphologyEx(img, cv.MORPH_TOPHAT, kernel)
cv.imshow('tophat', tophat)
cv.waitKey(0)
cv.destroyAllWindows()

# 黑帽
blackhat = cv.morphologyEx(img, cv.MORPH_BLACKHAT, kernel)
cv.imshow('blackhat', blackhat)
cv.waitKey(0)
cv.destroyAllWindows()