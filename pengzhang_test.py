import cv2 as cv
import numpy as np

# 腐蚀测试
# 读取图像
img = cv.imread("img/niupi.png")
cv.imshow('fushi', img)
cv.waitKey(0)
cv.destroyAllWindows()

kernel = np.ones((10,10),np.uint8)
erosion = cv.erode(img,kernel,iterations = 1)

cv.imshow("erosion",erosion)
cv.waitKey(0)
cv.destroyAllWindows()

dilate = cv.dilate(erosion,kernel,iterations = 1)
cv.imshow("dilate",dilate)
cv.waitKey(0)
cv.destroyAllWindows()