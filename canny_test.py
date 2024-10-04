import cv2 as cv
import numpy as np

# 边缘提取
def cv_show(name, image):
    cv.imshow(name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()

img = cv.imread("img/niupi.png")
cv_show("niupi.png", img)

# canny边缘检测
v1 = cv.Canny(img, 50, 150)
v2 = cv.Canny(img, 50, 100)

res = np.hstack((v1, v2))
cv_show("res", res)