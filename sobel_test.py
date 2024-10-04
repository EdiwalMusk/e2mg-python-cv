import cv2 as cv
import numpy as np

# 边缘提取
def cv_show(name, image):
    cv.imshow(name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()

img = cv.imread("img/niupi.png")
cv_show("niupi.png", img)

sobelx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=3)
sobelx = cv.convertScaleAbs(sobelx)
cv_show("sobelx", sobelx)
sobely = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=3)
sobely = cv.convertScaleAbs(sobely)
cv_show("sobely", sobely)
sobel = cv.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
cv_show("sobel", sobel)

scharrx = cv.Scharr(img, cv.CV_64F, 1, 0)
scharrx = cv.convertScaleAbs(scharrx)
cv_show("scharrx", scharrx)
scharry = cv.Scharr(img, cv.CV_64F, 0, 1)
scharry = cv.convertScaleAbs(scharry)
cv_show("scharry", scharry)
scharr = cv.addWeighted(scharrx, 0.5, scharry, 0.5, 0)
cv_show("scharr", scharr)

# 拉普拉斯算子
laplacian = cv.Laplacian(img, cv.CV_64F)
laplacian = cv.convertScaleAbs(laplacian)

res = np.hstack((sobel, scharr, laplacian))
cv_show("res", res)