import cv2 as cv

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

