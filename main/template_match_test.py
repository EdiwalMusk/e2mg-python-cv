import cv2 as cv

from common import cv_show

img = cv.imread("../img/liuyifei.jpeg")
img_face = cv.imread("../img/liuyifei_face.jpeg")

h, w = img_face.shape[:2]
print(h, w)

res = cv.matchTemplate(img, img_face, cv.TM_SQDIFF)
print(res.shape)

min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
bottom_right = (min_loc[0] + w, min_loc[1] + h)

img2 = img.copy()
cv.rectangle(img2, min_loc, bottom_right, 255, 2)
cv_show("img2", img2)
