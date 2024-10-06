import cv2 as cv

from Stitcher import Stitcher
from main.utils.common import cv_show

imageA = cv.imread("../../img/concat/msg_right.png")
imageB = cv.imread("../../img/concat/msg_left.png")

stitcher = Stitcher()
(result, vis) = stitcher.stitch([imageA, imageB], showMatches=True)

cv_show("imageA", imageA)
cv_show("imageB", imageB)
cv_show("result", result)
cv_show("vis", vis)