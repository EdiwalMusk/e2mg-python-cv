import cv2 as cv

from main.utils.common import cv_show

# 读取图像
image = cv.imread('../../img/fanzhendong.png', 0)
cv_show('image', image)
image_check = cv.imread("../../img/malong_hezhao.jpeg", 0)
cv_show('image_check', image_check)

# 初始化ORB检测器
orb = cv.ORB_create()

# 检测关键点
kp, des = orb.detectAndCompute(image, None)
kp_check, des_check = orb.detectAndCompute(image_check, None)

bf = cv.BFMatcher(crossCheck=True)
matches = bf.match(des, des_check)
matches = sorted(matches, key=lambda x: x.distance)

img3 = cv.drawMatches(image, kp, image_check, kp_check, matches[:10], None, flags=2)
cv_show('img3', img3)