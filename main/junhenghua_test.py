import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from main.utils.common import cv_show

# 读取图像
img = cv.imread("../img/junheng.png", 0)
plt.hist(img.ravel(), 256)
plt.show()

equ = cv.equalizeHist(img)
plt.hist(equ.ravel(), 256)
plt.show()

res = np.hstack((img, equ))
cv_show('res', res)

clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
res_clahe = clahe.apply(img)

res = np.hstack((img, equ, res_clahe))
cv_show("res", res)