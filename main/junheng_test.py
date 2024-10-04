import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from main.common import cv_show

# 读取图像
img = cv.imread("../img/cat.jpg")

mask = np.zeros(img.shape[:2], np.uint8)
print(type(mask))
print(mask.shape)

# 结果是?-?行 ?-?列
mask[50:150, 50:150] = 255
cv_show("mask", mask)

# 掩码图片输出
masked_img = cv.bitwise_and(img, img, mask=mask)
cv_show("masked_img", masked_img)

hist_full = cv.calcHist([img], [0], None, [256], [0, 256])
hist_mask = cv.calcHist([img], [0], mask, [256], [0, 256])

plt.subplot(221)
plt.imshow(img, 'gray')
plt.subplot(222)
plt.imshow(mask, 'gray')
plt.subplot(223)
plt.imshow(masked_img, 'gray')
plt.subplot(224)
plt.plot(hist_full)
plt.plot(hist_mask)
plt.xlim([0, 256])
plt.show()
