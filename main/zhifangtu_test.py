import cv2 as cv
import matplotlib.pyplot as plt

# 读取图像
img = cv.imread("../img/cat.jpg", 0)
hist = cv.calcHist([img], [0], None, [256], [0, 256])
print(hist.shape)

# 绘制图像
plt.hist(img.ravel(), 256)
plt.show()

# 绘制彩色曲线
img = cv.imread("../img/cat.jpg")
color = ('b', 'g', 'r')
for i, col in enumerate(color):
    histr = cv.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(histr, color=col)
    plt.xlim([0, 256])
plt.show()
