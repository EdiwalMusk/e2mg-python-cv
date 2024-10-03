import cv2 as cv

# 读取图像
img = cv.imread("img/cat.jpg")

# 边界填充
top_size, bottom_size, left_size, right_size = (50, 50, 50, 50)
replicate = cv.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType=cv.BORDER_REPLICATE)
cv.imshow("replicate", replicate)

cv.waitKey(0)
cv.destroyAllWindows()