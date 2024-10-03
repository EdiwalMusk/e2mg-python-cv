import cv2 as cv

# 读取图像
img = cv.imread("img/cat.jpg")
print(img.shape)

cat = img[0:50, 0:200]

r,g,b = cv.split(cat)

cur_img = img.copy()
cur_img[:,:,1] = 0
cur_img[:,:,2] = 0
cv.imshow('R', cur_img)

cv.imshow("cat", cat)
cv.waitKey(0)
cv.destroyAllWindows()