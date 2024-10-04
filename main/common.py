import cv2 as cv

# 边缘提取
def cv_show(name, image):
    cv.imshow(name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()