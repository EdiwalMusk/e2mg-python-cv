import cv2 as cv


# 边缘提取
def cv_show(name, image):
    cv.imshow(name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()


# 排序轮廓
def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0

    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    boundingBoxes = [cv.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))

    return cnts, boundingBoxes
