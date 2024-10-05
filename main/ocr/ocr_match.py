import argparse

import cv2 as cv
import numpy as np

from main.utils.common import cv_show, sort_contours

# 设置参数
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-t", "--template", required=True, help="path to template OCR-A image")
args = vars(ap.parse_args())

# 指定信用卡类型
FIRST_NUMBER = {
    "3": "American Express",
    "4": "Vista",
    "5": "MasterCard",
    "6": "Discover Card"
}

# 读取模版文件
img = cv.imread(args["template"])
cv_show('template', img)

# 灰度图
ref = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv_show('ref', ref)

# 二值图像
ref = cv.threshold(ref, 150, 255, cv.THRESH_BINARY_INV)[1]
cv_show('ref', ref)

# 轮廓检测
refCnts, hierarchy = cv.findContours(ref.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cv.drawContours(img, refCnts, -1, (0, 0, 255), 1)
cv_show('img', img)

refCnts = sort_contours(refCnts, method="left-to-right")[0]
digits = {}

# 遍历每一个轮廓
for (i, c) in enumerate(refCnts):
    (x, y, w, h) = cv.boundingRect(c)
    roi = ref[y:y + h, x:x + w]
    roi = cv.resize(roi, (57, 88))

    # 每个数字对应一个模板
    digits[i] = roi

# 初始化卷积核
rectKernel = cv.getStructuringElement(cv.MORPH_RECT, (20, 10))
sqKernel = cv.getStructuringElement(cv.MORPH_RECT, (10, 10))

# 读取输入图像
image = cv.imread(args["image"])
cv_show('image', image)
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
cv_show('gray', gray)

# 礼帽操作
tophat = cv.morphologyEx(gray, cv.MORPH_TOPHAT, rectKernel)
cv_show('tophat', tophat)

# 轮廓提取
gradX = cv.Sobel(tophat, cv.CV_32F, 1, 0, ksize=-1)
gradX = np.absolute(gradX)
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
gradX = gradX.astype("uint8")
cv_show('gradX', gradX)

# 通过闭操作（先膨胀，再腐蚀）将数字连在一起
gradX = cv.morphologyEx(gradX, cv.MORPH_CLOSE, rectKernel)
cv_show('gradX', gradX)

# 阈值操作
thres = cv.threshold(gradX, 0, 255, cv.THRESH_BINARY |
                     cv.THRESH_OTSU)[1]
cv_show('thres', thres)

# 再来一个闭操作
thres = cv.morphologyEx(thres, cv.MORPH_CLOSE, sqKernel)
cv_show('thres', thres)

# 轮廓检测
thres_cnts, thres_hierarchy = cv.findContours(thres.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cur_image = image.copy()
cv.drawContours(cur_image, thres_cnts, -1, (0, 0, 255), 3)
cv_show('cur_image', cur_image)
locs = []

# 遍历轮廓
for (i, c) in enumerate(thres_cnts):
    (x, y, w, h) = cv.boundingRect(c)
    ar = w / float(h)
    if 2.5 < ar < 4.0:
        if 85 < w < 100:
            locs.append((x, y, w, h))
locs = sorted(locs, key=lambda x: x[0])
output = []

# 遍历每一个轮廓中的数字
for (i, (gX, gY, gW, gH)) in enumerate(locs):
    groupOutput = []
    group = gray[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
    cv_show('group', group)
    group = cv.threshold(group, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
    digitCnts, hierarchy = cv.findContours(group.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    digitCnts = sort_contours(digitCnts, method="left-to-right")[0]

    for c in digitCnts:
        (x, y, w, h) = cv.boundingRect(c)
        roi = group[y:y + h, x:x + w]
        roi = cv.resize(roi, (57, 88))
        cv_show('roi', roi)

        # 计算匹配得分
        scores = []

        # 计算每一个得分
        for (digit, digit_roi) in digits.items():
            # 模版匹配
            result = cv.matchTemplate(roi, digit_roi, cv.TM_CCOEFF)
            (_, score, _, _) = cv.minMaxLoc(result)
            scores.append(score)
        # 得到合适的数字
        groupOutput.append(str(np.argmax(scores)))

    # 画出来
    cv.rectangle(image, (gX - 5, gY - 5), (gX + gW + 5, gY + gH + 5), (0, 0, 255), 1)
    cv.putText(image, "".join(groupOutput), (gX, gY - 15), cv.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

    # 得到结果
    output.extend(groupOutput)

print(output)
cv_show("image", image)