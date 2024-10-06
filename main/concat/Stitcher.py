import cv2 as cv
import numpy as np

from main.utils.common import cv_show


class Stitcher:
    def stitch(self, images, ratio=0.75, reprojThresh=0.5, showMatches=True):
        (imageA, imageB) = images
        cv_show('imageA', imageA)
        cv_show('imageB', imageB)
        (kpsA, featuresA) = self.detectAndDescribe(imageA)
        (kpsB, featuresB) = self.detectAndDescribe(imageB)

        M = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)

        if M is None:
            return None

        (matches, H, status) = M
        result = cv.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
        cv_show('result', result)

        result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
        cv_show('result', result)

        if showMatches:
            vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches, status)
            return result, vis

        return result

    def detectAndDescribe(self, image):
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        # 初始化ORB检测器
        orb = cv.ORB_create()
        # 检测关键点
        kps, des = orb.detectAndCompute(gray, None)
        kps = np.float32([kp.pt for kp in kps])
        return kps, des

    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
        matcher = cv.BFMatcher()
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []
        for m in rawMatches:
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        if len(matches) > 4:
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])

            (H, status) = cv.findHomography(ptsA, ptsB, cv.RANSAC, reprojThresh)

            return matches, H, status

    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        # 初始化可视化图片，将A、B图左右连接到一起
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB

        # 联合遍历，画出匹配对
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # 当点对匹配成功时，画到可视化图上
            if s == 1:
                # 画出匹配对
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                cv.line(vis, ptA, ptB, (0, 255, 0), 1)

        # 返回可视化结果
        return vis
