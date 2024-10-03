import cv2 as cv

# 读取视频
vc = cv.VideoCapture("D:\迅雷下载\死侍BD中英双字.电影天堂.www.dy2018.com.mp4")
while True:
    ret, frame = vc.read()
    if frame is None:
        break
    if ret:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        cv.imshow('frame', gray)
        if cv.waitKey(1) & 0xFF == 27:
            break
vc.release()
cv.destroyAllWindows()