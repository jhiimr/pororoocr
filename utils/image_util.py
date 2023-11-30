import cv2
import numpy as np
import imutils


def find_rect_contour(image, width = 800, ksize=(5,5), min_threshold=60, max_threshold=200, debug=False):
    resized = imutils.resize(image, width=width)
    ratio = image.shape[1] / float(resized.shape[1])
    
    # 이미지를 grayscale로 변환하고 blur를 적용 -> 모서리를 찾기위한 이미지 연산
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, ksize, 0)
    edged = cv2.Canny(blurred, min_threshold, max_threshold)
    
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)   # 외곽선 검출
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    
    findCnt = None
    
    # 정렬된 contours를 반복문으로 수행하며 4개의 꼭지점을 갖는 도형을 검출
    for c in cnts:
        peri = cv2.arcLength(c, True)                       # 외곽선 길이 구하기
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)     # 외곽선 근사화(단순화)
    
        if len(approx) == 4:        # 가장 큰 사각형
            findCnt = approx
            break
    
    # 만약 추출한 윤곽이 없을 경우 원본 이미지 반환
    if not findCnt.any():
        return findCnt

    contour = findCnt * ratio
    contour = contour.round(0)
    contour = contour.astype('int')

    if debug:
        from pororo.plt_util import plt_imshow
        cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
        plt_imshow("Edge", image, (8, 5))

    return contour


def find_text_contour(image, position_Y, ksize=(5,5), min_threshold=60, max_threshold=200, debug=False):
    # === Edge 검출을 위한 이미지 전처리
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, ksize, 0)
    edged = cv2.Canny(blurred, min_threshold, max_threshold)

    # === 텍스트 외곽선 검출
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    grab = dict()
    grab["index"], grab["minY"], grab["maxY"] = None, None, None

    for i, cnts in enumerate(contours):
        max_y = cnts.max(axis=0)[0][1]

        if max_y <= position_Y:
            if not grab["maxY"] or grab["maxY"] < max_y:
                grab["index"] = i
                grab["minY"] = cnts.min(axis=0)[0][1]
                grab["maxY"] = max_y
            elif max_y == grab["maxY"]:
                min_y = cnts.min(axis=0)[0][1]
                if min_y < grab["minY"]:
                    grab["index"] = i
                    grab["minY"] = min_y
                    grab["maxY"] = max_y
    
    if debug:
        from pororo.plt_util import plt_imshow
        cv2.drawContours(image, [contours[grab["index"]]], -1, (0, 255, 0), 2)
        plt_imshow("ROI", image, (8, 5))

    return grab["minY"], grab["maxY"]
    
