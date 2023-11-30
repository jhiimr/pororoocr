import cv2
import numpy as np
import platform
from PIL import ImageFont, ImageDraw, Image
from matplotlib import pyplot as plt
import imutils
from imutils.perspective import four_point_transform


def plt_imshow(title='image', img=None, figsize=(16, 10)):
    plt.figure(figsize=figsize)

    if type(img) is str:
        img = cv2.imread(img)

    if type(img) == list:
        if type(title) == list:
            titles = title
        else:
            titles = []

            for i in range(len(img)):
                titles.append(title)

        for i in range(len(img)):
            if len(img[i].shape) <= 2:
                rgbImg = cv2.cvtColor(img[i], cv2.COLOR_GRAY2RGB)
            else:
                rgbImg = cv2.cvtColor(img[i], cv2.COLOR_BGR2RGB)

            plt.subplot(1, len(img), i + 1), plt.imshow(rgbImg)
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])

    else:
        if len(img.shape) < 3:
            rgbImg = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.imshow(rgbImg)
        plt.title(title)
        plt.xticks([]), plt.yticks([])
    print("[ arrive! ]")
    plt.show(block=False)


def put_text(image, text, x, y, color=(0, 255, 0), font_size=22):
    if type(image) == np.ndarray:
        color_coverted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(color_coverted)

    if platform.system() == 'Darwin':
        font = 'AppleGothic.ttf'
    elif platform.system() == 'Windows':
        font = 'malgun.ttf'

    image_font = ImageFont.truetype(font, font_size)
    font = ImageFont.load_default()
    draw = ImageDraw.Draw(image)

    draw.text((x, y), text, font=image_font, fill=color)

    numpy_image = np.array(image)
    opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

    return opencv_image


def find_rect_contour(image, width = 800, ksize=(5,5), min_threshold=60, max_threshold=200):
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

    cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
    cv2.imshow("contour", image)
    cv2.waitKey(0)

    return contour

def make_scan_image(image, width=800, ksize=(5,5), min_threshold=60, max_threshold=200, debug=False):
    if isinstance(image, str):
        image = cv2.imread(image)
    org_image = image.copy()

    image = imutils.resize(image, width=width)
    ratio = org_image.shape[1] / float(image.shape[1])
    
    # 이미지를 grayscale로 변환하고 blur를 적용 -> 모서리를 찾기위한 이미지 연산
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
        return findCnt, org_image

    output = image.copy()
    cv2.drawContours(output, [findCnt], -1, (0, 255, 0), 2)

    outline_image = four_point_transform(org_image, findCnt.reshape(4, 2) * ratio)

    if debug:
        plt_imshow(["Origin", "Contours", "Transform"], [org_image, output, outline_image])

    return findCnt, outline_image


if __name__ == "__main__":
    img_path = "image_text.png"
    img = cv2.imread(img_path)
    contour = find_rect_contour(img)
    # contour, outline_image = make_scan_image(img_path)