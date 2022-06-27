import logging

import cv2
import mss
import numpy as np
import pyautogui
import pytesseract
import time

from AtitdScripts.utils import almost_equal, extract_match


def get_circles_from_foreground(img, debug=False, intensity_lower=70, intensity_upper=150):
    img = cv2.medianBlur(img, 5)
    cimg = cv2.cvtColor(img, cv.COLOR_GRAY2BGR)

    edge = cv2.canny(img, intensity_lower, intensity_upper)
    circles = cv.HoughCircles(edge,
                              cv.HOUGH_GRADIENT,
                              2,
                              20,
                              param1=50,
                              param2=30,
                              minRadius=90,
                              maxRadius=100)

    if debug:
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                cv.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
                cv.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
                cv.imshow('detected circles', cimg)

    return circles


def matched_pixel_colors(points, expected_colors, monitor_bounds):
    matched = True
    with mss.mss() as sct:
        tmp_img = np.array(sct.grab(monitor_bounds))

        cv2.imshow("Color match at Pixel Check", tmp_img)

        for p in points:
            point = [p[0], p[1]]

            curr_idx = -1
            for idx, val in enumerate(expected_colors):
                if tuple(point) == expected_colors[idx][0]:
                    curr_idx = idx
            if curr_idx == -1:
                print("Couldn't find matching point")
                matched = False
            if not almost_equal(expected_colors[curr_idx][1], tuple(tmp_img[int(point[0]), int(point[1]), :].tolist())):
                print(f"Color at {point} did not equal previous matched. Skipping combination")
                matched = False
    return matched


def template_match_click(template_path, bounds, rightClick=False,
                         retry_attempt_limits=35,
                         sleep_time=0.05,
                         sleepUntilTrue=False, checkUntilGone=False):
    clicked = False
    with mss.mss() as sct:
        retry_attempts = retry_attempt_limits
        tmp_img = np.array(sct.grab(bounds))

        ok = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)

        result = cv2.matchTemplate(tmp_img, ok, cv2.TM_CCOEFF_NORMED)

        mn, mx, mnLoc, mxLoc = cv2.minMaxLoc(result)
        threshold = .8

        if sleepUntilTrue:
            for i in range(retry_attempts):
                tmp_img = np.array(sct.grab(bounds))
                result = cv2.matchTemplate(tmp_img, ok, cv2.TM_CCOEFF_NORMED)
                mn, _mx, mnLoc, mxLoc = cv2.minMaxLoc(result)
                mx = _mx
                if mx > threshold:
                    break
                time.sleep(0.01)

        while mx > threshold:
            MPx, MPy = mxLoc
            trows, tcols = ok.shape[:2]
            pyautogui.moveTo(bounds['left'] + ((MPx + MPx + tcols) / 2), bounds['top'] + ((MPy + MPy + trows) / 2))
            time.sleep(sleep_time)
            if not rightClick:
                pyautogui.click()
            else:
                pyautogui.rightClick()
            clicked = True
            mx = -1 if not checkUntilGone else mx
            if checkUntilGone:
                tmp_img = np.array(sct.grab(bounds))
                result = cv2.matchTemplate(tmp_img, ok, cv2.TM_CCOEFF_NORMED)
                mn, _mx, mnLoc, mxLoc = cv2.minMaxLoc(result)
                mx = _mx
                time.sleep(1)
    return clicked


def maintain_aspect_ratio_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # Grab the image size and initialize dimensions
    dim = None
    (h, w) = image.shape[:2]

    # Return original image if no need to resize
    if width is None and height is None:
        return image

    # We are resizing height if width is none
    if width is None:
        # Calculate the ratio of the height and construct the dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    # We are resizing width if height is none
    else:
        # Calculate the ratio of the width and construct the dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # Return the resized image
    return cv2.resize(image, dim, interpolation=inter)


def save_ocr_bounding_box(boxes, img):
    h, w = img.shape
    _boxes = boxes.split("\n")
    for idx, val in enumerate(_boxes):
        bounds = val.split(" ")
        logging.debug(bounds)
        if len(bounds) == 6:
            _tmp = img.copy()
            logging.debug(f"{int(bounds[1])}, {h - int(bounds[2])}")
            logging.debug(f"{int(bounds[3])}, {h - int(bounds[4])}")
            cv2.rectangle(_tmp,
                          (int(bounds[1]), h - int(bounds[2])),
                          (int(bounds[3]), h - int(bounds[4])),
                          (0, 0, 0),
                          2)
            if idx < len(_boxes) - 1:
                cv2.imwrite(f"temp_{idx + 1}.png", _tmp)
