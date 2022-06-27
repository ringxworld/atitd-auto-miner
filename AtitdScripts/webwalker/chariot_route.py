import os
import time

import cv2
import mss
import numpy as np
import pyautogui

from AtitdScripts.image import template_match_click


def handle_chariot_route(coordinates, current, monitor_bounds, use_travel_time):
    pyautogui.moveTo(960, 540)
    time.sleep(0.1)
    pyautogui.click()

    template_match_click(os.path.join(os.getcwd(), 'images',
                                      f'{coordinates[current]}.png'),
                         monitor_bounds,
                         sleepUntilTrue=True)
    template_match_click(os.path.join(os.getcwd(), 'images', 'travel_to.png'),
                         monitor_bounds,
                         rightClick=True)

    mx = 0.0
    threshold = 0.8

    while mx < threshold:
        template_match_click(os.path.join(os.getcwd(), 'images', 'travel_to.png'),
                             monitor_bounds)

        with mss.mss() as sct:
            tmp_img = np.array(sct.grab(monitor_bounds))
            if not use_travel_time:
                ok = cv2.imread(os.path.join(os.getcwd(), 'images', 'travel_now_for_free.png'),
                                cv2.IMREAD_UNCHANGED)
                result = cv2.matchTemplate(tmp_img, ok, cv2.TM_CCOEFF_NORMED)

                mn, _mx, mnLoc, mxLoc = cv2.minMaxLoc(result)
                mx = _mx
                print("Sleeping 5 seconds... waiting for free travel")

                time.sleep(5)
            else:
                ok = cv2.imread(os.path.join(os.getcwd(), 'images', 'travel_using_travel_time.png'),
                                cv2.IMREAD_UNCHANGED)
                result = cv2.matchTemplate(tmp_img, ok, cv2.TM_CCOEFF_NORMED)

                mn, _mx, mnLoc, mxLoc = cv2.minMaxLoc(result)
                mx = _mx
                if mx < threshold:
                    ok = cv2.imread(os.path.join(os.getcwd(), 'images', 'travel_now_for_free.png'),
                                    cv2.IMREAD_UNCHANGED)
                    result = cv2.matchTemplate(tmp_img, ok, cv2.TM_CCOEFF_NORMED)

                    mn, _mx, mnLoc, mxLoc = cv2.minMaxLoc(result)
                    if _mx > threshold:
                        mx = _mx
                time.sleep(0.2)

    MPx, MPy = mxLoc
    trows, tcols = ok.shape[:2]
    pyautogui.moveTo(0 + ((MPx + MPx + tcols) / 2),
                     0 + ((MPy + MPy + trows) / 2))
    pyautogui.click()
    time.sleep(1)

    if use_travel_time:
        template_match_click(os.path.join(os.getcwd(), 'images', 'yes.png'),
                             monitor_bounds)
        time.sleep(2)

    template_match_click(os.path.join(os.getcwd(), 'images', 'travel_to.png'),
                         monitor_bounds,
                         rightClick=True)
    time.sleep(0.5)
    template_match_click(os.path.join(os.getcwd(), 'images', 'chariot_stop.png'),
                         monitor_bounds,
                         rightClick=True,
                         checkUntilGone=True)
    time.sleep(0.5)
    template_match_click(os.path.join(os.getcwd(), 'images', 'pinned_icon.png'),
                         monitor_bounds,
                         checkUntilGone=True)
    time.sleep(1)
    return
