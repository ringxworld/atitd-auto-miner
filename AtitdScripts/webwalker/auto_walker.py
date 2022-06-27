import logging
import os
import time

import cv2
import mss
import numpy as np
import pyautogui
import pydirectinput
from pytesseract import pytesseract, Output

from AtitdScripts.image import template_match_click
from AtitdScripts.utils import extract_match, manhattan
from AtitdScripts.webwalker.WebTreeStructure import WebWalkerTree
from AtitdScripts.webwalker.chariot_route import handle_chariot_route


class AutoWalker(object):

    def __init__(self, web, end_coordinate, **kwargs):
        self.running = True

        self.web = WebWalkerTree(node_definitions=web)

        self.monitor_bounds = {"top": 0, "left": 0, "width": 1920, "height": 1080}
        self.ocr_bounds = {"top": 40, "left": 855, "width": 210, "height": 45}
        if kwargs.get('ocr_bounds'):
            self.ocr_bounds = kwargs.get('ocr_bounds')

        self.end_coordinate = end_coordinate
        if kwargs.get("end_coord"):
            x, y = kwargs.get("end_coord")
            self.end_coordinate = [int(x), int(y)]

        self.training_dir = None
        if kwargs.get("training_dir"):
            training_dir = kwargs.get("training_dir")
            self.training_dir = training_dir
            if not os.path.exists(os.path.join(training_dir, "ocr_debug")):
                os.mkdir(os.path.join(training_dir, "ocr_debug"))

        ocr_result = self.get_coordinates(self.ocr_bounds, r'-?\d+\.?\d*')
        if not ocr_result:
            return
        x, y = ocr_result

        self.coordinates = self.web.get_best_path_from_coordinates(start=[x, y], end=self.end_coordinate)
        self.current = 0

        self.curr_press_dir = None
        self.prev_press_dir = None

        self.prev_x = x
        self.prev_y = y

        pydirectinput.keyUp("left")
        pydirectinput.keyUp("right")
        pydirectinput.keyUp("up")
        pydirectinput.keyUp("down")

        print(f"Walking to: {self.coordinates[self.current]}")

    def run(self):
        while self.running:
            self.run_handler()
        if not self.running:
            pydirectinput.keyUp("left")
            pydirectinput.keyUp("right")
            pydirectinput.keyUp("up")
            pydirectinput.keyUp("down")

    def run_handler(self):
        while self.running:
            # map coordinates, come up with a better pattern later
            ocr_result = self.get_coordinates(self.ocr_bounds, r'-?\d+\.?\d*')

            if not ocr_result:
                return

            x, y = ocr_result

            if isinstance(self.coordinates[self.current], str):
                handle_chariot_route(self.coordinates, self.current, self.monitor_bounds, use_travel_time=True)
                self.current += 1
                self.prev_x = self.coordinates[self.current][0]
                self.prev_y = self.coordinates[self.current][1]
                return

            x, y = self.resolve_invalid_coordinate(x, y)
            x, y = self.resolve_bad_parse(x, y)

            logging.info(f"Current:{x},{y}, moving towards: {self.coordinates[self.current]}")

            if (x, y) == tuple(self.coordinates[self.current]):
                logging.info(f"Walking to: {self.coordinates[self.current]}")
                self.curr_press_dir = None
                self.current += 1
                if self.current > len(self.coordinates) - 1:
                    self.running = False
                return

            curr_press_dir = None

            if y < self.coordinates[self.current][1]:
                curr_press_dir = "up"
            if y > self.coordinates[self.current][1]:
                curr_press_dir = "down"
            if x < self.coordinates[self.current][0]:
                curr_press_dir = "right"
            if x > self.coordinates[self.current][0]:
                curr_press_dir = "left"

            shouldPress_lr = False
            shouldPress_ud = False

            if 0 < abs(self.coordinates[self.current][1] - y) < 3:
                shouldPress_ud = True
            if 0 < self.coordinates[self.current][0] - x < 3:
                shouldPress_lr = True

            is_prev_string = isinstance(self.coordinates[self.current - 1], str)
            if is_prev_string:
                return self.run_standard_walk(x, y, curr_press_dir, shouldPress_lr, shouldPress_ud)

            dist = manhattan(np.asarray(self.coordinates[self.current - 1]), np.asarray(self.coordinates[self.current]))
            if not dist:
                return self.run_standard_walk(x, y, curr_press_dir, shouldPress_lr, shouldPress_ud)
            if dist > 5:
                return self.run_standard_walk(x, y, curr_press_dir, shouldPress_lr, shouldPress_ud)

            if self.current > 0 and curr_press_dir:
                pydirectinput.keyUp("left")
                pydirectinput.keyUp("right")
                pydirectinput.keyUp("up")
                pydirectinput.keyUp("down")
                pydirectinput.press(curr_press_dir)
                time.sleep(0.05)

    def run_standard_walk(self, x, y, curr_press_dir, shouldPress_lr , shouldPress_ud):
        if curr_press_dir != self.prev_press_dir:
            self.prev_press_dir = curr_press_dir
            if self.prev_press_dir is not None:
                pydirectinput.keyUp("left")
                pydirectinput.keyUp("right")
                pydirectinput.keyUp("up")
                pydirectinput.keyUp("down")
                time.sleep(0.1)
            else:
                time.sleep(0.3)

        if curr_press_dir:
            if shouldPress_lr:
                if x < self.coordinates[self.current][0]:
                    curr_press_dir = "right"
                if x > self.coordinates[self.current][0]:
                    curr_press_dir = "left"
                pydirectinput.keyUp(curr_press_dir)
                pydirectinput.press(curr_press_dir)
                time.sleep(0.2)
            if shouldPress_ud:
                if y < self.coordinates[self.current][1]:
                    curr_press_dir = "up"
                if y > self.coordinates[self.current][1]:
                    curr_press_dir = "down"
                pydirectinput.keyUp(curr_press_dir)
                pydirectinput.press(curr_press_dir)
                time.sleep(0.2)
            if not shouldPress_lr and not shouldPress_ud:
                pydirectinput.keyDown(curr_press_dir)

        self.curr_press_dir = curr_press_dir
        self.prev_x = x
        self.prev_y = y
        return

    def resolve_invalid_coordinate(self, x, y):
        bad_x = False
        bad_y = False
        if x > 5300 or x < -3251:
            print(f"Something went wrong with the OCR. Updated {x} to:{self.prev_x}")
            bad_x = True

        if y < -8372 or y > 9000:
            print(f"Something went wrong with the OCR. Updated {y} to:{self.prev_y}")
            bad_y = True
        return self.update_from_dir(x, y, bad_x, bad_y)

    def resolve_bad_parse(self, x, y):
        bad_x = False
        bad_y = False
        if abs(x - self.prev_x) > 20:
            print(f"Something went wrong with the OCR. Updated {x} to:{self.prev_x}")
            bad_x = True
        if abs(y - self.prev_y) > 20:
            print(f"Something went wrong with the OCR. Updated {y} to:{self.prev_y}")
            bad_y = True
        return self.update_from_dir(x, y, bad_x, bad_y)

    def update_from_dir(self, x, y, bad_x, bad_y):
        if bad_x:
            x = self.prev_x
            if self.curr_press_dir == "left":
                x = x - 1
            if self.curr_press_dir == "right":
                x = x + 1
        if bad_y:
            y = self.prev_y
            if self.curr_press_dir == "up":
                y = y + 1
            if self.curr_press_dir == "down":
                y = y - 1
        return x, y

    def write_failed_ocr_parse(self, img, x, y):
        if x > 5300 or x < -3251 and self.training_dir:
            cv2.imwrite(os.path.join(self.training_dir, "ocr_debug", f"{x}.png"), img)
        if y < -8372 or y > 9000 and self.training_dir:
            cv2.imwrite(os.path.join(self.training_dir, "ocr_debug", f"{y}.png"), img)

    def get_coordinates(self, ocr_bounds, pattern):
        with mss.mss() as sct:
            _img = cv2.cvtColor(np.array(sct.grab(ocr_bounds)), cv2.COLOR_BGR2GRAY)
            img = cv2.resize(_img, None, fx=7, fy=7)
            img = cv2.threshold(img, 135, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            # img = cv2.threshold(img, 135, 155, cv2.THRESH_BINARY)[1]
            img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV)[1]
            img = cv2.GaussianBlur(img, (9, 9), 0)

            custom_oem_psm_config = r'--psm 3 ' \
                                    r'-c tessedit_char_whitelist=' \
                                    r'ABCDEFGHIJKLMNOPQRSTUVWXYZ' \
                                    r'abcdefghijklmnopqrstuvwxyz' \
                                    r'0123456789' \
                                    r':,-'
            results = pytesseract.image_to_string(img, config=custom_oem_psm_config, output_type=Output.DICT)
            found_text = [i for i in results['text'].split("\n") if i != "" and "REGION" not in i]
            if len(found_text) < 2:
                return False
            replacements = {"- ": "-", " ": "", "Pi": "11", "O": "0", "G": "6", "/": "", "t": "1", "f": "1", "S": "8",
                            }
            coordinates = found_text[1]
            for key, value in replacements.items():
                coordinates = coordinates.replace(key, value)
            coordinates = coordinates.split(":")
            if len(coordinates) == 2:
                coordinates = coordinates[1]
            else:
                coordinates = coordinates[0]

            text = extract_match(pattern, coordinates)

            if text and len(text) > 1:
                x = int(float(text[-2]))
                y = int(float(text[-1]))
                # self.write_failed_ocr_parse(img, x, y)

                return x, y
            if len(text) == 1:
                # Handle the parsing case of ex: ["1000,343"]
                # TODO: Improve the pattern to prevent this
                text = text[0].split(",")
                if len(text) < 2:
                    text = text[0].split(".")
                if len(text) == 2:
                    if text[0] == "" or text[1] == "":
                        return
                    x = int(float(text[0]))
                    y = int(float(text[1]))
                    # self.write_failed_ocr_parse(_img, x, y)
                    return x, y

            return False
        return text
