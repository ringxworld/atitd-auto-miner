import logging
import queue
import time

import cv2
import mss
import numpy as np
import pydirectinput
from PIL import Image, ImageOps, ImageEnhance
from pytesseract import pytesseract

from AtitdScripts.utils import extract_match
from AtitdScripts.image import maintain_aspect_ratio_resize
from AtitdScripts.webwalker.WebTreeStructure import WebWalkerTree


class AutoWalker(object):

    def __init__(self, web, end_coordinate, **kwargs):
        self.running = True

        self.web = WebWalkerTree(node_definitions=web)

        self.ocr_bounds = {"top": 0, "left": 850, "width": 220, "height": 100}
        if kwargs.get('ocr_bounds'):
            self.ocr_bounds = kwargs.get('ocr_bounds')

        self.end_coordinate = end_coordinate
        if kwargs.get("end_coord"):
            x, y = kwargs.get("end_coord")
            self.end_coordinate = [int(x), int(y)]

        ocr_result = self.get_coordinates(self.ocr_bounds, r'-?\d+\.?\d*')
        if not ocr_result:
            return
        x, y = ocr_result

        self.coordinates = self.web.get_best_path_from_coordinates(start=[x, y], end=self.end_coordinate)
        self.current = 0

        self.curr_press_dir = None
        self.prev_press_dir = None

        pydirectinput.keyUp("left")
        pydirectinput.keyUp("right")
        pydirectinput.keyUp("up")
        pydirectinput.keyUp("down")

        print(f"Walking to: {self.coordinates[self.current]}")

    def run(self):
        while self.running:
            self.run_handler()

    def run_handler(self):
        while self.running:
            # map coordinates, come up with a better pattern later
            ocr_result = self.get_coordinates(self.ocr_bounds, r'-?\d+\.?\d*')

            if not ocr_result:
                return

            x, y = ocr_result

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

            shouldPress = False
            if abs(self.coordinates[self.current][0] - x) < 2 and abs(self.coordinates[self.current][1] -y) < 2:
                shouldPress = True

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
                if not shouldPress:
                    pydirectinput.keyDown(curr_press_dir)
                else:
                    pydirectinput.keyUp(curr_press_dir)
                    pydirectinput.press(curr_press_dir)

            self.curr_press_dir = curr_press_dir

    @staticmethod
    def get_coordinates(ocr_bounds, pattern):
        with mss.mss() as sct:
            img = cv2.cvtColor(np.array(sct.grab(ocr_bounds)), cv2.COLOR_BGR2GRAY)
            img = maintain_aspect_ratio_resize(img, width=int(img.shape[1] * 1.2))
            enhancer = ImageEnhance.Contrast(Image.fromarray(img))
            img = enhancer.enhance(100)
            img = ImageOps.expand(img, border=10, fill='white')

            custom_oem_psm_config = r'--psm 6 '

            found_text = [i for i in pytesseract.image_to_string(img, lang='eng', config=custom_oem_psm_config).split("\n")
                          if i != "" and "HOME REGION" not in i]
            if len(found_text) < 2:
                return False
            datetime = found_text[0]
            coordinates = found_text[1]
            text = extract_match(pattern, coordinates)

            if text and len(text) > 1:
                return int(float(text[-2])), int(float(text[-1]))
            if len(text) == 1:
                # Handle the parsing case of ex: ["1000,343"]
                # TODO: Improve the pattern to prevent this
                text = text[0].split(",")
                if len(text) == 2:
                    return int(text[0]), int(text[1])
            return False
        return text
