import os

import mss
import numpy as np
import pyautogui
import time

import pytesseract

from AtitdScripts.image import matched_pixel_colors, template_match_click
from itertools import combinations

from AtitdScripts.utils import extract_match


class OreHandler(object):

    def __init__(self, total, cluster_points, center_colors, **kwargs):

        self.total = total

        self.cluster_points = [tuple(x) for x in cluster_points]
        self.center_colors = center_colors
        self._combinations = list(combinations(self.cluster_points, 3))  # Default combinations, groups of 3

        self.run_ocr = False
        if kwargs.get('run_ocr'):
            self.run_ocr = kwargs.get('run_ocr')

        self.monitor_bounds = {"top": 200, "left": 500, "width": 950, "height": 740}
        if kwargs.get('monitor_bounds'):
            self.monitor_bounds = kwargs.get('monitor_bounds')

        self.ocr_bounds = {"top": 989, "left": 1500, "width": 400, "height": 17}
        if kwargs.get('ocr_bounds'):
            self.ocr_bounds = kwargs.get('ocr_bounds')

        self.do_four_combinations = False
        if kwargs.get('four_combinations'):
            self.do_four_combinations = kwargs.get('four_combinations')

        self.successful_combinations = []  # Successful combinations from this current run

    def play(self):
        for idx, points in enumerate(self._combinations):
            print(f"Current iteration:{idx + 1} out of: {len(self._combinations)}")
            if self.run_ocr:
                print(f"Total Ore collected: {self.total}")
            successful = self.clickOres(points)
            if successful:
                self.successful_combinations.append(points)

        first_pass_count = len(self._combinations)
        should_run_four = self.do_four_combinations
        if self.successful_combinations and should_run_four:
            four_combinations = list(combinations(self.cluster_points, 4))
            self._combinations = list(set([item for sublist in
                                      [[item for item in four_combinations if set(i) == set(i).intersection(set(item))]
                                       for
                                       i in
                                       self.successful_combinations] for item in sublist]))
            print(f"Running an additional {len(self._combinations)} "
                  f"to try 4 node harvests from successful ore harvests")
            for idx, points in enumerate(self._combinations):
                print(f"Current iteration:{first_pass_count + idx + 1} out of:"
                      f" {len(self._combinations) + first_pass_count}")
                if self.run_ocr:
                    print(f"Total Ore collected: {self.total}")
                self.clickOres(points)

        return self.total

    def clickOres(self, points):
        time.sleep(0.5)
        if matched_pixel_colors(points, self.center_colors, self.monitor_bounds):
            for idx, item in enumerate(points):
                point = [self.monitor_bounds['left'] + item[1], self.monitor_bounds['top'] + item[0]]
                pyautogui.moveTo(point[0], point[1])
                values = ('s', 0.01) if idx == len(points) - 1 else ('a', 0.76)
                pyautogui.press(values[0])
                time.sleep(values[1])

            clicked = template_match_click(os.path.join(os.path.dirname(__file__), '..', 'cli', 'images',
                                                        'ok.png'),
                                           self.monitor_bounds,
                                           sleepUntilTrue=True,
                                           checkUntilGone=True)

            if clicked:
                return False
            if not self.run_ocr:
                return False
            success, count = self.ocr_extract_text(self.ocr_bounds, "workload had:(.*)")
            self.total += count
            if not success:
                return False
            return True
        return False

    @staticmethod
    def get_ore_ocr(ocr_bounds, pattern):
        res = (False, 0)
        with mss.mss() as sct:
            img = np.array(sct.grab(ocr_bounds))
            text = extract_match(pattern, pytesseract.image_to_string(img))
            count = None

            if text:
                count = [int(s) for s in text[0].split() if s.isdigit()]

            if count:
                res = (True, count[0])
        return res
