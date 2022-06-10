import os

import cv2
import logging
import numpy as np
import time

from AtitdScripts.image import template_match_click
from AtitdScripts.tracker import Tracker
from AtitdScripts.mining.ore_minigame import OreHandler
from pyrr import aabb
from sklearn.cluster import DBSCAN


class Reactory(object):

    def __init__(self, tracker, **kwargs):
        self.tracker = tracker


    def run(self):
        while self.running:
            self.run_handler()

    def run_handler(self):
        # Heat reactory then start the tracker
        # Wait until enough foreground pixels are found
        # Extract circles from foreground
        # dynamically look for optimal solution from starter circles
        # save click coordinates from the solution and click without updating the tracker
        pass
