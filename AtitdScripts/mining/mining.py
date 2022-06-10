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


class Mining(object):

    def __init__(self, **kwargs):
        self.tracker = Tracker(**kwargs)

        self.total_collected = 0

        # Debug info
        self.previous_cluster_count = 0
        self.previous_cluster_coordinates = []
        self._drawables = []

        self.kwargs = kwargs

        self.debug_mode = False
        if kwargs.get('debug'):
            self.debug_mode = True

        self.downsample = 0
        if kwargs.get('downsample'):
            self.downsample = int(kwargs.get('downsample'))

        self.clusters = -1
        if kwargs.get('clusters'):
            self.clusters = int(kwargs.get('clusters'))

        self.dbscan_eps = 0.0
        if kwargs.get('eps'):
            self.dbscan_eps = float(kwargs.get('eps'))

        self.dbscan_min_samples = 0
        if kwargs.get('min_samples'):
            self.dbscan_min_samples = int(kwargs.get('min_samples'))

    def run(self):
        # Always Reset mine on start
        template_match_click(
            os.path.join(os.path.dirname(__file__), 'AtitdScripts', 'images', 'stop_working_this_mine.png'),
            {"top": 0, "left": 0, "width": 500, "height": 400})
        time.sleep(2.5)

        # Start Foreground tracker
        img, mask = self.tracker.run()

        logging.info("Checking for clusters")
        downsample = mask.copy()
        for _l in range(self.downsample):
            downsample = cv2.pyrDown(downsample)

        idx = (downsample > 0)
        points = np.column_stack(np.nonzero(idx))
        weights = downsample[idx].ravel().astype(float)

        db = DBSCAN(eps=self.dbscan_eps,
                    min_samples=self.dbscan_min_samples,
                    metric='euclidean',
                    algorithm='auto')
        labels = db.fit_predict(points, sample_weight=weights)

        n_clusters = int(np.max(labels)) + 1
        logging.info(n_clusters)
        self.previous_cluster_count = n_clusters
        if n_clusters != self.clusters:
            if self.debug_mode:
                for _l in range(n_clusters):
                    idx = (labels == _l)
                    current = points[idx]
                    out = aabb.create_from_points(current)
                    self.previous_cluster_coordinates.append(out)
            logging.info("Didn't match the expected number of clusters. Retrying")

            time.sleep(1)
            template_match_click(
                os.path.join(os.path.dirname(__file__), 'images', 'stop_working_this_mine.png'),
                {"top": 0, "left": 0, "width": 500, "height": 400})
            self.tracker.curr_frame = 0

        if n_clusters == self.clusters:
            logging.info("Found correct number of clusters. Gathering center points to start Mining mini-game")
            self.previous_cluster_coordinates = []
            cluster_points = []
            center_colors = []

            for _l in range(n_clusters):
                idx = (labels == _l)
                current = points[idx]

                center = aabb.centre_point(aabb.create_from_points(current))
                center[0] = center[0] * (2 ** self.downsample)
                center[1] = center[1] * (2 ** self.downsample)

                cluster_points.append(center)
                center_colors.append((tuple(center), tuple(img[int(center[0]), int(center[1]), :].tolist())))

                if self.debug_mode:
                    self.add_debug_overlay(self, current, _l)

            self.tracker.running = False
            if self.debug_mode:
                self.tracker.set_drawables(self._drawables)
                self._drawables = []  # clear after sending
            total = OreHandler(self.total_collected, cluster_points, center_colors, **self.kwargs).play()
            self.total_collected += total
            template_match_click(
                os.path.join(os.path.dirname(__file__), 'AtitdScripts', 'images', 'stop_working_this_mine.png'),
                {"top": 0, "left": 0, "width": 500, "height": 400})
            time.sleep(2.5)

    def add_debug_overlay(self, current, idx):
        out = aabb.create_from_points(current)
        UPSCALE_OFFSET = (2 ** self.downsample)
        self._drawables.extend([{
            "type": "cv2_put_text",
            "text": f"Cluster:{idx}",
            "org": (out[0, 1] * UPSCALE_OFFSET,
                    out[0, 0] * UPSCALE_OFFSET - 10),
            "fontFace": cv2.FONT_HERSHEY_PLAIN,
            "fontScale": 0.9,
            "color": (255, 0, 0),
            "thickness": 2
        },
            {
                "type": "cv2_rectangle",
                "pt1": (out[0, 1] * UPSCALE_OFFSET,
                        out[0, 0] * UPSCALE_OFFSET),
                "pt2": (out[1, 1] * UPSCALE_OFFSET,
                        out[1, 0] * UPSCALE_OFFSET),
                "color": (255, 0, 0),
                "thickness": 2
            }])
