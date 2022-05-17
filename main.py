import cv2
from itertools import combinations
import matplotlib.pyplot as plt
import mss
import numpy as np
import os
import pyautogui
from pyrr import aabb
from scipy import ndimage as ndi
from sklearn.cluster import DBSCAN
import time


def run_bot(cluster_points, *args, **kwargs):
    print(cluster_points)

    _combinations = list(combinations(cluster_points, 3))

    for idx, points in enumerate(_combinations):
        print(f"Current iteration:{idx + 1} out of: {len(_combinations)}")
        clickThreeOres(points, kwargs.get('monitor_bounds'))

    template_match_click(os.path.join(os.path.dirname(__file__), 'images', 'stop_working_this_mine.png'),
                         {"top": 0, "left": 0, "width": 500, "height": 400})
    time.sleep(2.5)
    run(**kwargs)


def clickThreeOres(points, monitor_bounds):
    for idx, item in enumerate(points):
        point = [monitor_bounds['left'] + item[1], monitor_bounds['top'] + item[0]]
        print(f"Moving to:{point[0]},{point[1]}")
        pyautogui.moveTo(point[0], point[1])
        values = ('s', 0.01) if idx == 2 else ('a', 0.76)
        print(f"keypressed:{values[0]},timeWait:{values[1]}")
        pyautogui.press(values[0])
        time.sleep(values[1])

    template_match_click(os.path.join(os.path.dirname(__file__), 'images', 'ok.png'),
                         monitor_bounds,
                         sleepUntilTrue=True,
                         checkUntilGone=True)


def template_match_click(template_path, bounds, sleepUntilTrue=False, checkUntilGone=False):
    clicked = False
    with mss.mss() as sct:
        retry_attempts = 35
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
            pyautogui.click()
            clicked = True
            mx = -1 if not checkUntilGone else mx
            if checkUntilGone:
                tmp_img = np.array(sct.grab(bounds))
                result = cv2.matchTemplate(tmp_img, ok, cv2.TM_CCOEFF_NORMED)
                mn, _mx, mnLoc, mxLoc = cv2.minMaxLoc(result)
                mx = _mx
                time.sleep(1)
    return clicked


def run(*args, **kwargs):
    with mss.mss() as sct:
        count = 0
        fgbg = cv2.createBackgroundSubtractorMOG2()

        running = True

        monitor = kwargs.get('monitor_bounds')

        template_match_click(os.path.join(os.path.dirname(__file__), 'images', 'stop_working_this_mine.png'),
                             {"top": 0, "left": 0, "width": 500, "height": 400})

        time.sleep(2.5)

        cluster_points = []
        previous_cluster_count = -1

        while running:

            last_time = time.time()

            # Get raw pixels from the screen, save it to a Numpy array
            img = np.array(sct.grab(monitor))

            mask = fgbg.apply(img)
            color = ('b', 'g', 'r')

            tmp = cv2.bitwise_and(img, img, mask=mask)

            foreground_count = cv2.countNonZero(mask)
            if 300000 > foreground_count > 10000 and count >= int(kwargs.get('frames')):
                print("Checking for clusters")
                downsample = mask.copy()
                for l in range(int(kwargs.get('downsample'))):
                    downsample = cv2.pyrDown(downsample)

                idx = (downsample > 0)
                points = np.column_stack(np.nonzero(idx))
                weights = np.full((downsample.shape[0], downsample.shape[1]), 255)[idx].ravel().astype(float)

                db = DBSCAN(eps=float(kwargs.get('eps')),
                            min_samples=int(kwargs.get('min_samples')),
                            metric='euclidean',
                            algorithm='auto')
                labels = db.fit_predict(points, sample_weight=weights)

                n_clusters = int(np.max(labels)) + 1
                print(n_clusters)
                previous_cluster_count = n_clusters
                if n_clusters != int(kwargs.get('clusters')):
                    count = 0

                if n_clusters == int(kwargs.get('clusters')):

                    for l in range(n_clusters):
                        idx = (labels == l)
                        current = points[idx]

                        center = aabb.centre_point(aabb.create_from_points(current))
                        center[0] = center[0] * (2 ** int(kwargs.get('downsample')))
                        center[1] = center[1] * (2 ** int(kwargs.get('downsample')))

                        cluster_points.append(center)

                        if kwargs.get('debug'):
                            out = aabb.create_from_points(current)
                            UPSCALE_OFFSET = (2 ** int(kwargs.get('downsample')))
                            cv2.putText(img=tmp,
                                        text=f"Cluster:{l}",
                                        org=(out[0, 1] * UPSCALE_OFFSET,
                                             out[0, 0] * UPSCALE_OFFSET - 10),
                                        fontFace=cv2.FONT_HERSHEY_PLAIN,
                                        fontScale=0.9,
                                        color=(255, 0, 0),
                                        thickness=2)

                            cv2.rectangle(img=tmp,
                                          pt1=(out[0, 1] * UPSCALE_OFFSET,
                                               out[0, 0] * UPSCALE_OFFSET),
                                          pt2=(out[1, 1] * UPSCALE_OFFSET,
                                               out[1, 0] * UPSCALE_OFFSET),
                                          color=(255, 0, 0),
                                          thickness=2)
                            cv2.putText(img=img,
                                        text=f"Cluster:{l}",
                                        org=(out[0, 1] * UPSCALE_OFFSET,
                                             out[0, 0] * UPSCALE_OFFSET - 10),
                                        fontFace=cv2.FONT_HERSHEY_PLAIN,
                                        fontScale=0.9,
                                        color=(255, 0, 0),
                                        thickness=2)

                            cv2.rectangle(img=img,
                                          pt1=(out[0, 1] * UPSCALE_OFFSET,
                                               out[0, 0] * UPSCALE_OFFSET),
                                          pt2=(out[1, 1] * UPSCALE_OFFSET,
                                               out[1, 0] * UPSCALE_OFFSET),
                                          color=(255, 0, 0),
                                          thickness=2)
                    count = 0
                    running = False

            count += 1
            if count > int(kwargs.get('frames')):
                template_match_click(os.path.join(os.path.dirname(__file__), 'images', 'work_this_mine.png'),
                                     {"top": 0, "left": 0, "width": 500, "height": 400})
                clicked = template_match_click(os.path.join(os.path.dirname(__file__), 'images', 'ok.png'),
                                               kwargs.get('monitor_bounds'),
                                               sleepUntilTrue=True,
                                               checkUntilGone=True)
                if clicked:
                    count = -int(kwargs.get('frames'))  # double the normal loading time

            # Display the pictuare
            if kwargs.get('debug'):
                cv2.putText(img=tmp,
                            text=f"FPS:{1 / (time.time() - last_time)}",
                            org=(5, 25),
                            fontFace=cv2.FONT_HERSHEY_PLAIN,
                            fontScale=1.2,
                            color=(255, 0, 0),
                            thickness=2)
                if previous_cluster_count > 0:
                    cv2.putText(img=tmp,
                                text=f"Previous Cluster Count:{previous_cluster_count}",
                                org=(5, 60),
                                fontFace=cv2.FONT_HERSHEY_PLAIN,
                                fontScale=1.2,
                                color=(255, 0, 0),
                                thickness=2)
                cv2.imshow("OpenCV Foreground detection", tmp)

            # Press "q" to quit
            if cv2.waitKey(25) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break
        run_bot(cluster_points, **kwargs)


def update_global_clip_bounds(bounds_params, default_bounds):
    keys = list(default_bounds.keys())
    for idx in range(len(bounds_params)):
        default_bounds[keys[idx]] = int(bounds_params[idx])
    return default_bounds


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Arguments for ATITD Auto miner with opencv")

    parser.add_argument('--clusters',
                        dest='clusters',
                        help='number of ore nodes expected to be found from dbscan',
                        default=8
                        )

    parser.add_argument('--wait_frames',
                        dest='frames',
                        help='Use this flag to change the amount of frames collected for foreground detection',
                        default=60)

    parser.add_argument('--downsample',
                        dest='downsample',
                        help='https://docs.opencv.org/3.4/d4/d1f/tutorial_pyramids.html.'
                             'Scaled by 2^n. Where --downsample of 3 is 1/8th the size of original',
                        default=3)

    parser.add_argument('--eps',
                        dest='eps',
                        help='DBScan parameter see '
                             'https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html',
                        default=15
                        )

    parser.add_argument('--min_samples',
                        dest='min_samples',
                        help='DBScan parameter see '
                             'https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html',
                        default=5000)

    parser.add_argument('--debug',
                        dest='debug',
                        action='store_true',
                        help='include --debug flag to have debug output')

    parser.add_argument('--bounds', nargs='+',
                        help='Usage:              --bounds  25, 50, 75, 100. '
                             'Updates: top, left, right, bottom in that order. '
                             'Missing params will result in default values. '
                             'Put nothing to keep the values at [200,500,950, 740]')

    args = vars(parser.parse_args())

    default_bounds = {"top": 200, "left": 500, "width": 950, "height": 740}

    args['monitor_bounds'] = default_bounds
    if args.get('bounds') is not None:
        params = args['bounds']
        args['monitor_bounds'] = update_global_clip_bounds(params, default_bounds)

    run(**args)
