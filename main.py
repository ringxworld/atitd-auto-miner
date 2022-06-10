import os
import re
import time
from itertools import combinations
import uuid

import cv2
import mss
import numpy as np
import pyautogui
import pytesseract
from pyrr import aabb
from sklearn.cluster import DBSCAN

from playsound import playsound
total = 0


def run_bot(cluster_points, center_colors, *args, **kwargs):
    print(cluster_points)
    cluster_points = [tuple(x) for x in cluster_points]

    _combinations = list(combinations(cluster_points, 3))

    successful_combinations = []

    for idx, points in enumerate(_combinations):
        print(f"Current iteration:{idx + 1} out of: {len(_combinations)}")
        if kwargs.get('run_ocr'):
            print(f"Total Ore collected: {total}")
        successful = clickOres(points, center_colors, kwargs.get('monitor_bounds'), kwargs.get('run_ocr'))
        if successful:
            successful_combinations.append(points)

    first_pass_count = len(_combinations)
    should_run_four = kwargs.get('four_combinations')
    if successful_combinations and should_run_four:
        four_combinations = list(combinations(cluster_points, 4))
        _combinations = list(set([item for sublist in
                                  [[item for item in four_combinations if set(i) == set(i).intersection(set(item))] for
                                   i in
                                   successful_combinations] for item in sublist]))
        print(f"Running an additional {len(_combinations)} to try 4 node harvests from successful ore harvests")
        for idx, points in enumerate(_combinations):
            print(f"Current iteration:{first_pass_count + idx + 1} out of: {len(_combinations) + first_pass_count}")
            if kwargs.get('run_ocr'):
                print(f"Total Ore collected: {total}")
            clickOres(points, center_colors, kwargs.get('monitor_bounds'), kwargs.get('run_ocr'))
        pass

    template_match_click(os.path.join(os.path.dirname(__file__), 'AtitdScripts/images', 'stop_working_this_mine.png'),
                         {"top": 0, "left": 0, "width": 500, "height": 400})
    time.sleep(2.5)
    run(**kwargs)


def clickOres(points, center_colors, monitor_bounds, run_ocr=False):
    time.sleep(0.5)
    if matched_pixel_colors(points, center_colors, monitor_bounds):
        for idx, item in enumerate(points):
            point = [monitor_bounds['left'] + item[1], monitor_bounds['top'] + item[0]]
            # print(f"Moving to:{point[0]},{point[1]}")
            pyautogui.moveTo(point[0], point[1])
            values = ('s', 0.01) if idx == len(points) - 1 else ('a', 0.76)
            # print(f"keypressed:{values[0]},timeWait:{values[1]}")
            pyautogui.press(values[0])
            time.sleep(values[1])

        clicked = template_match_click(os.path.join(os.path.dirname(__file__), 'AtitdScripts/images', 'ok.png'),
                                       monitor_bounds,
                                       sleepUntilTrue=True,
                                       checkUntilGone=True)
        if not run_ocr:
            return False
        if clicked:
            return False
        if not ocr_extract_text():
            return False

        return True

    return False


def extract_match(pattern, search_string):
    p = re.compile(pattern)
    text = p.findall(search_string)
    count = None

    if text:
        count = [int(s) for s in text[0].split() if s.isdigit()]

    if not count:
        return False

    if count:
        global total
        total += count[0]

    return True


def ocr_extract_text():
    with mss.mss() as sct:
        img = np.array(sct.grab({"top": 989, "left": 1500, "width": 400, "height": 17}))
        count = extract_match("workload had:(.*)", pytesseract.image_to_string(img))
        if count:
            return True
    return False


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


def almost_equal(current, previous):
    equal = True
    assert len(current) == len(previous)
    for idx, val in enumerate(current):
        if abs(previous[idx] - current[idx]) > 10:
            equal = False
    return equal


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
            time.sleep(0.05)
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

        template_match_click(os.path.join(os.path.dirname(__file__), 'AtitdScripts/images', 'stop_working_this_mine.png'),
                             {"top": 0, "left": 0, "width": 500, "height": 400})

        time.sleep(2.5)

        center_colors = []
        cluster_points = []
        previous_cluster_coordinates = []
        previous_cluster_count = -1
        my_frame = 0
        previous_triggered = False
        while running:

            last_time = time.time()

            # Get raw pixels from the screen, save it to a Numpy array
            img = np.array(sct.grab(monitor))

            mask = fgbg.apply(img)
            color = ('b', 'g', 'r')

            tmp = cv2.bitwise_and(img, img, mask=mask)
            foreground_count = cv2.countNonZero(mask)
            if 300000 > foreground_count > 10000:
                sound_effect = os.path.join(os.path.dirname(__file__), "AtitdScripts", "sounds", "alert.wav")
                print(os.path.exists(sound_effect))
                playsound(sound_effect)

            foreground_count = cv2.countNonZero(mask)
            if 300000 > foreground_count > 10000 and count >= int(kwargs.get('frames')):
                print("Checking for clusters")
                downsample = mask.copy()
                for l in range(int(kwargs.get('downsample'))):
                    downsample = cv2.pyrDown(downsample)

                idx = (downsample > 0)
                points = np.column_stack(np.nonzero(idx))
                # weights = np.full((downsample.shape[0], downsample.shape[1]), 255)[idx].ravel().astype(float)
                weights = downsample[idx].ravel().astype(float)

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
                    previous_cluster_coordinates = []
                    if kwargs.get('debug'):
                        for l in range(n_clusters):
                            idx = (labels == l)
                            current = points[idx]
                            out = aabb.create_from_points(current)
                            previous_cluster_coordinates.append(out)
                    path = os.path.join(os.path.dirname(__file__), "training_set",
                                        f'{uuid.uuid4()}_badmatch_downsampled.png')
                    print("Didn't match the expected number of clusters. Retrying")
                    if kwargs.get('debug'):
                        cv2.imwrite(path, downsample)
                    time.sleep(1)
                    template_match_click(
                        os.path.join(os.path.dirname(__file__), 'images', 'stop_working_this_mine.png'),
                        {"top": 0, "left": 0, "width": 500, "height": 400})

                if n_clusters == int(kwargs.get('clusters')):
                    print("Found correct number of clusters. Gathering center points to start Mining mini-game")
                    previous_cluster_coordinates = []
                    path = os.path.join(os.path.dirname(__file__), "training_set", f'{uuid.uuid4()}_goodmatch_downsampled.png')
                    cv2.imwrite(path, downsample)
                    for l in range(n_clusters):
                        idx = (labels == l)
                        current = points[idx]

                        center = aabb.centre_point(aabb.create_from_points(current))
                        center[0] = center[0] * (2 ** int(kwargs.get('downsample')))
                        center[1] = center[1] * (2 ** int(kwargs.get('downsample')))

                        cluster_points.append(center)
                        center_colors.append((tuple(center), tuple(img[int(center[0]), int(center[1]), :].tolist())))

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
                print("Attempting to work the mine")
                template_match_click(os.path.join(os.path.dirname(__file__), 'images', 'work_this_mine.png'),
                                               {"top": 0, "left": 0, "width": 500, "height": 400})
                clicked = template_match_click(os.path.join(os.path.dirname(__file__), 'images', 'ok.png'),
                                               kwargs.get('monitor_bounds'),
                                               sleepUntilTrue=True,
                                               checkUntilGone=True)
                if clicked:
                    print(f"Waiting for {int(kwargs.get('frames'))} frames")
                    count = - int(kwargs.get('frames'))  # double the normal loading time

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
                    print(len(previous_cluster_coordinates))
                    for l in range(len(previous_cluster_coordinates)):
                        UPSCALE_OFFSET = (2 ** int(kwargs.get('downsample')))
                        out = previous_cluster_coordinates[l]
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
                cv2.imshow("OpenCV Foreground detection", tmp)

            # Press "q" to quit
            if cv2.waitKey(25) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break
        # run_bot(cluster_points, center_colors, **kwargs)


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

    parser.add_argument('--run_ocr',
                        dest='run_ocr',
                        action='store_true',
                        help="Also include --four_combinations to do 4 stone breaks on successful harvests"
                             "In order to run OCR you need Tesseract installed. For instructions on how to do"
                             "do this on windows check here: https://github.com/UB-Mannheim/tesseract/wiki")

    parser.add_argument('--four_combinations',
                        dest='four_combinations',
                        action='store_true',
                        help="Use with --run_ocr to run 4 harvests on successful harvests")

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

    if args['run_ocr']:
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

    if not os.path.exists(os.path.join(os.path.dirname(__file__), "training_set")):
        os.mkdir(os.path.join(os.path.dirname(__file__), "training_set"))

    run(**args)
