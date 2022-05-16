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
        print(f"Current iteration:{idx} out of: {len(_combinations)}")
        clickThreeOres(points, kwargs.get('monitor_bounds'))

    template_match_click(os.path.join(os.path.dirname(__file__), 'images', 'stop_working_this_mine.png'),
                         {"top": 0, "left": 0, "width": 500, "height": 400})
    run(**kwargs)


def clickThreeOres(points, monitor_bounds):
    for idx, item in enumerate(points):
        point = [monitor_bounds['left'] + item[1], monitor_bounds['top'] + item[0]]
        print(f"Moving to:{point[0]},{point[1]}")
        pyautogui.moveTo(point[0], point[1])
        values = ('s', 1.5) if idx == 2 else ('a', 0.5)
        print(f"keypressed:{values[0]},timeWait:{values[1]}")
        pyautogui.press(values[0])
        time.sleep(values[1])

    template_match_click(os.path.join(os.path.dirname(__file__), 'images', 'ok.png'),
                         monitor_bounds, checkUntilGone=True)


def template_match_click(template_path, bounds, checkUntilGone=False):
    with mss.mss() as sct:
        tmp_img = np.array(sct.grab(bounds))

        ok = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)

        result = cv2.matchTemplate(tmp_img, ok, cv2.TM_CCOEFF_NORMED)
        mn, mx, mnLoc, mxLoc = cv2.minMaxLoc(result)
        threshold = .8
        while mx > threshold:
            MPx, MPy = mxLoc
            trows, tcols = ok.shape[:2]
            pyautogui.moveTo(bounds['left'] + ((MPx + MPx + tcols) / 2), bounds['top'] + ((MPy + MPy + trows) / 2))
            pyautogui.click()
            mx = -1 if not checkUntilGone else mx
            if checkUntilGone:
                tmp_img = np.array(sct.grab(bounds))
                result = cv2.matchTemplate(tmp_img, ok, cv2.TM_CCOEFF_NORMED)
                mn, _mx, mnLoc, mxLoc = cv2.minMaxLoc(result)
                mx = _mx
                time.sleep(1)


def run(*args, **kwargs):

    with mss.mss() as sct:
        count = 0
        fgbg = cv2.createBackgroundSubtractorMOG2()

        running = True

        while running:
            last_time = time.time()

            # Get raw pixels from the screen, save it to a Numpy array
            img = np.array(sct.grab(kwargs.get('monitor_bounds')))

            mask = fgbg.apply(img)
            color = ('b', 'g', 'r')

            tmp = cv2.bitwise_and(img, img, mask=mask)

            foreground_count = cv2.countNonZero(mask)
            if 300000 > foreground_count > 10000:
                if kwargs.get('debug'):
                    cv2.imwrite(f"{count}_foreground.png", cv2.cvtColor(tmp, cv2.COLOR_BGR2LAB))
                    cv2.imwrite(f"{count}_mask.png", mask)

                idx = (mask > 0)
                points = np.column_stack(np.nonzero(idx))
                weights = np.full((mask.shape[0], mask.shape[1]), 255)[idx].ravel().astype(float)

                db = DBSCAN(eps=int(kwargs.get('eps')),
                            min_samples=int(kwargs.get('min_samples')),
                            metric='euclidean',
                            algorithm='auto')
                labels = db.fit_predict(points, sample_weight=weights)

                n_clusters = int(np.max(labels)) + 1
                print(n_clusters)
                if n_clusters == int(kwargs.get('clusters')):
                    cluster_points = []
                    for l in range(n_clusters):
                        idx = (labels == l)
                        current = points[idx]

                        center = aabb.centre_point(aabb.create_from_points(current))
                        cluster_points.append(center)

                        if kwargs.get('debug'):
                            out = aabb.create_from_points(current)
                            cv2.putText(img=tmp,
                                        text=f"Cluster:{l}",
                                        org=(out[0, 1], out[0, 0] - 10),
                                        fontFace=cv2.FONT_HERSHEY_PLAIN,
                                        fontScale=0.9,
                                        color=(255, 0, 0),
                                        thickness=2)

                            cv2.rectangle(img=tmp,
                                          pt1=(out[0, 1], out[0, 0]),
                                          pt2=(out[1, 1], out[1, 0]),
                                          color=(255, 0, 0),
                                          thickness=2)
                            cv2.putText(img=img,
                                        text=f"Cluster:{l}",
                                        org=(out[0, 1], out[0, 0] - 10),
                                        fontFace=cv2.FONT_HERSHEY_PLAIN,
                                        fontScale=0.9,
                                        color=(255, 0, 0),
                                        thickness=2)

                            cv2.rectangle(img=img,
                                          pt1=(out[0, 1], out[0, 0]),
                                          pt2=(out[1, 1], out[1, 0]),
                                          color=(255, 0, 0),
                                          thickness=2)
                    if kwargs.get('debug'):
                        cv2.imwrite(f"{count}_clusters.jpg", tmp)

                    running = False
                    count = 0
                    run_bot(cluster_points, **kwargs)

            count += 1
            if count > 150:
                template_match_click(os.path.join(os.path.dirname(__file__), 'images', 'work_this_mine.png'),
                                     {"top": 0, "left": 0, "width": 500, "height": 400})
                time.sleep(4)
                template_match_click(os.path.join(os.path.dirname(__file__), 'images', 'ok.png'),
                                     kwargs.get('monitor_bounds'), checkUntilGone=True)
                time.sleep(15)

            # Display the pictuare
            if kwargs.get('debug'):
                print("fps: {}".format(1 / (time.time() - last_time)))
                cv2.imshow("OpenCV Foreground detection", mask)

            # Press "q" to quit
            if cv2.waitKey(25) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break


def update_global_clip_bounds(bounds_params, default_bounds):
        keys = list(default_bounds.keys())
        for idx in range(len(params)):
            default_bounds[keys[idx]] = int(params[idx])
        return default_bounds


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Arguments for ATITD Auto miner with opencv")

    parser.add_argument('--clusters',
                        dest='clusters',
                        help='number of ore nodes expected to be found from dbscan',
                        default=8
                        )

    parser.add_argument('--eps',
                        dest='eps',
                        help='DBScan parameter see '
                             'https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html',
                        default=15
                        )

    parser.add_argument('--min_samples',
                        dest='min_samples',
                        help='DBScan parameter see'
                             'https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html',
                        default=5000)

    parser.add_argument('--debug',
                        dest='debug',
                        action='store_true',
                        help='include --debug flag to have debug output')

    parser.add_argument('--bounds', nargs='+',
                        help='Usage:              --bounds -i 25, 50, 75, 100. '
                             'Updates: top, left, right, bottom in that order. '
                             'Missing params will result in default values. '
                             'Put nothing to keep the values at [200,500,950, 740]')

    args = vars(parser.parse_args())

    default_bounds = {"top": 200, "left": 500, "width": 950, "height": 740}

    args['monitor_bounds'] = default_bounds
    if args.get('bounds') is not None:
        params = args['bounds']
        args['monitor_bounds'] = update_global_clip_bounds(params, default_bounds)

    print(args)

    run(**args)
