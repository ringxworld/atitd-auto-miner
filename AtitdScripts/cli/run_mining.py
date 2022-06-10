import os
import pytesseract
import time


def run(*args, **kwargs):
    tracker = Tracker()
    tracker.run(startup, )
    pass


def startup():
    template_match_click(os.path.join(os.path.dirname(__file__), 'AtitdScripts","images', 'stop_working_this_mine.png'),
                         {"top": 0, "left": 0, "width": 500, "height": 400})

    time.sleep(2.5)


def runnable(mask, downsample):
    print("Checking for clusters")
    downsample = mask.copy()
    for l in range(int(downsample)):
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
