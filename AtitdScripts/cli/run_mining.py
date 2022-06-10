import os
import pytesseract
from AtitdScripts.mining.mining import Mining
import os

import pytesseract

from AtitdScripts.tracker import Tracker
from AtitdScripts.mining.mining import Mining


def run(*args, **kwargs):
    tracker = Tracker(**kwargs)
    mining = Mining(tracker, **kwargs)
    mining.run()


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
