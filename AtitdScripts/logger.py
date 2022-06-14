import datetime
import logging
import os
import pathlib


def setup_logger(filepath=os.path.join(os.path.dirname(__file__), 'logs'), filename='generic_log', level=20):

    # Logger will not be set up twice.
    if logging.getLogger('').handlers:
        return

    # make directory if necessary
    if not os.path.exists(filepath):
        pathlib.Path(filepath).mkdir(parents=True)

    logging.basicConfig(level=level,
                        format='%(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        )

    # define a Handler which writers INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    # set a formatter which is simpler for console use
    formatter = logging.Formatter('%(message)s', "%Y-%m-%d %H:%M.%S")

    # tell the handler to use this format
    console.setFormatter(formatter)

    # add the logger to the root logger
    logging.getLogger('').addHandler(console)

