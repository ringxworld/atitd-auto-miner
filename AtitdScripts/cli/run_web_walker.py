import logging
import os

import pytesseract

from AtitdScripts.webwalker.auto_walker import AutoWalker


def run(*args, **kwargs):
    auto_walker = AutoWalker(os.path.join(os.path.dirname(__file__), "..", "test", "data", "test_web.yml"),
                             None,
                             **kwargs)

    auto_walker.run()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="")

    parser.add_argument('--run_ocr',
                        dest='run_ocr',
                        action='store_true',
                        help="Also include --four_combinations to do 4 stone breaks on successful harvests"
                             "In order to run OCR you need Tesseract installed. For instructions on how to do"
                             "do this on windows check here: https://github.com/UB-Mannheim/tesseract/wiki")

    parser.add_argument('--end_coord', nargs='+',
                        help='Usage:              --end_coord  1000 10000 '
                             'Type the coordinate to path to using the web walker')

    args = vars(parser.parse_args())

    if args['end_coord']:
        try:
            assert len(args['end_coord']) == 2
        except Exception:
            logging.critical("Invalid argument was entered for argument --end_coord. Enter --help for usage guide")

    if args['run_ocr']:
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

    run(**args)


if __name__ == '__main__':
    main()
