import os

import pytesseract

from AtitdScripts.webwalker.auto_walker import AutoWalker


def run(*args, **kwargs):
    auto_walker = AutoWalker(os.path.join(os.path.dirname(__file__), "..", "test", "data", "test_web.yml"),
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

    args = vars(parser.parse_args())

    if args['run_ocr']:
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

    run(**args)


if __name__ == '__main__':
    main()
