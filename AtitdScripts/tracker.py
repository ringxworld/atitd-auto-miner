import cv2
import numpy as np
import mss
import time


class Tracker(object):

    def __init__(self, *args, **kwargs):
        self.fgbg = cv2.createBackgroundSubtractorMOG2()
        self.running = True

        self.curr_frame = 0
        self._drawables = []

        self.debug_mode = False
        if kwargs.get('debug'):
            self.debug_mode = kwargs.get('debug')

        self.wait_frames = 100
        if int(kwargs.get('frames')):
            self.wait_frames = int(kwargs.get('frames'))

        self.monitor_bounds = {"top": 200, "left": 500, "width": 950, "height": 740}
        if kwargs.get('monitor_bounds'):
            self.monitor_bounds = kwargs.get('monitor_bounds')

        self.minForegroundPixels = 10000
        if kwargs.get('minForegroundCount'):
            self.minForegroundPixels = kwargs.get('minForegroundCount')

        self.maxForegroundPixels = 300000
        if kwargs.get('maxForegroundCount'):
            self.maxForegroundPixels = kwargs.get('maxForegroundCount')

    def run(self):
        with mss.mss() as sct:
            while self.running:
                last_time = time.time()
                img = np.array(sct.grab(monitor))  # Get raw pixels from the screen, save it to a Numpy array
                mask = self.fgbg.apply(img)
                tmp = cv2.bitwise_and(img, img, mask=mask)
                if self.debug_mode:
                    self.run_debug(tmp)

                if self.maxForegroundPixels > cv2.countNonZero(mask) > self.minForegroundPixels \
                        and self.wait_frames > self.curr_frame:
                    return img, mask

                # Press "q" to quit
                if cv2.waitKey(25) & 0xFF == ord("q"):
                    cv2.destroyAllWindows()
                    self.running = False
                self.curr_frame += 1

    def run_debug(self, img):
        cv2.imshow("OpenCV Foreground detection", img)
        for _drawable in self._drawables:
            if _drawable["cv2_put_text"]:
                cv2.putText(img=img, text=_drawable['text'], org=_drawable['org'], fontFace=_drawable['fontFace'],
                            fontScale=_drawable['fontScale'], color=_drawable['color'],
                            thickness=_drawable['thickness'])
            if _drawable["cv2_rectangle"]:
                cv2.rectangle(img=img, pt1=_drawable['pt1'], pt2=_drawable['pt2'], color=_drawable['color'],
                              thickness=_drawable['thickness'])

    def set_drawables(self, drawables):
        self._drawables = drawables
