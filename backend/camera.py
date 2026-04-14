import base64
import logging
import numpy as np
import cv2

logger = logging.getLogger(__name__)

MOTION_THRESHOLD = 25
MOTION_MIN_PERCENT = 0.5

class Camera:
    def __init__(self, source=0):
        self.source = source
        self.cap = cv2.VideoCapture(source)
        self._fail_count = 0
        self._max_failures = 10
        self._prev_gray = None

    def _reconnect(self):
        logger.warning("Camera lost, attempting reconnect (source=%s)", self.source)
        self.cap.release()
        self.cap = cv2.VideoCapture(self.source)
        if self.cap.isOpened():
            logger.info("Camera reconnected")
            self._fail_count = 0
        else:
            logger.error("Camera reconnect failed")

    def get_frame(self) -> tuple[str, np.ndarray] | tuple[None, None]:
        ret, frame = self.cap.read()
        if not ret:
            self._fail_count += 1
            if self._fail_count >= self._max_failures:
                self._reconnect()
                self._fail_count = 0
            return None, None
        self._fail_count = 0
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return base64.b64encode(buf.tobytes()).decode(), frame

    def has_motion(self, frame: np.ndarray) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        if self._prev_gray is None:
            self._prev_gray = gray
            return True
        diff = cv2.absdiff(self._prev_gray, gray)
        self._prev_gray = gray
        _, thresh = cv2.threshold(diff, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)
        changed = (thresh > 0).sum() / thresh.size * 100
        return changed >= MOTION_MIN_PERCENT

    def release(self):
        self.cap.release()
