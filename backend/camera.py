import base64
import logging
import cv2

logger = logging.getLogger(__name__)

class Camera:
    def __init__(self, source=0):
        self.source = source
        self.cap = cv2.VideoCapture(source)
        self._fail_count = 0
        self._max_failures = 10

    def _reconnect(self):
        logger.warning("Camera lost, attempting reconnect (source=%s)", self.source)
        self.cap.release()
        self.cap = cv2.VideoCapture(self.source)
        if self.cap.isOpened():
            logger.info("Camera reconnected")
            self._fail_count = 0
        else:
            logger.error("Camera reconnect failed")

    def get_frame_b64(self) -> str | None:
        ret, frame = self.cap.read()
        if not ret:
            self._fail_count += 1
            if self._fail_count >= self._max_failures:
                self._reconnect()
                self._fail_count = 0
            return None
        self._fail_count = 0
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return base64.b64encode(buf.tobytes()).decode()

    def release(self):
        self.cap.release()
