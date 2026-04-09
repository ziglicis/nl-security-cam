import cv2
import base64
from PIL import Image
import io

class Camera:
    def __init__(self, source=0):
        self.cap = cv2.VideoCapture(source)

    def get_frame_b64(self) -> str:
        ret, frame = self.cap.read()
        if not ret:
            return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=75)
        return base64.b64encode(buf.getvalue()).decode()

    def release(self):
        self.cap.release()