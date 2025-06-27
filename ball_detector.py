import numpy as np
import cv2
from pathlib import Path
import logging
from typing import List, Dict, Optional

# Use YOLOv11 for detection
from yolov11 import YOLO

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BallDetector:
    def __init__(self, model_path: Optional[str] = None, confidence_threshold: float = 0.5):
        self.model = None
        self.confidence_threshold = confidence_threshold
        self.class_names = ['golf_ball']
        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str):
        self.model = YOLO(model_path)
        logging.info(f"YOLOv11 model loaded from {model_path}")

    def detect_balls(self, image: np.ndarray) -> List[Dict]:
        if self.model is None:
            raise ValueError("YOLOv11 model not loaded. Please load a model first.")
        # YOLOv11 expects RGB images
        if image.shape[2] == 3:
            img_rgb = image
        else:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.model.predict(img_rgb)
        detections = []
        for r in results:
            for box in r.boxes:
                conf = float(box.conf)
                if conf < self.confidence_threshold:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy)
                w = x2 - x1
                h = y2 - y1
                center = (x1 + w // 2, y1 + h // 2)
                detections.append({
                    'bbox': [x1, y1, w, h],
                    'confidence': conf,
                    'center': center
                })
        return detections

if __name__ == "__main__":
    detector = BallDetector()