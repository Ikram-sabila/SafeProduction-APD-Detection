from typing import List
from .model import HumanDetectionModel
from .tracker import HumanTracker

class HumanPredictor:
    def __init__(self, model_path):
        self.model = HumanDetectionModel(model_path)
        self.tracker = HumanTracker()

    def detect_humans(self, image, conf: float = 0.5) -> List[dict]:
        """
        Returns list of detected humans with bounding boxes
        """
        results = self.model.predict(image, conf=conf)
        
        detections = []
        
        for r in results:
            for box in r.boxes:
                score = float(box.conf[0])
                
                if score < conf:
                    continue
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = r.names[int(box.cls[0])]
                
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": score,
                    "label": label
                })
        
        return detections

    def track_humans(self, image, conf: float = 0.5):
        result = self.model.predict(image, conf=conf)[0]
        tracked = self.tracker.update(frame=image, result=result)
        return tracked