from typing import List
from .model import HumanDetectionModel

_model = None

def load_model(model_path: str):
    global _model
    if _model is None:
        _model = HumanDetectionModel(model_path)
    return _model

def detect_humans(image, model_path: str, conf: float = 0.5) -> List[dict]:
    """
    Returns list of detected humans with bounding boxes
    """
    model = load_model(model_path)
    results = model.predict(image, conf=conf)
    
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
    
    if len(detections) == 0:
        return []
    else:
        return detections