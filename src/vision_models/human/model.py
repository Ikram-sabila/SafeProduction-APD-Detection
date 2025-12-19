from ultralytics import YOLO

class HumanDetectionModel:
    def __init__(self, model_path: str, device: str = 'cpu'):
        self.model = YOLO(model_path)
        self.model.to(device)
        
    def predict(self, image, conf: float = 0.5):
        """
        Run human detection in image
        """
        return self.model(
            image,
            conf=conf,
            classes=[1],
            verbose=False
        )