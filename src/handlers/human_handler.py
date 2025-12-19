from vision_models.human.predictor import detect_humans

MODEL_PATH = "models/human.pt"

def process_human_detections(frame):
    """
    Process 1 video frame
    """
    return detect_humans(
        image=frame,
        model_path=MODEL_PATH,
        conf=0.5
    )