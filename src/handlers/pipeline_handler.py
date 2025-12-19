from handlers.human_handler import process_human_detections

def run_pipeline(frame):
    detections = process_human_detections(frame)
    
    return {
        "detections": detections
    }