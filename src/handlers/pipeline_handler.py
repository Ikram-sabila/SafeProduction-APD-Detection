from handlers.human_handler import HumanHandler

_human_handler = None

def run_pipeline(frame):
    global _human_handler
    if _human_handler is None:
        _human_handler = HumanHandler()
    
    # For detections
    # detections = _human_handler.process_human_detections(frame)
    # return {
    #     "detections": detections
    # }
    
    # For tracking
    annotated_frame  = _human_handler.process_human_tracking(frame)
    return annotated_frame