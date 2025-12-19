import supervision as sv

class HumanTracker:
    def __init__(self):
        self.tracker = sv.ByteTrack()
        self.label_annator = sv.LabelAnnotator()
        self.box_annator = sv.BoxAnnotator()
        
    def update(self, result, frame):
        detections = sv.Detections.from_ultralytics(result)
        detections = self.tracker.update_with_detections(detections)
        
        if detections.tracker_id is None:
            return frame
        
        labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]
        
        annotated_frame = self.box_annator.annotate(
            scene=frame.copy(), detections=detections
        )
        annotated_frame = self.label_annator.annotate(
            scene=annotated_frame, detections=detections, labels=labels
        )
        return annotated_frame