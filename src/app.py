from streamlit_webrtc import webrtc_streamer
import av
from handlers.pipeline_handler import run_pipeline
import cv2

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    
    result = run_pipeline(img)
    print(result)
    
    for human in result.get("detections", []):
        x1, y1, x2, y2 = human["bbox"]
        label = human["label"]      
        conf = human["confidence"]   
        
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{label} ({conf:.2f})",
                    (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    (0, 255, 0), 
                    2)
        
    return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(key="example", video_frame_callback=video_frame_callback)