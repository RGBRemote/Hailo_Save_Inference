from pathlib import Path
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import numpy as np
import cv2
import hailo
import time
import datetime
import json

from hailo_apps.hailo_app_python.core.common.buffer_utils import get_caps_from_pad, get_numpy_from_buffer
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import app_callback_class
from hailo_apps.hailo_app_python.apps.detection.detection_pipeline import GStreamerDetectionApp

# -----------------------------------------------------------------------------------------------
# User-defined class to be used in the callback function
# -----------------------------------------------------------------------------------------------
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        self.new_variable = 42
        # Frame saving variables
        self.last_save_time = time.time()
        self.save_interval = 3  # Save every 3 seconds
        self.save_dir = "saved_frames"
        self.frame_counter = 0
        
        # Create save directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)

    def new_function(self):
        return "The meaning of life is: "

    def should_save_frame(self, current_time):
        """Check if it's time to save a frame based on the interval"""
        return current_time - self.last_save_time >= self.save_interval

    def save_frame_with_detections(self, frame, detections_data):
        """Save frame with detections to disk"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_base = f"frame_{timestamp}_{self.frame_counter:04d}"
        
        # Draw bounding boxes on the frame before saving
        frame_with_boxes = self.draw_detections(frame.copy(), detections_data)
        
        # Save image with bounding boxes
        image_path = os.path.join(self.save_dir, f"{filename_base}.jpg")
        cv2.imwrite(image_path, frame_with_boxes)
        
        # Save detection data as JSON
        json_path = os.path.join(self.save_dir, f"{filename_base}.json")
        with open(json_path, 'w') as f:
            json.dump(detections_data, f, indent=2)
        
        print(f"Saved frame and detections: {filename_base}")
        
        self.last_save_time = time.time()
        self.frame_counter += 1

    def draw_detections(self, frame, detections_data):
        """Draw bounding boxes and labels on the frame"""
        for detection in detections_data['detections']:
            bbox = detection['bbox']
            label = detection['label']
            confidence = detection['confidence']
            track_id = detection.get('track_id', 0)
            
            # Convert normalized coordinates to pixel coordinates
            height, width = frame.shape[:2]
            xmin = int(bbox['xmin'] * width)
            ymin = int(bbox['ymin'] * height)
            xmax = int(bbox['xmax'] * width)
            ymax = int(bbox['ymax'] * height)
            
            # Draw bounding box
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            
            # Draw label background
            label_text = f"{label} {confidence:.2f} ID:{track_id}"
            label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame, (xmin, ymin - label_size[1] - 10), 
                         (xmin + label_size[0], ymin), (0, 255, 0), -1)
            
            # Draw label text
            cv2.putText(frame, label_text, (xmin, ymin - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Add timestamp and frame info
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, f"Time: {timestamp}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Frame: {self.frame_counter}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame

# -----------------------------------------------------------------------------------------------
# User-defined callback function
# -----------------------------------------------------------------------------------------------
def app_callback(pad, info, user_data):
    # Get the GstBuffer from the probe info
    buffer = info.get_buffer()
    # Check if the buffer is valid
    if buffer is None:
        return Gst.PadProbeReturn.OK

    # Using the user_data to count the number of frames
    user_data.increment()
    string_to_print = f"Frame count: {user_data.get_count()}\n"

    # Get the caps from the pad
    format, width, height = get_caps_from_pad(pad)

    # If the user_data.use_frame is set to True, we can get the video frame from the buffer
    frame = None
    if user_data.use_frame and format is not None and width is not None and height is not None:
        # Get video frame
        frame = get_numpy_from_buffer(buffer, format, width, height)

    # Get the detections from the buffer
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    # Parse the detections and prepare data for saving
    detection_count = 0
    detections_data = {
        'timestamp': time.time(),
        'datetime': datetime.datetime.now().isoformat(),
        'frame_count': user_data.get_count(),
        'detections': []
    }
    
    for detection in detections:
        label = detection.get_label()
        bbox = detection.get_bbox()
        confidence = detection.get_confidence()
        
        detection_info = {
            'label': label,
            'confidence': float(confidence),
            'bbox': {
                'xmin': float(bbox.xmin()),
                'ymin': float(bbox.ymin()),
                'xmax': float(bbox.xmax()),
                'ymax': float(bbox.ymax())
            }
        }
        
        if label == "person":
            # Get track ID
            track_id = 0
            track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
            if len(track) == 1:
                track_id = track[0].get_id()
                detection_info['track_id'] = track_id
                
            string_to_print += (f"Detection: ID: {track_id} Label: {label} Confidence: {confidence:.2f}\n")
            detection_count += 1
        
        detections_data['detections'].append(detection_info)

    # Save frame with detections every 3 seconds if we have detections
    if user_data.use_frame and frame is not None and detections_data['detections']:
        current_time = time.time()
        if user_data.should_save_frame(current_time):
            user_data.save_frame_with_detections(frame, detections_data)

    if user_data.use_frame and frame is not None:
        # Note: using imshow will not work here, as the callback function is not running in the main thread
        # Let's print the detection count to the frame
        cv2.putText(frame, f"Detections: {detection_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # Example of how to use the new_variable and new_function from the user_data
        # Let's print the new_variable and the result of the new_function to the frame
        cv2.putText(frame, f"{user_data.new_function()} {user_data.new_variable}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # Convert the frame to BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        user_data.set_frame(frame)

    print(string_to_print)
    return Gst.PadProbeReturn.OK

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    env_file     = project_root / ".env"
    env_path_str = str(env_file)
    os.environ["HAILO_ENV_FILE"] = env_path_str
    # Create an instance of the user app callback class
    user_data = user_app_callback_class()
    app = GStreamerDetectionApp(app_callback, user_data)
    app.run()
