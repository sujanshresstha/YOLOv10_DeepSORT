import os
import cv2
import torch
import datetime
import numpy as np
import logging
from collections import defaultdict
from absl import app, flags
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLOv10

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define command line flags
flags.DEFINE_string("video", "./data/test_1.mp4", "Path to input video or webcam index (0)")
flags.DEFINE_string("output", "./output/output.mp4", "Path to output video")
flags.DEFINE_float("conf", 0.50, "Confidence threshold")
flags.DEFINE_integer("blur_id", None, "Class ID to apply Gaussian Blur")
flags.DEFINE_integer("class_id", None, "Class ID to track")

FLAGS = flags.FLAGS

def initialize_video_capture(video_input):
    if video_input.isdigit():
        video_input = int(video_input)
        cap = cv2.VideoCapture(video_input)
    else:
        cap = cv2.VideoCapture(video_input)
    
    if not cap.isOpened():
        logger.error("Error: Unable to open video source.")
        raise ValueError("Unable to open video source")
    
    return cap

def initialize_model():
    model_path = "./weights/yolov10x.pt"
    if not os.path.exists(model_path):
        logger.error(f"Model weights not found at {model_path}")
        raise FileNotFoundError("Model weights file not found")
    
    model = YOLOv10(model_path)
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    model.to(device)
    logger.info(f"Using {device} as processing device")
    return model

def load_class_names():
    classes_path = "./configs/coco.names"
    if not os.path.exists(classes_path):
        logger.error(f"Class names file not found at {classes_path}")
        raise FileNotFoundError("Class names file not found")
    
    with open(classes_path, "r") as f:
        class_names = f.read().strip().split("\n")
    return class_names

def process_frame(frame, model, tracker, class_names, colors):
    results = model(frame, verbose=False)[0]
    detections = []
    for det in results.boxes:
        label, confidence, bbox = det.cls, det.conf, det.xyxy[0]
        x1, y1, x2, y2 = map(int, bbox)
        class_id = int(label)
        
        if FLAGS.class_id is None:
            if confidence < FLAGS.conf:
                continue
        else:
            if class_id != FLAGS.class_id or confidence < FLAGS.conf:
                continue
        
        detections.append([[x1, y1, x2 - x1, y2 - y1], confidence, class_id])
    
    tracks = tracker.update_tracks(detections, frame=frame)
    return tracks

def draw_tracks(frame, tracks, class_names, colors, class_counters, track_class_mapping):
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        class_id = track.get_det_class()
        x1, y1, x2, y2 = map(int, ltrb)
        color = colors[class_id]
        B, G, R = map(int, color)

        # Assign a new class-specific ID if the track_id is seen for the first time
        if track_id not in track_class_mapping:
            class_counters[class_id] += 1
            track_class_mapping[track_id] = class_counters[class_id]
        
        class_specific_id = track_class_mapping[track_id]
        text = f"{class_specific_id} - {class_names[class_id]}"
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 2)
        cv2.rectangle(frame, (x1 - 1, y1 - 20), (x1 + len(text) * 12, y1), (B, G, R), -1)
        cv2.putText(frame, text, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        if FLAGS.blur_id is not None and class_id == FLAGS.blur_id:
            if 0 <= x1 < x2 <= frame.shape[1] and 0 <= y1 < y2 <= frame.shape[0]:
                frame[y1:y2, x1:x2] = cv2.GaussianBlur(frame[y1:y2, x1:x2], (99, 99), 3)
    
    return frame

def main(_argv):
    try:
        cap = initialize_video_capture(FLAGS.video)
        model = initialize_model()
        class_names = load_class_names()
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(FLAGS.output, fourcc, fps, (frame_width, frame_height))
        
        tracker = DeepSort(max_age=20, n_init=3)
        
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(len(class_names), 3))
        
        class_counters = defaultdict(int)
        track_class_mapping = {}
        frame_count = 0
        
        while True:
            start = datetime.datetime.now()
            ret, frame = cap.read()
            if not ret:
                break
            
            tracks = process_frame(frame, model, tracker, class_names, colors)
            frame = draw_tracks(frame, tracks, class_names, colors, class_counters, track_class_mapping)
            
            end = datetime.datetime.now()
            logger.info(f"Time to process frame {frame_count}: {(end - start).total_seconds():.2f} seconds")
            frame_count += 1
            
            fps_text = f"FPS: {1 / (end - start).total_seconds():.2f}"
            cv2.putText(frame, fps_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)
            
            writer.write(frame)
            cv2.imshow("YOLOv10 Object tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        
        logger.info("Class counts:")
        for class_id, count in class_counters.items():
            logger.info(f"{class_names[class_id]}: {count}")
    
    except Exception as e:
        logger.exception("An error occurred during processing")
    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass
