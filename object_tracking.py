import cv2
import torch
import datetime
import numpy as np
from absl import app, flags
from absl.flags import FLAGS
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLOv10

# Define command line flags
flags.DEFINE_string("video", "./data/test.mp4", "Path to input video or webcam index (0)")
flags.DEFINE_string("output", "./output/output.mp4", "path to output video")
flags.DEFINE_float("conf", 0.50, "confidence threshold")
flags.DEFINE_integer("blur_id", None, "class ID to apply Gaussian Blur")
flags.DEFINE_integer("class_id", None, "class ID to track")


def main(_argv):
    # Start time to compute FPS
    start = datetime.datetime.now()

    # Initialize the video capture
    video_input = FLAGS.video

    # Check if the video input is an integer (webcam index)
    if FLAGS.video.isdigit():
        video_input = int(video_input)
        cap = cv2.VideoCapture(video_input)
    else:
        cap = cv2.VideoCapture(video_input)
    if not cap.isOpened():
        print("Error: Unable to open video source.")
        return
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # video writer objects
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(FLAGS.output, fourcc, fps, (frame_width, frame_height))

    # Initialize the DeepSort tracker
    tracker = DeepSort(max_age=20, n_init=3)

    # Load YOLO model
    model = YOLOv10("./weights/yolov10x.pt")

    # Choose cpu, gpu, mps
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model.to("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using {device} as processing device")

    # Load the COCO class labels
    classes_path = "./configs/coco.names"
    with open(classes_path, "r") as f:
        class_names = f.read().strip().split("\n")

    # Create a list of random colors to represent each class
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(class_names), 3))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Run model on each frame
        results = model(frame, verbose=False)[0]
        detect = []
        for det in results.boxes:
            label, confidence, bbox = det.cls, det.conf, det.xyxy[0]
            x1, y1, x2, y2 = map(int, bbox)
            class_id = int(label)

            # Filter out weak detections by confidence threshold and class_id
            if FLAGS.class_id is None:
                if confidence < FLAGS.conf:
                    continue

            else:
                if class_id != FLAGS.class_id or confidence < FLAGS.conf:
                    continue

            detect.append([[x1, y1, x2 - x1, y2 - y1], confidence, class_id])

        tracks = tracker.update_tracks(detect, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            class_id = track.get_det_class()
            x1, y1, x2, y2 = map(int, ltrb)
            color = colors[class_id]
            B, G, R = map(int, color)
            text = f"{track_id} - {class_names[class_id]}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 2)
            cv2.rectangle(frame, (x1 - 1, y1 - 20), (x1 + len(text) * 12, y1), (B, G, R), -1)
            cv2.putText(frame, text, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Apply Gaussian Blur
            if FLAGS.blur_id is not None and class_id == FLAGS.blur_id:
                if 0 <= x1 < x2 <= frame.shape[1] and 0 <= y1 < y2 <= frame.shape[0]:
                    frame[y1:y2, x1:x2] = cv2.GaussianBlur(frame[y1:y2, x1:x2], (99, 99), 3)

        cv2.imshow("YOLOv10 Object tracking", frame)

        # End time to compute the FPS
        end = datetime.datetime.now()

        # Show the time it took to process 1 frame
        print(f"Time to process 1 frame: {(end - start).total_seconds():.2f} seconds")

        # Calculate the frames per second and draw it on the frame
        fps = f"FPS: {1 / (end - start).total_seconds():.2f}"

        cv2.putText(frame, fps, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)

        writer.write(frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release video capture and writer
    cap.release()
    writer.release()


if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass
