[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1LF_LTIt878PgqgffdQ2xP1aLcL5oU81_?usp=sharing)

# YOLOv10_DeepSORT
This repository contains code for object detection and tracking in videos using the YOLOv10 object detection model and the DeepSORT algorithm.

## Demo 
<p align="center">
  <img src="data/helper/test.gif" alt="demo" width="80%">
</p>

## Installation
1. Clone this repository:
  ```
   git clone https://github.com/sujanshresstha/YOLOv10_DeepSORT.git
   cd YOLOv10_DeepSORT
  ```

2. Create new environment
  - Using Conda
  ```
  conda env create -f conda.yml
  conda activate yolov10-deepsort
  ```
  - Using pip
  ```
  python3 -m virtualenv -p python3.11 yolov10-deepsort
  source yolov10-deepsort/bin/activate
  pip install -r requirements.txt
  ```

3. Download model weight
  ```
   mkdir -p weights
   wget -P weights -q https://github.com/jameslahm/yolov10/releases/download/v1.0/yolov10n.pt
   wget -P weights -q https://github.com/jameslahm/yolov10/releases/download/v1.0/yolov10s.pt
   wget -P weights -q https://github.com/jameslahm/yolov10/releases/download/v1.0/yolov10m.pt
   wget -P weights -q https://github.com/jameslahm/yolov10/releases/download/v1.0/yolov10b.pt
   wget -P weights -q https://github.com/jameslahm/yolov10/releases/download/v1.0/yolov10x.pt
   wget -P weights -q https://github.com/jameslahm/yolov10/releases/download/v1.0/yolov10l.pt
   ls -lh weights
  ```

## Usage
1. Prepare the video file:
   - Place the video file in the desired location.
   - Update the `video` flag in the path of the video file or set it to `0` to use the webcam as the input.
2. Download YOLOv10 model:
   - Make sure the corresponding model weights are available.(yolov10n/yolov10s/yolov10m/yolov10b/yolov10x/yolov10l)
3. Configure the output video:
   - Update `output` flag in the code to specify the path and filename of the output video file.
4. Set the confidence threshold:
   - Adjust the `conf` flag in the code to set the confidence threshold for object detection. Objects with confidence below this threshold will be filtered out.
5. If you want to detect and track certain object on video 
   - Modify the `class_id` flag in the code to specify the class ID for detection. The default value of the flag is set to None. If you wish to detect and track only persons, set it to 0, or refer to the coco.names file for other options.
6. If you want to blur certain object while tracking
   - Modify the `bulr_id` flag in the code to specify the class ID for detection. The default value of the flag is set to None. 

7. Run the code:
   ```
   # Run object tracking
   python object_tracking.py --video ./data/test.mp4 --output ./output/output.mp4

   # Run object tracking on webcam (set video flag to 0)
   python object_tracking.py --video 0 --output ./output/webcam.mp4

   # Run person tracking (set class_id flag to 0 for person)
   python object_tracking.py --video ./data/test.mp4 --output ./output/output.mp4 --class_id 0
   
   # Run tracking on a video with burring certain objects (set blur_id flag to 0 for person)
   python object_tracking.py --video ./data/test.mp4 --output ./output/output.mp4 --blur_id 0
   ```
   

## Acknowledgements
- This code is built upon the YOLOv10 model and the DeepSort algorithm.
- Credits to the authors and contributors of the respective repositories used in this project.

## References
- [YOLOv10: Real-Time End-to-End Object Detection](https://github.com/THU-MIG/yolov10)
- [Simple Online and Realtime Tracking with a Deep Association Metric](https://arxiv.org/abs/1703.07402)