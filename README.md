# Headpose Detection
---
### Referenced Code
* https://www.learnopencv.com/head-pose-estimation-using-opencv-and-dlib
* https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python
* https://github.com/lincolnhard/head-pose-estimation

### Main file 
* retinaface_video_n_head_pose.py
* headpose_pure.py

### Requirements
* Python 3.7
  * dlib
  * opencv-python
  * numpy

* Please check `Dockerfile` for more information.

### Setup
* `./setup.sh`

### Usage
* Headpose detection for images
  * `python3.7 headpose.py -i [input_dir] -o [output_dir]`
* Headpose detection for videos
  * `python3.7 headpose_video.py -i [input_video] -o [output_file]`
* Headpose detection for webcam
  * `python3.7 headpose_video.py`

### Demo
[<img src="./img/back1.png">](https://photos.app.goo.gl/tA3Qd22tM2DzQNCD9)
