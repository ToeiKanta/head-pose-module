# Headpose Detection
---
### Referenced Code
* https://www.learnopencv.com/head-pose-estimation-using-opencv-and-dlib
* https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python
* https://github.com/lincolnhard/head-pose-estimation

### Venv
* https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/26/python-virtual-env/

### Geting start

`pip install --upgrade firebase-admin`

### Main file 
* retinaface_video_headpose_recog.py
* headpose_module.py
* FaceRecognition_module.py
### Conda env
ref: https://stackoverflow.com/questions/41274007/anaconda-export-environment-file

`conda env export | grep -v "^prefix: " > environment.yml`

Either way, the other user then runs:

`conda env create -f environment.yml`

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
