# AnimeFaceBoxes

Super fast anime face detection (> 200 fps for 512x512 resolution, RTX2080 + i7 9700k)

## Dependencies
- Python 3.6+ (Anaconda)
- PyTorch-1.0 +
- OpenCV3 (Python)

## Usage
- build nms: sh make.sh
- Manual data labeling: LabelFaceBox.py (you can skip this if you have danbooru2018 dataset)
- labeled data: faceboxes (danbooru2018 dataset)
- train: MyTrain.py
- eval: eval.py

![alt text](https://github.com/WynMew/AnimeFaceBoxes/blob/master/out.png)


codes borrowed a lot from [FaceBoxes.PyTorch](https://github.com/zisianw/FaceBoxes.PyTorch)
