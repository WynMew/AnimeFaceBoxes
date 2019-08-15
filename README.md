# AnimeFaceBoxes

Super fast anime face detection (> 200 fps for 512x512 resolution, RTX2080)

## Dependencies
- Python 3.6+ (Anaconda)
- PyTorch-1.0 +
- scipy, numpy, sklearn etc.
- OpenCV3 (Python)

## Usage
- build nms: sh make.sh
- Manual data labeling: LabelFaceBox.py (you can skip this if you have danbooru2018 dataset)
- labeled data: faceboxes (danbooru2018 dataset)
- train: MyTrain.py
- eval: eval.py

- ![alt text](https://github.com/WynMew/AnimeFaceBoxes/blob/master/out.png)
