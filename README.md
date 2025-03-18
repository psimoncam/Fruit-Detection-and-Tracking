# Fruit-Detection-and-Tracking

This repository provides the code and the trained model to implement object detection and tracking on apple trees videos using the YOLOv5 architecture and the deepSORT algorithm.

## Installation

The repository contains a Google Colab notebook which shows the procedure to run the whole system correctly. If you want to run the code locally, you should clone this repository and install all the dependencies:

```
pip install -r requirements.txt
```

With this you are already able to track your own video. If you have the ground truth annotations and want to obtain some metrics, you can use any kind of evaluating method. During my project, I used an [adaptation][my py-motmetrics] of the py-motmetrics [original repository][py-motmetrics]. The requirements.txt file already includes the installation of this adaptation.

[my py-motmetrics]: https://github.com/psimoncam/motmetrics_adaptation
[py-motmetrics]: https://github.com/cheind/py-motmetrics

## Example video tracked

https://github.com/user-attachments/assets/3d07626d-1873-4b3b-8927-d8a97dfdf126

