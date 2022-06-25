# Object Tracking using DeepSORT - Computer Vision Lab

## Installation:

Optional: create virtualenv:

```
virtualenv --python=python3.6 ~/venv/deepsort
source ~/venv/deepsort/bin/activate
```

Install packages:
```
pip install numpy scipy Pillow
pip install torch torchvision
pip install opencv_python
pip install tensorflow_gpu
```

Tested with tensorflow==2.2.0 and torch==1.5.0, with CUDA 10.0

## Description

This module is a modification of the [nanonets deepsort repository](https://blog.nanonets.com/object-tracking-deepsort/), which is built on top of the [original deep sort module](https://github.com/nwojke/deep_sort). The original nwojke repo is built only for validating the algorithm with the MARS test dataset. The abhyantrika nanonets.com repo adds a custom class deepsort.py that acts as a bridge and also takes in any custom configurations like a different feature extractor and other parameters. It assumes that the detections are already available for the given video and does not provide support for object detection.

This repository provides a simplified version of abhyantrika repo, removing the training scripts and pre-computed detection files.I also modified slightly the code to remove dependencies and now it works with modern versions of scipy. A new test video has been added. The purpose of this repo is to be the basis for a lab where the students can add object detection capabilities to the algorithm, creating a tracker than can be used on any video.. 

Do not use this repository if you are not  in our Computer Vision course. The code in nanonets repo is more complete.

```deepsort.py``` is the nanonets bridge class that utilizes the original deep sort implementation, with their custom configs. We simply need to specify the encoder (feature extractor) we want to use and pass on the detection outputs to get the tracked bounding boxes. 
```test_on_video.py``` is their example code, that runs deepsort on a video whose detection bounding boxes are already given. 

# A simplified overview:
```sh
#Initialize deep sort object.
deepsort = deepsort_rbc(wt_path='ckpts/model640.pt') #path to the feature extractor model.

#Obtain all the detections for the given frame.
detections,out_scores = ... (Add here an object detector)

#Pass detections to the deepsort object and obtain the track information.
tracker,detections_class = deepsort.run_deep_sort(frame,out_scores,detections)

#Obtain info from the tracks.
for track in tracker.tracks:
    bbox = track.to_tlbr() #Get the corrected/predicted bounding box
    id_num = str(track.track_id) #Get the ID for the particular track.
    features = track.features #Get the feature vector corresponding to the detection.
```
The ```tracker``` object returned by deepsort contains all necessary info like the track_id, the predicted bounding boxes and the corresponding feature vector of the object. 

The pre-trained weights of the feature extractor are present in ```ckpts/``` folder.

With all packages installed correctly, you should be able to run the demo by adding an object detector to test_on_video.py

```
python test_on_video.py
```
