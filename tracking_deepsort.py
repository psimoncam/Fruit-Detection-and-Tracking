from email.policy import default
import numpy as np
import cv2
import sys
from yolov5.detect_modified import yolo_detector
from deepsort_lab.deepsort import *

from base64 import b64encode
import os
import glob
import warnings
import torch
import random
import argparse

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='./yolov5/weights/best_train7.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default='./tmp/', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default='./yolov5/data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--video', type=str, default=None, help='path to the video file')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[1920], help='inference size h,w')
    parser.add_argument('--maxage', type=int, default=100, help='maximum number of missed misses before a track is deleted')
    parser.add_argument('--conf-thres', type=float, default=0.7, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--half', default=False, action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', default=False, action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--lastframe', type=int, default=None, help='last annotated frame of the video')
    parser.add_argument('--firstframe', type=int, default=0, help='first annotated frame of the video')
    parser.add_argument('--lambda_deepsort', type=float, default=0, help='hyperparameter to control the combined association cost')
    parser.add_argument('--max_iou_distance', type=float, default=0.7, help='max iou distance to perform tracking')
    parser.add_argument('--n_init', type=int, default=3, help='Number of frames that a track remains in initialization phase')
    parser.add_argument('--enlarge', default=False, action='store_true', help='enlarge bounding boxes or not')
    parser.add_argument('--save', default=False, action='store_true', help='save results for future metrics evaluation')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    #print_args(vars(opt))
    return opt

def enlarge_bboxs(bbox_list):
    to_return = []
    for bbox in bbox_list:
        bbox_ampliada = [0] * 4
        w, h = bbox[2], bbox[3]
        bbox_ampliada[0] = bbox[0] - (int(w // 4)) if int(bbox[0] - (int(w // 4))) > 0 else 0
        bbox_ampliada[1] = bbox[1] - (int(h // 4)) if int(bbox[1] - (int(h // 4))) > 0 else 0
        new_w = w + int(2 * int(w // 4))
        new_h = h + int(2 * int(h // 4))
        bbox_ampliada[2] = new_w if int(bbox_ampliada[0] + new_w) < 1080 else int(1080 - bbox_ampliada[0])
        bbox_ampliada[3] = new_h if int(bbox_ampliada[1] + new_h) < 1920 else int(1920 - bbox_ampliada[1])
        to_return.append(bbox_ampliada)
    return to_return

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    opt = parse_opt()
    video_file = opt.video
    #temporal_folder = "./tmp"
    #if not os.path.exists(temporal_folder):
    #    os.makedirs(temporal_folder)

    # Check a video path is passed
    if video_file == None:
        print ('A path to a video is required!!')
        sys.exit

    # Read Video sequence
    if not os.path.isfile(video_file):
        print ('Can not find {}'.format(video_file))
        sys.exit

    # cap = cv2.VideoCapture('videos/race.mp4')
    cap = cv2.VideoCapture(video_file)
    # Exit if video not opened.
    if not cap.isOpened():
        print('Could not open {} video'.format(video_file))
        sys.exit()

    # Initialize writer
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    writer = cv2.VideoWriter(os.path.split(video_file)[-1].replace(".mp4", "_tracked.mp4"), fourcc, 20, (1080, 1920), True)

    frames_to_process = []

    nb_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)-1)
    if opt.lastframe is not None:
        nb_frame = opt.lastframe
    #for c_frame in range(0, 139):
    for c_frame in range(opt.firstframe, nb_frame + 1):
        frames_to_process.append(c_frame)

    last_frame_to_process = max(frames_to_process)

    frame_ID = 1

    #Initialize deep sort. 
    deepsort = deepsort_rbc(max_age=opt.maxage, _lambda=opt.lambda_deepsort)
    detections = None
    colors = {}
    pomes_totals = 0
    bottom_right = 200

    detector = yolo_detector(weights=opt.weights, imgsz=opt.imgsz, data=opt.data, conf_thres=opt.conf_thres, iou_thres=opt.iou_thres, half=opt.half, dnn=opt.dnn, max_det=opt.max_det)
    
    lines = []

    while (frame_ID < last_frame_to_process):
        print('Processing frame {}'.format(frame_ID))
        ret,frame = cap.read()
        
        # frame = frame.astype(np.uint8)

        if "_a_" in os.path.split(video_file)[-1]:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        
        elif "_k_" in os.path.split(video_file)[-1]:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if ret and frame_ID in frames_to_process:

            detections, out_scores = detector.run(frame)

            if len(detections) == 0:
                print("No dets")
                frame_ID+=1
                continue
            
            detections = np.array(detections)
            out_scores = np.array(out_scores)
            
            if opt.enlarge:
                detections = enlarge_bboxs(detections)
            
            tracker,detections_class = deepsort.run_deep_sort(frame,out_scores,detections)
            try:
                aux = tracker.get_total_tracks()
            except:
                aux = pomes_totals
            pomes_totals = aux
            #pomes_totals = tracker.total_tracks
            
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue

                bbox = track.to_tlbr() #Get the corrected/predicted bounding box                                                   
                id_num = str(track.track_id) #Get the ID for the particular track.                                                 
                features = track.features #Get the feature vector corresponding to the detection.
                if id_num not in colors.keys():
                    colors[id_num] = (random.randint(0,255),random.randint(0,255),random.randint(0,255))                                  

                # Overlay bbox from tracker.                                                                                           
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), colors[id_num], 3)
                cv2.putText(frame, str(id_num), (int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0, 130, 255), 3)
                
                w = int(bbox[2]) - int(bbox[0])
                h = int(bbox[3]) - int(bbox[1])
                
                lines.append(str(frame_ID) + "," + str(id_num) + "," + str(int(bbox[0])) + "," + str(int(bbox[1])) + "," + str(w) + "," + str(h) + "," + str(track.confidence) + "," + str(-1) + "," + str(-1) + "," + str(-1))
                
            bottom_right = 200
            if pomes_totals < 100:
                bottom_right = 200
            elif pomes_totals < 1000:
                bottom_right = 220
            else:
                bottom_right = 240
                
            cv2.rectangle(frame, (5, 1920 - 50, bottom_right, 1920-5), (255,255,255), -1)
            text = "Pomes: {}".format(pomes_totals)
            cv2.putText(frame, text, (10, 1910), 0, 5e-3 * 200, (0, 0, 0), 2)
    
            print("Pomes fins el frame {}: {}\n".format(frame_ID, pomes_totals))
            writer.write(frame)
            
        frame_ID += 1

    # When everything done, release the capture
    writer.release()
    
    if opt.save:
        dst = r"./results"
        if not os.path.exists(dst):
            os.makedirs(dst)
        output_file = os.path.join(dst, os.path.split(video_file)[-1].replace(".mp4", ".txt"))
        with open(output_file, 'w') as test_file:
            test_file.write("\n".join(lines))
    
    print("\n---------FINISHED---------")
    print("Pomes totals:", pomes_totals)