import os
import glob
import json
import shutil
import cv2
import numpy as np

width = 1080
height = 1920

dicc_obj = {}
poma = 0
with open(r"/home/usuaris/imatge/pol.simon/apple_detection/videos/210928_165030_k_r2_w_015_125_162/ds0/ann/210928_165030_k_r2_w_015_125_162.mp4.json", 'r') as json_file:
 data = json.load(json_file)
for frame in data["frames"]:
 figures_list = []
 index = int(frame["index"]) + 1
 jpg = glob.glob(os.path.join(r"/home/usuaris/imatge/pol.simon/apple_detection/videos/210928_165030_k_r2_w_015_125_162/ds0/frames/", 'frame' + '*0' + str(index) + '.jpg'))[0]
 img = cv2.imread(jpg)
 for figura in frame["figures"]:
   objectKey = figura['objectKey']
   if objectKey not in dicc_obj:
     dicc_obj[objectKey] = poma
     poma += 1
     
   # Videos Camera Z (orientacio correcta)
   #xmin = figura["geometry"]["points"]["exterior"][0][0]
   #ymin = figura["geometry"]["points"]["exterior"][0][1]
   #xmax = figura["geometry"]["points"]["exterior"][1][0]
   #ymax = figura["geometry"]["points"]["exterior"][1][1]
   
   # Videos Camera K (rotats a la dreta)
   #img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
   #cv2.imwrite("rotated_frame%04d.jpg" % index, img)
   xmin = figura["geometry"]["points"]["exterior"][0][1]
   ymin = height - figura["geometry"]["points"]["exterior"][1][0]
   xmax = figura["geometry"]["points"]["exterior"][1][1]
   ymax = height - figura["geometry"]["points"]["exterior"][0][0]
   cropped_image = img[int(ymin):int(ymax), int(xmin):int(xmax)]
   # cv2_imshow(cropped_image)
   cv2.imwrite(os.path.join(r"/home/usuaris/imatge/pol.simon/apple_detection/videos/210928_165030_k_r2_w_015_125_162/ds0/cropped_objects_vid6/", "video_6_frame_%04d_objecte_%04d.jpg" % (index, dicc_obj[objectKey])), cropped_image)
print(poma)