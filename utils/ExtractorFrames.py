import cv2
import os

capture = cv2.VideoCapture('/home/usuaris/imatge/pol.simon/apple_detection/videos/210906_143531_z_r2_w_015_225_162/ds0/video/210906_143531_z_r2_w_015_225_162_left.mp4')
cont = 1
path = '/home/usuaris/imatge/pol.simon/apple_detection/videos/210906_143531_z_r2_w_015_225_162/ds0/frames/'

if not os.path.exists(path):
    os.makedirs(path)

while (capture.isOpened()):
    ret, frame = capture.read()
    if cont > 140:
        break
    if (ret == True):
        #frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imwrite(path + 'frame_%04d.jpg' % cont, frame)
        cont += 1
        if (cv2.waitKey(1) == ord('s')):
            break
    else:
        break

capture.release()
cv2.destroyAllWindows()