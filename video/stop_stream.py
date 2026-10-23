import functools
import queue
import threading
import time
import torch

from vimba import *
import cv2
import vimba

def  frame_handler(cam , frame ):
    opencv_frame=frame.as_opencv_image()
    cv2.imshow("sd",opencv_frame)
    cam.queue_frame(frame)

with Vimba.get_instance () as vimba:
    cams = vimba.get_all_cameras ()
    with cams [0] as cam:
        q=cam.start_streaming(frame_handler)
        print(q)
        time.sleep (5)
        cam.stop_streaming ()