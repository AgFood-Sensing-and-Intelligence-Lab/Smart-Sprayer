import threading
import sys
import cv2
import torch
import json
import os
import time
from typing import Optional
from vimba import *
from time import time
import numpy as np

#model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')
model = torch.hub.load('/home/nvidia/DCW-main/YOLOv5', 'custom', path='bestS.pt', source='local')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def print_preamble():
    print('///////////////////////////////////////////////////////')
    print('/// Vimba API Asynchronous Grab with OpenCV Example ///')
    print('///////////////////////////////////////////////////////\n')


def print_usage():
    print('Usage:')
    print('    python asynchronous_grab_opencv.py [camera_id]')
    print('    python asynchronous_grab_opencv.py [/h] [-h]')
    print()
    print('Parameters:')
    print('    camera_id   ID of the camera to use (using first camera if not specified)')
    print()


def abort(reason: str, return_code: int = 1, usage: bool = False):
    print(reason + '\n')

    if usage:
        print_usage()

    sys.exit(return_code)


def parse_args() -> Optional[str]:
    args = sys.argv[1:]
    argc = len(args)

    for arg in args:
        if arg in ('/h', '-h'):
            print_usage()
            sys.exit(0)

    if argc > 1:
        abort(reason="Invalid number of arguments. Abort.", return_code=2, usage=True)

    return None if argc == 0 else args[0]


def get_camera(camera_id: Optional[str]) -> Camera:
    with Vimba.get_instance() as vimba:
        if camera_id:
            try:
                return vimba.get_camera_by_id(camera_id)

            except VimbaCameraError:
                abort('Failed to access Camera \'{}\'. Abort.'.format(camera_id))

        else:
            cams = vimba.get_all_cameras()
            if not cams:
                abort('No Cameras accessible. Abort.')

            return cams[0]


def setup_camera(cam: Camera):
    with cam:
        # Enable auto exposure time setting if camera supports it
        cam.ExposureAuto.set('Off')
        cam.ExposureMode.set('Timed')
        cam.ExposureTime.set('13996.768')

        # setting the color
        cam.BalanceWhiteAuto.set('Continuous')
        cam.Hue.set('0.00')
        cam.Saturation.set('1.00')
        cam.Gamma.set('0.70')

        # setting image resolution (affects acquisition frame rate)
        # range of height (8-2056), range of width (8-2464)
        #cam.Height.set('2056')
        #cam.Width.set('2464')

        # setting the binning reduce the res
        cam.BinningHorizontal.set('2')
        cam.BinningVertical.set('2')

        # setting pixel format
        cam.set_pixel_format(PixelFormat.Bgr8)

        # setting bandwidth of the data that will be streaming, range from (16250000-450000000)
        # (affects acquisition frame rate)
        cam.DeviceLinkThroughputLimit.set('450000000')

        cam.TriggerSelector.set('FrameStart')

        # optional in this example but good practice as it might be needed for hardware triggering
        cam.TriggerActivation.set('RisingEdge')

        # Make camera listen to Software trigger
        cam.TriggerSource.set('Software')
        cam.TriggerMode.set('Off')


def score_frame(frame):
    """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
    global model
    global device
    i = 0
    while(os.path.exists("js_data/data%s.json"%i)):
        i += 1
    model.to(device)
    frame = [frame]
    results = model(frame)
    #results.save()
    parsed = json.loads(results.pandas().xyxy[0].to_json(orient="records"))
    f = open('js_data/data%s.json'%i, 'w')
    json.dump(parsed, f)
    f.close()
    labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    return labels, cord


def class_to_label(x):
    """
    For a given label value, return corresponding string label.
    :param x: numeric label
    :return: corresponding string label
    """
    global model
    classes = model.names
    return classes[int(x)]


def plot_boxes(results, frame):
    """
    Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
    :param results: contains labels and coordinates predicted by model on the given frame.
    :param frame: Frame which has been scored.
    :return: Frame with bounding boxes and labels ploted on it.
    """

    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    for i in range(n):
        row = cord[i]
        if row[4] >= 0.3:
            x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(
                row[3] * y_shape)
            bgr = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
            cv2.putText(frame, class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
    return frame


class Handler:
    def __init__(self):
        self.shutdown_event = threading.Event()

    def __call__(self, cam: Camera, frame: Frame):
        ENTER_KEY_CODE = 13
        key = cv2.waitKey(1)
        if key == ENTER_KEY_CODE:
            self.shutdown_event.set()
            return

        elif frame.get_status() == FrameStatus.Complete:
            print('{} acquired {}'.format(cam, frame), flush=True)
            msg = 'Stream from \'{}\'. Press <Enter> to stop stream.'
            start_time = time()
            opencv_frame = frame.as_opencv_image()
            resize_frame = cv2.resize(opencv_frame, (640, 640))
            results = score_frame(resize_frame)
            final_frame = plot_boxes(results, resize_frame)
            end_time = time()
            fps = 1 / np.round(end_time - start_time, 2)

            cv2.putText(final_frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            cv2.imshow(msg.format(cam.get_name()), final_frame)
        cam.queue_frame(frame)
        
def main():
    print_preamble()
    cam_id = parse_args()

    with Vimba.get_instance():
        with get_camera(cam_id) as cam:

            # Start Streaming, wait for five seconds, stop streaming
            setup_camera(cam)
            handler = Handler()
            cam.start_streaming(handler=handler, buffer_count=10)
            handler.shutdown_event.wait()
            cam.TriggerSoftware.run()
            cam.stop_streaming()
	   


if __name__ == '__main__':
    main()
