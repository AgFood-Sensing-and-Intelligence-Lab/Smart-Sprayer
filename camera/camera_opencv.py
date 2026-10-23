# First import the library
from camera_abstract import CameraBase
import cv2
import config as cfg
from metric.util import get_ann_info_from_xml
from datetime import datetime, timedelta
import os
from time import time, sleep
import numpy as np


use_video = True
use_img = not use_video
if use_img:
    with_ann = True

class CameraOpenCV(CameraBase):
    def __init__(self) -> None:
        super().__init__()
        self.frame_idx = 0
        self.frame_last = None
        if use_video:
            video_path = r'D:\BoyangDeng\OneDrive\OneDrive - Michigan State University\Dataset\Weed\Weed2026\FarmNG\2026_08_25_14_34_43\video_origin_14_34_43_614076.mp4'
            # video_path = r'D:\BoyangDeng\OneDrive\OneDrive - Michigan State University\Dataset\Weed\Weed2026\FarmNG\2026_07_01_16_55_05\video_origin_16_55_05_280012.mp4'
            self.cap = cv2.VideoCapture(video_path)
            # self.cap = cv2.VideoCapture(0)
            assert self.cap.isOpened()
            pass
        elif use_img:
            im_dir = r'D:\BoyangDeng\OneDrive\OneDrive\Dataset\Weed\Weed2026\FarmNG'
            self.im_paths = cfg.list_dir(im_dir, [], '.jpg')
            if with_ann:
                self.ann_paths = cfg.list_dir(im_dir, [], '.xml')
            self.im_idx = 0
        cfg.cam_start_streaming_time = datetime.now()

    def capture(self):
        success = True
        img = None
        ann = None

        if not hasattr(self, '_fps_frame_count'):
            self._fps_frame_count = 0
            self._fps_last_time = time()
            self._fps_value = 0
        try:
            if use_video:
                # sleep(0.008)
                # sleep(0.001)
                # sleep(0.02)
                timestamp_ms = self.cap.get(cv2.CAP_PROP_POS_MSEC)
                success, img = self.cap.read()
                if cfg.frame_need_rotate_angle is not None:
                    img = cv2.rotate(img, cfg.frame_need_rotate_angle)

                # alpha = 1.2   # contrast (>1 increases contrast)
                # beta = 50     # brightness
                # img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
            elif use_img:
                if self.im_idx == -1:
                    success = False
                elif self.im_idx >= len(self.im_paths):
                    self.im_idx = -1
                else:
                    self.cur_img_path = self.im_paths[self.im_idx]
                    img = cv2.imread(self.cur_img_path)
                    if with_ann:
                        self.cur_ann_path = self.ann_paths[self.im_idx]
                        ann = get_ann_info_from_xml(self.cur_ann_path)
                self.im_idx += 1
        except Exception as e:
            print(e)
            success = False

        img_info_dict = {'img': img, 'ann': ann, 'status': 'Finished'}
        if cfg.use_camera_timestamp:
            frame_time = cfg.cam_start_streaming_time + timedelta(milliseconds=timestamp_ms)
            img_info_dict['timestamp'] = frame_time
        else:
            img_info_dict['timestamp'] = datetime.now()
        img_info_dict['frame_idx'] = self.frame_idx

        if img_info_dict['img'] is not None:
            self.frame_last = img_info_dict['img']
        else:
            img_info_dict['img'] = self.frame_last
        cfg.camera_latest_frame = img_info_dict['img']
        cfg.camera_latest_frame_timestamp = img_info_dict['timestamp']
        cfg.camera_latest_frame_idx = img_info_dict['frame_idx']
        self.frame_idx += 1

        if success:
            img_info_dict['status'] = 'OK'
        # depth = frames.get_depth_frame()
        # if not depth: 
        #     continue

        self._fps_frame_count += 1
        now = time()
        if now - self._fps_last_time >= 1.0:
            self._fps_value = np.round(self._fps_frame_count / (now - self._fps_last_time), 2)
            self._fps_last_time = now
            self._fps_frame_count = 0
        cfg.global_camera_handler.camera_fps = self._fps_value
        return success, img_info_dict

    def end(self):
        self.cap.release()

def demo():
    pass


def main():
    demo()

if __name__ =='__main__':
    main()

# import numpy as np
# depth = frames.get_depth_frame()
# depth_data = depth.as_frame().get_data()
# np_image = np.asanyarray(depth_data)