import sys
import cv2
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QVBoxLayout, QSlider, QLabel, QFileDialog, QMainWindow, QButtonGroup, QApplication
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QCoreApplication, QObject, QRunnable, QThread, QThreadPool, pyqtSignal, QTimer, QRect
# from  PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
# from  PyQt5.QtMultimediaWidgets import QVideoWidget
import torch
import json
import os
import threading
from typing import Optional
from time import time, sleep
import datetime
import numpy as np
from queue import SimpleQueue, Queue
from itertools import count
from loguru import logger
# from yolox.data.data_augment import ValTransform
# from yolox.data.datasets import COCO_CLASSES
# from yolox.exp import get_exp
# from yolox.utils import fuse_model, get_model_info, postprocess, vis
import ctypes
import config as cfg


class VideoHandler():
    def __init__(self) -> None:
        self.vclose = True
        self.videoPath = []
        self.makeResize = True
        self.size_data = cfg.detect_size_with_pad
        self.is_streaming = False

    def video_path_dialog(self):
        self.videoPath, _ = QFileDialog.getOpenFileName(self, "open video file", ".", "Video Files (*.mp4 *.flv *.ts *.mts *.avi)") 
        # playVideoWithThreading()

    def video_streaming(self, queue):
        self.is_streaming = True
        cap = cv2.VideoCapture(self.videoPath)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.videoSlider.setMaximum(int(frames/24))
        if not cap.isOpened():
            print("Cannot open Video File")
            exit()
        print("video started")
        while self.vclose:
            ret, frame = cap.read()
            if not ret:
                if frame is None:
                    print("The video has end.")
                else:
                    print("Read video error!")
                    break
            if queue.full():
                with queue.mutex:
                    queue.queue.clear()
            queue.put(frame)
            print("put")
        # self.timer_video.stop()
        # self.label_pic.clear()
        print("video stopped")

    def show_video(self):
        if not self.queue.empty() and not self.detect_check.isChecked():
            c_frame = self.queue.get()
            # cv2.putText(c_frame, f'Time: {int(time())}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 6)
            label_width = self.label_pic.width()
            label_height = self.label_pic.height()
            c_frame = cv2.cvtColor(c_frame, cv2.COLOR_BGR2RGB)
            temp_imgSrc = QImage(c_frame, c_frame.shape[1], c_frame.shape[0], c_frame.shape[1] * 3,
                                    QImage.Format_RGB888)
            pixmap_imgSrc = QPixmap.fromImage(temp_imgSrc).scaled(label_width, label_height)
            self.label_pic.setPixmap(pixmap_imgSrc)
        elif not self.queue.empty() and self.detect_check.isChecked():
            new_size = (640, 640)
            c_frame = self.queue.get()
            start_time = time()
            c_frame = cv2.cvtColor(c_frame, cv2.COLOR_BGR2RGB)
            resize_frame = cv2.resize(c_frame, new_size)
            results = self.model_detect_one_frame(resize_frame)
            final_frame = self.plot_boxes(results, resize_frame)
            end_time = time()
            fps = 1/np.round(end_time-start_time, 2)
            cv2.putText(final_frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            label_width = self.label_pic.width()
            label_height = self.label_pic.height()
            temp_imgSrc = QImage(final_frame, final_frame.shape[1], final_frame.shape[0], final_frame.shape[1] * 3,
                                    QImage.Format_RGB888)
            pixmap_imgSrc = QPixmap.fromImage(temp_imgSrc).scaled(label_width, label_height)
            self.label_pic.setPixmap(pixmap_imgSrc)
            # print(json_save_dir,results,Jnum)
            # print(c_frame.shape)
            if int(time() % cfg.global_window.save_num.value()) == 0 and cfg.global_window.save_check.isChecked():  # some  variable parameters
                img_name = str(cfg.global_window.Inum) + ".png"
                cv2.imwrite(".../DetectedImages/"+img_name, final_frame)
                json_save_dir = ".../json"

                cfg.global_detector.save_json(results, json_save_dir, cfg.global_window.Inum)
                print(cfg.global_window.Inum)-++++++++++++++.3
                cfg.global_window.Inum += 1

    def play_video_on_thread(self):
        if self.timer_video.isActive() is False:
            threading.Thread(target=self.video_streaming, args=(self.queue,), daemon=True).start()
            self.timer_video.start(20)
            print("video start")
        else:
            self.vclose = False
            self.timer_video.stop()
            # cam.stop_streaming()
            self.label_pic.clear()
            # self.OpenCamButton.setText("Open Camera")

    def clear_frame(self):
        if self.timer_img.isActive():
            self.timer_video.stop()
            self.label_pic.clear()

    def resize_frame(self):
        assert False
        size_data = self.size_data
        print(size_data)
        self.makeResize = True
        if size_data[0] == "reset":
            self.makeResize = False
        else:
            self.makeResize = True
            # self.label_pic.resize(size_data[2],size_data[3])
            scale_factor_x = 2464/self.label_pic.width()
            scale_factor_y = 2056/self.label_pic.height()

            self.original_x = int(size_data[0] * scale_factor_x)
            self.original_y = int(size_data[1] * scale_factor_y)
            self.original_width = int(size_data[2] * scale_factor_x)
            self.original_height = int(size_data[3] * scale_factor_y)

            scale_factor_x = self.original_size['x'] / self.scaled_size[0]
            scale_factor_y = self.original_size['y'] / self.scaled_size[1]

            x, y, width, height = size_data[0], size_data[1], size_data[2], size_data[3]
            # x, y = map(int, position_text.split(","))
            self.mapped_rect = QRect( int(x), int(y), int(width), int(height), )  

    def stop_video(self):
        self.vclose = False
