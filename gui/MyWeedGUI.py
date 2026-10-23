import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))  # FasterRCNN

import config as cfg
if cfg.detection_model_name == 'v26l_68epochs_800_DINOv3':
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../ultralytics_DINOv3'))  # FasterRCNN
    # sys.path.insert(0, r'D:\BoyangDeng\Detection\ultralytics')  # FasterRCNN
else:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../ultralytics_basic'))  # FasterRCNN

from ultralytics import YOLO
cfg.YOLO = YOLO
if cfg.detection_model_name == 'v26l_68epochs_800_DINOv3':
    sys.path.insert(0, r'D:\BoyangDeng\FoundationModel\dinov3')

# import importlib.util

# def load_module(name, path):
#     spec = importlib.util.spec_from_file_location(name, path)
#     module = importlib.util.module_from_spec(spec)
#     spec.loader.exec_module(module)
#     return module

# config_model = load_module("config_model", r"D:\BoyangDeng\Detection\ultralytics/config_model.py")
# config = load_module("config", r"config.py")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python" # much slower

# from PyQt5.QtWidgets import *
# from PyQt5.QtGui import *
# from PyQt5.QtCore import *
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QVBoxLayout, QSlider, QLabel, QFileDialog, QMainWindow, QButtonGroup, QApplication
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QCoreApplication, QObject, QRunnable, QThread, QThreadPool, pyqtSignal, QTimer, QRect
from PyQt5 import QtCore
from PyQt5.QtGui import QColor
# from  PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
# from  PyQt5.QtMultimediaWidgets import QVideoWidget
import torch
torch.backends.cudnn.benchmark = True
import json
import os
import threading
from typing import Optional
from time import time, sleep
import datetime
import numpy as np 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from image_display import ImageAdjustmentUI
from MyWeed import Ui_OpenWeedGUI
from drawing_label import DrawingLabel
from roi_window import ROI_Window


from detection.detection import WorkerDetection, WorkerCameraRetrieve, WorkerDetectionTrack
import sys
import traceback

def global_exception_handler(exc_type, exc_value, exc_traceback):
    """Catches hidden exceptions in PyQt5 slots and prints them."""
    print("CRITICAL ERROR CAUGHT:", file=sys.stderr)
    traceback.print_exception(exc_type, exc_value, exc_traceback)
    sys.exit(1)

sys.excepthook = global_exception_handler


class MainWindow(QMainWindow, Ui_OpenWeedGUI):
    ExpClicked = pyqtSignal(str)

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)

        cfg.global_window = self

        self.group = QButtonGroup(self)
        self.group.setExclusive(True)

        if cfg.use_old_GUI_functions:
            self.group.addButton(self.Exposure_offBox)
            self.group.addButton(self.Exposure_onceBox)
            self.group.addButton(self.Exposure_continueBox)
            self.Exposure_offBox.toggled.connect(cfg.global_camera_handler.set_exposure)
            self.Exposure_continueBox.toggled.connect(cfg.global_camera_handler.set_exposure)

        self.actionSet_ROI.triggered.connect(self.open_ROI_window)

        self.cur_model = self.comboBox_pt.currentText()

        self.ExpState = "Continous"

        cfg.global_timer.timer_camera.timeout.connect(self.show_camera)
        # cfg.global_timer.timer_img.timeout.connect(self.show_img)
        # cfg.global_timer.timer_video.timeout.connect(self.show_video)

        self.ReControl.clicked.connect(self.relayControl)

        cfg.global_detector.cur_model = self.comboBox_pt.currentText()
        cfg.global_detector.cur_model_version = self.comboBox_model_name.currentText()
        self.model_selection_finished = True

        self.thread_detection = {}
        self.worker_detection = {}
        self.worker_detection_idx = 0

        self.init_components()

    def init_components(self):
        self.comboBox_model_name.currentIndexChanged.connect(self.combo_box_YOLO_index_change)
        self.comboBox_pt.currentIndexChanged.connect(self.model_change)
        if cfg.use_old_GUI_functions:
            self.camMButton.clicked.connect(self.show_adjustment_window)

        self.stopVideoButton.clicked.connect(cfg.global_video_handler.stop_video)
        self.runVideoButton.clicked.connect(cfg.global_video_handler.play_video_on_thread)
        self.OpenCamButton.clicked.connect(cfg.global_camera_handler.open_camera)
        self.save_check.clicked.connect(cfg.global_image_handler.create_save_path)
        if cfg.use_old_GUI_functions:
            self.OpenVideoButton.clicked.connect(cfg.global_video_handler.video_path_dialog)
            self.fileButton.clicked.connect(cfg.global_image_handler.open_img)
            self.clearButton.clicked.connect(cfg.global_video_handler.clear_frame)
        # self.relayControlButton.clicked.connect(cfg.global_relay_monitor.relay_control)

    def show_camera(self):
        self.update_current_vehicle_speed()

        cfg.global_detector.main_loop()

        for nearest_valve_idx in cfg.valve_states_dict:
            cfg.valve_states_dict[nearest_valve_idx] = 'Closed'   
            for valve_info in cfg.nearest_valve_idxes_open_now.copy():
                assert isinstance(valve_info, tuple)
                valve_idx_in_process = valve_info[0]
                if nearest_valve_idx == valve_idx_in_process:
                    cfg.valve_states_dict[nearest_valve_idx] = 'Open'
                    break
            item = self.resultWidget.item(nearest_valve_idx-1)
            _translate = QtCore.QCoreApplication.translate
            if item is not None:
                item.setText(_translate("OpenWeedGUI", "Valve No."+str(nearest_valve_idx)+": " + cfg.valve_states_dict[nearest_valve_idx]))      
                if cfg.valve_states_dict[nearest_valve_idx] == 'Open':  
                    item.setForeground(QColor('#32cd32'))
                else:
                    item.setForeground(QColor('#222222'))

    def relayControl(self):
        valve_idx = self.spinBox.cleanText()
        valve_idx = int(valve_idx)

        if valve_idx in cfg.nearest_valve_idxes_manual_control:
            cfg.nearest_valve_idxes_manual_control.remove(valve_idx)
            for valve_info in cfg.nearest_valve_idxes_in_process.copy():
                assert isinstance(valve_info, tuple)
                valve_idx_in_process = valve_info[0]
                if valve_idx == valve_idx_in_process:
                    cfg.nearest_valve_idxes_in_process.remove(valve_info)
                    break
        else:
            cfg.nearest_valve_idxes_manual_control.append(valve_idx)
        
    def launch_detect_thread(self):
        print('launch_detect_thread')
        # self.thread_detection[self.worker_detection_idx] = QThread()
        # self.worker_detection[self.worker_detection_idx] = WorkerDetection()
        # self.worker_detection[self.worker_detection_idx].moveToThread(self.thread_detection[self.worker_detection_idx])
        # self.thread_detection[self.worker_detection_idx].started.connect(self.worker_detection[self.worker_detection_idx].run)
        # self.worker_detection[self.worker_detection_idx].finished.connect(self.thread_detection[self.worker_detection_idx].quit)
        # self.worker_detection[self.worker_detection_idx].finished.connect(self.worker_detection[self.worker_detection_idx].deleteLater)
        # self.thread_detection[self.worker_detection_idx].finished.connect(self.thread_detection[self.worker_detection_idx].deleteLater)
        # self.thread_detection[self.worker_detection_idx].start()

        worker_detection = WorkerDetection()
        t1 = threading.Thread(target=worker_detection.run, args=(), daemon=True)  
        t1.start()  
        if cfg.use_WorkerDetectionTrack:
            worker_detection_track = WorkerDetectionTrack()
            t2 = threading.Thread(target=worker_detection_track.run, args=(), daemon=True)  
            t2.start()  
        self.worker_detection_idx += 1

    def launch_camera_retrieve_thread(self):
        print('launch_camera_retrieve_thread')

        worker = WorkerCameraRetrieve()
        t1 = threading.Thread(target=worker.run, args=(), daemon=True)  
        t1.start()  

    def show_adjustment_window(self):
        self.adjustment_window = QMainWindow(self)
        self.adjustment_ui = ImageAdjustmentUI(self.adjustment_window)
        self.adjustment_ui.sliderValuesChanged.connect(self.adjust_image_by_slider)
        self.adjustment_window.setCentralWidget(self.adjustment_ui)
        self.adjustment_window.setWindowTitle("Image Adjustment")
        self.adjustment_window.setGeometry(200, 200, 300, 200)
        self.adjustment_window.show()

    def open_ROI_window(self):
        ROIWindow = ROI_Window(self)
        ROIWindow.size_signal.connect(self.video.resize_frame(self))
        ROIWindow.show()

    def update_combo_box_with_model_options(self):
        self.comboBox_pt.clear()
        if cfg.global_detector.cur_model_version == "yolov8":
            self.comboBox_pt.addItems(["default.pt"])

    def combo_box_YOLO_index_change(self):
        cfg.global_detector.model_is_changed = True
        cfg.global_detector.cur_model_version = self.comboBox_model_name.currentText()
        self.update_combo_box_with_model_options(self)
        sleep(0.1)
        assert cfg.async_detection
        if cfg.async_detection:
            cfg.global_detector.launch_detect_thread()

    def model_change(self):
        cfg.global_detector.model_is_changed = True
        cfg.global_detector.cur_model = self.comboBox_pt.currentText()
        cfg.global_detector.cur_model_version = self.comboBox_model_name.currentText()
        sleep(0.1)
        self.model_selection_finished = True
        assert cfg.async_detection
        if cfg.async_detection:
            cfg.global_detector.launch_detect_thread()

    def update_current_vehicle_speed(self):
        cur_speed, ave_cur_speed = cfg.global_speed_monitor.caculate_speed()
        self.label_speed.setText(f'Speed (cm/s) Ave: {ave_cur_speed}, Current: {cur_speed}')

    def adjust_image_by_slider(self, slider_values):
        self.saturation_factor, self.hue_shift, self.value_factor = slider_values

    def adjust_image_in_hsv(self, frame):
        from im_operation.im_operation import adjust_image_in_hsv
        adjusted_image = adjust_image_in_hsv(frame)
        return adjusted_image


if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWin = MainWindow()
    myWin.show()
    sys.exit(app.exec_())
