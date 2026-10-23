from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QVBoxLayout, QSlider, QLabel, QFileDialog, QMainWindow, QButtonGroup, QApplication
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QCoreApplication, QObject, QRunnable, QThread, QThreadPool, pyqtSignal, QTimer, QRect

import serial
import numpy as np
import config as cfg
import time
import threading
import traceback
from sensing.worker_send_signal import WorkerSendSignalValve, WorkerSendSignalSpeed

class RelayMonitor:
    def __init__(self, allow_relay_control=True):
        # for weed platform
        self.allow_relay_control = allow_relay_control
        
        # self.wheel_radius = 6.5 * 2.54  # cm
        self.wheel_radius = 5 * 2.54  # cm small wheel
        self.wheel_perimeter = np.pi * 2 * self.wheel_radius

        self.speed_control = False
        self.valve_control = cfg.async_valve_control

        if self.speed_control:
            self.worker_send_signal_speed = WorkerSendSignalSpeed()
        if self.valve_control:
            self.worker_send_signal_valve = WorkerSendSignalValve()

    def read_magnetic_signal(self):
        with open('/home/agfoodsensinglab/Desktop/Arduino/speed_ave_count.txt', 'r') as f:
            ave_count_by_time = f.read()
            try:
                ave_count_by_time = float(ave_count_by_time)
            except Exception as e:
                print(e)
                ave_count_by_time = 0
        return ave_count_by_time

    def caculate_speed(self):
        ave_count_by_time = self.read_magnetic_signal()
        speed = ave_count_by_time * self.wheel_perimeter
        speed = round(speed, 3)
        return speed

    def relay_control(self):
        if self.allow_relay_control:
            self.monitor_relay()

    def monitor_relay(self):
        if self.speed_control:
            print('start async speed_control')
            # self.thread = QThread()
            # self.worker = WorkerSendSignalSpeed()
            # self.worker.moveToThread(self.thread)
            # self.thread.started.connect(self.worker.run)
            # self.worker.finished.connect(self.thread.quit)
            # self.worker.finished.connect(self.worker.deleteLater)
            # self.thread.finished.connect(self.thread.deleteLater)
            # self.thread.start()
            t1 = threading.Thread(target=self.worker_send_signal_speed.run, args=(), daemon=True)  
            t1.start()  
        if self.valve_control:
            print('start async valve_control')
            # self.thread = QThread()
            # self.worker = WorkerSendSignalValve()
            # self.worker.moveToThread(self.thread)
            # self.thread.started.connect(self.worker.run)
            # self.worker.finished.connect(self.thread.quit)
            # self.worker.finished.connect(self.worker.deleteLater)
            # self.thread.finished.connect(self.thread.deleteLater)
            # self.thread.start()
            self.worker_send_signal_valve.need_exit_async_run = False
            t2 = threading.Thread(target=self.worker_send_signal_valve.run, args=(), daemon=True)  
            t2.start()  
