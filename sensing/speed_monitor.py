from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QVBoxLayout, QSlider, QLabel, QFileDialog, QMainWindow, QButtonGroup, QApplication
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QCoreApplication, QObject, QRunnable, QThread, QThreadPool, pyqtSignal, QTimer, QRect
import config as cfg
import serial
import numpy as np
import re
import os


need_start_relay = True


class WorkerSpeed(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.sr = SpeedReceiver()

    def read(self):
        data = self.sr.read_magnetic_signal()
        return data

    def run(self):
        while True:
            self.read()
        self.finished.emit()


class SpeedReceiver:
    def __init__(self):
        self.serial_portal = '/dev/ttyACM0'
        self.port = 9600
        self.serial_controller = serial.Serial(self.serial_portal, self.port, timeout=1)
        self.wheel_radius = 6.5 * 2.54  # cm
        self.wheel_perimeter = np.pi * 2 * self.wheel_radius

    def receive_magnetic_signal(self):
        if self.serial_controller.in_waiting > 0:
            data = self.serial_controller.readline().decode("ascii")
            with open('/home/agfoodsensinglab/Desktop/Arduino/speed_ave_count.txt', 'w') as f:
                f.write(data)
        return data


class SpeedMonitor:
    def __init__(self):
        # self.wheel_radius = 6.5 * 2.54  # cm
        self.wheel_radius = 5 * 2.54  # cm small wheel
        self.wheel_perimeter = np.pi * 2 * self.wheel_radius

    def read_magnetic_signal(self):
        with open('/home/agfoodsensinglab/Desktop/Arduino/speed_ave_count.txt', 'r') as f:
            ave_count_by_time = f.read()
            ave_count_by_time = float(ave_count_by_time)
        return ave_count_by_time

    def extract_speeds_from_file(self, file_path):
        speed_pattern = re.compile(r"Speed:\s+([\d.]+)\s+cm/s\s+\(([\d.]+)\s+mph\)")
        ave_speed_pattern = re.compile(r"AVG Speed:\s+([\d.]+)\s+cm/s\s+\(([\d.]+)\s+mph\)")
        speeds = self.extract_speeds_from_file_imp(file_path, speed_pattern)
        # ave_speeds = self.extract_speeds_from_file_imp(file_path, ave_speed_pattern)
        ave_speeds = 0
        return speeds, ave_speeds

    def extract_speeds_from_file_imp(self, file_path, speed_pattern, max_lines=500):
        speeds = []
        with open(file_path, 'r') as file:
            lines = file.readlines()
            line_count = len(lines)
            for line in lines:
                match = speed_pattern.search(line)
                if match:
                    cm_per_s = float(match.group(1))
                    mph = float(match.group(2))
                    speeds.append({
                        'cm_per_s': cm_per_s,
                        'mph': mph
                    })
        if line_count > max_lines:
            with open(file_path, 'w') as file:
                file.truncate()
            print(f"File exceeded {max_lines} lines. It has been cleared.")
        return speeds

    def caculate_speed(self):
        # ave_count_by_time = self.read_magnetic_signal()
        # speed = ave_count_by_time * self.wheel_perimeter
        log_path = './arduino_code/valve_control/signal/speed/speed_info.txt'
        log_dir = os.path.dirname(log_path)
        if os.path.exists(log_dir) is False:
            os.makedirs(log_dir, exist_ok=True)
        with open(log_path, 'w') as f:
            f.write('')
        speeds, ave_speeds = self.extract_speeds_from_file(log_path) 
        speed =0
        ave_speed =0
        if len(speeds) > 2:
            speed = speeds[-2]['cm_per_s']
            ave_speed = speeds[-1]['cm_per_s']
            speed = round(speed, 3)
            ave_speed = round(ave_speed, 3)
            cfg.current_physical_speed = speed
            assert False, "Speed is not updated in config.py"
        else:
            speed = cfg.current_physical_speed 
            ave_speed = cfg.current_physical_speed 
        return speed, ave_speed

    def start_worker(self):
        self.thread = QThread()
        self.worker = WorkerSpeed()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()
