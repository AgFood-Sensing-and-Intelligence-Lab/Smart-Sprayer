from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QVBoxLayout, QSlider, QLabel, QFileDialog, QMainWindow, QButtonGroup, QApplication
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QCoreApplication, QObject, QRunnable, QThread, QThreadPool, pyqtSignal, QTimer, QRect
from PyQt5 import QtCore

import serial
import numpy as np
import torch
import torchvision
import config as cfg
from time import time, sleep
import traceback
import threading
import os
from collections import defaultdict
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from threading import Timer

lock = threading.Lock()

valve_signal_idx = 0

# class SignalValve(QObject):
class SignalValve():
    # finished = pyqtSignal()
    # progress = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        # self.signal_write_path = 'arduino_code/valve_control/signal_1to12.txt'
        # self.signal_write_path2 = 'arduino_code/valve_control/signal_13to24.txt'
        # with open(self.signal_write_path, 'w') as f:
        #     f.write('0')
        # with open(self.signal_write_path2, 'w') as f:
        #     f.write('0')
        self.signal_write_dir_odd = 'arduino_code/valve_control/signal/odd'
        self.signal_write_dir_even = 'arduino_code/valve_control/signal/even'
        self.signal_write_dir_Mark = 'arduino_code/valve_control/signal/Mark'


    def run_centered(self, nearest_valve_idxes_cur, valve_signal_idx):
        pass

    def safe_write_to_file(self, file_path, text, max_retries=5):
            """Attempts to write to a file, retrying if Windows has locked it."""
            for attempt in range(max_retries):
                try:
                    with open(file_path, 'w') as f:
                        f.write(text)
                        f.flush()
                    return True
                except PermissionError:
                    # The serial script is reading it. Wait 2ms and try again.
                    sleep(0.002) 
                except Exception as e:
                    # Print to standard console ONLY to avoid PyQt GUI crashes
                    print(f"Safe write failed: {e}")
                    break
            return False

    def check_more_open_request(self, nearest_valve_idx, actual_valve_idx, control_path, only_close_valve, time_offset, valve_opening_duration):
        valve_info_cur = (nearest_valve_idx, actual_valve_idx, time_offset)
        
        # 1. Define the safe closing logic
        def close_valve():
            text_close = ""
            if cfg.use_opening_duration_to_close_valve:
                text_close = f"{actual_valve_idx}-0"
            self.safe_write_to_file(control_path, text_close)
            
            if valve_info_cur in cfg.nearest_valve_idxes_open_now:
                cfg.nearest_valve_idxes_open_now.remove(valve_info_cur)

        # 2. If it's a close-only request, execute and exit
        if only_close_valve:
            close_valve()
            return

        # 3. --- OPEN THE VALVE ---
        if valve_info_cur not in cfg.nearest_valve_idxes_open_now:
            cfg.nearest_valve_idxes_open_now.append(valve_info_cur)
            
        text_open = f"{actual_valve_idx}-1"
        self.safe_write_to_file(control_path, text_open)

        # 4. --- CALCULATE DURATION ---
        if valve_opening_duration is not None:
            used_valve_opening_duration = valve_opening_duration
        else:
            used_valve_opening_duration = cfg.valve_opening_duration
            
        used_valve_opening_duration = min(cfg.valve_opening_duration_max, used_valve_opening_duration)
        used_valve_opening_duration = max(cfg.valve_opening_duration_min, used_valve_opening_duration)
        used_valve_opening_duration = round(used_valve_opening_duration, 3)

        # 5. --- CLEAN UP QUEUE ---
        for valve_info in cfg.nearest_valve_idxes_in_process.copy():
            if isinstance(valve_info, tuple):
                valve_idx = valve_info[0]            
                valve_time_offset = valve_info[2]            
            else:
                valve_idx = valve_info
                valve_time_offset = None
                
            if valve_idx == nearest_valve_idx and time_offset == valve_time_offset:
                cfg.nearest_valve_idxes_in_process.remove(valve_info)
                break

        # 6. --- SCHEDULE THE CLOSE ---
        # Starts the background timer so this thread can move on
        threading.Timer(used_valve_opening_duration, close_valve).start()

    def check_more_open_request_v1(self, nearest_valve_idx, actual_valve_idx, control_path, only_close_valve, time_offset, valve_opening_duration):
        valve_info_cur = (nearest_valve_idx, actual_valve_idx, time_offset)
        
        # Define the closing logic as an isolated function
        def close_valve():
            text_close = f"{actual_valve_idx}-0"
            with open(control_path, 'w') as f:
                f.write(text_close)
                f.flush()
            if valve_info_cur in cfg.nearest_valve_idxes_open_now:
                cfg.nearest_valve_idxes_open_now.remove(valve_info_cur)

        # If it's a close-only request, execute immediately and exit
        if only_close_valve:
            close_valve()
            return

        # --- OPEN THE VALVE ---
        if valve_info_cur not in cfg.nearest_valve_idxes_open_now:
            cfg.nearest_valve_idxes_open_now.append(valve_info_cur)
            
        text_open = f"{actual_valve_idx}-1"
        with open(control_path, 'w') as f:
            f.write(text_open)
            f.flush()

        # --- CALCULATE OPEN DURATION ---
        if valve_opening_duration is not None:
            used_valve_opening_duration = valve_opening_duration
        else:
            used_valve_opening_duration = cfg.valve_opening_duration
            
        used_valve_opening_duration = min(cfg.valve_opening_duration_max, used_valve_opening_duration)
        used_valve_opening_duration = max(cfg.valve_opening_duration_min, used_valve_opening_duration)
        used_valve_opening_duration = round(used_valve_opening_duration, 3)

        # Clean up process queue immediately
        for valve_info in cfg.nearest_valve_idxes_in_process.copy():
            if isinstance(valve_info, tuple):
                valve_idx = valve_info[0]            
                valve_time_offset = valve_info[2]            
            else:
                valve_idx = valve_info
                valve_time_offset = None
                
            if valve_idx == nearest_valve_idx and time_offset == valve_time_offset:
                cfg.nearest_valve_idxes_in_process.remove(valve_info)
                break

        # --- SCHEDULE THE CLOSE ---
        # Replaces the blocking sleep() with a lightweight background timer
        Timer(used_valve_opening_duration, close_valve).start()

    def check_more_open_request_v0(self, nearest_valve_idx, actual_valve_idx, control_path, only_close_valve, time_offset, valve_opening_duration):
        if only_close_valve is False:
            valve_info_cur = (nearest_valve_idx, actual_valve_idx, time_offset)
            if valve_info_cur not in cfg.nearest_valve_idxes_open_now:
                cfg.nearest_valve_idxes_open_now.append(valve_info_cur)
            text_open = str(actual_valve_idx) + '-1'
            with open(control_path, 'w') as f:
                f.write(text_open)
                f.flush()
            if valve_opening_duration is not None:
                used_valve_opening_duration = valve_opening_duration
            else:
                used_valve_opening_duration = cfg.valve_opening_duration
            used_valve_opening_duration = min(cfg.valve_opening_duration_max, used_valve_opening_duration)
            used_valve_opening_duration = max(cfg.valve_opening_duration_min, used_valve_opening_duration)
            used_valve_opening_duration = round(used_valve_opening_duration, 3)
            sleep(used_valve_opening_duration)

            # cfg.log_xalg(f'used_valve_opening_duration: {used_valve_opening_duration}',show=False)
            for valve_info in cfg.nearest_valve_idxes_in_process.copy():
                if isinstance(valve_info, tuple):
                    valve_idx = valve_info[0]            
                    valve_time_offset = valve_info[2]            
                else:
                    valve_idx = valve_info
                if valve_idx == nearest_valve_idx and time_offset == valve_time_offset:
                    cfg.nearest_valve_idxes_in_process.remove(valve_info)
                    break

            # more_open_request = False
            # for valve_info in cfg.nearest_valve_idxes_in_process.copy():
            #     if isinstance(valve_info, tuple):
            #         valve_idx = valve_info[0]
            #     else:
            #         valve_idx = valve_info
            #     if valve_idx == nearest_valve_idx:
            #         more_open_request = True
            #         break
            # if more_open_request:
            #     cfg.log_xalg(f'nozzle {nearest_valve_idx} more_open_request, no close')
            #     return
        
        text_close = str(actual_valve_idx) + '-0'
        with open(control_path, 'w') as f:
            f.write(text_close)
            f.flush()
        if valve_info_cur in cfg.nearest_valve_idxes_open_now:
            cfg.nearest_valve_idxes_open_now.remove(valve_info_cur)

    def create_signal_file(self, signal_file):
        directory_path = os.path.dirname(signal_file)
        os.makedirs(directory_path, exist_ok=True)
        # Create the file
        with open(signal_file, "w") as f:
            f.write("")
        assert os.path.isfile(signal_file)

    def run_valve_triger_with_delay(self, nearest_valve_idxes_cur, time_offset, only_close_valve, valve_signal_idx, is_close_nozzles=True, valve_opening_duration=None):
            try: 
                delay_time = -1
                delay_time_actual = 0
                
                # Calculate the initial delay before the valve opens
                if only_close_valve is False:
                    if cfg.end_effector_no_delay:
                        delay_time = 0
                    else:
                        if cfg.use_time_offset and time_offset >= 0:
                            delay_time = min(time_offset, cfg.max_time_offset)
                        else:   
                            if is_close_nozzles:
                                delay_time = cfg.delay_time_run_1_to_12
                            else:
                                delay_time = cfg.delay_time_run_13_to_24
                                
                    delay_time_actual = max(0, delay_time - cfg.valve_response_time - cfg.software_response_time - cfg.spray_touch_ground_time)
                    delay_time_actual = round(delay_time_actual, 3)

                # Pre-calculate all nozzle properties before scheduling
                valves_to_process = []
                for nearest_valve_idx in nearest_valve_idxes_cur:
                    nearest_valve_idx = max(cfg.valid_valve_range[0], nearest_valve_idx)
                    nearest_valve_idx = min(cfg.valid_valve_range[-1], nearest_valve_idx)
                    
                    if nearest_valve_idx % 2 == 0:
                        control_dir = self.signal_write_dir_even
                        actual_valve_idx = cfg.nozzle_mapping_all[nearest_valve_idx]
                        arduino_idx = 'close'
                    else:
                        control_dir = self.signal_write_dir_odd
                        actual_valve_idx = cfg.nozzle_mapping_all[nearest_valve_idx]
                        arduino_idx = 'far'

                    if getattr(cfg, 'is_for_Mark_sprayer', False):
                        control_dir = self.signal_write_dir_Mark
                        
                    control_file_name = 'signal_nozzle_' + str(nearest_valve_idx) + '.txt'
                    control_path = os.path.join(control_dir, control_file_name)
                    
                    if not os.path.exists(control_path):
                        self.create_signal_file(control_path)

                    # cfg.log_xalg(f'valve->pixel:{nearest_valve_idx}, actual:{actual_valve_idx}, delay:{delay_time}, (actual) {delay_time_actual}, arduino_idx:{arduino_idx}', show=False)
                    valves_to_process.append((nearest_valve_idx, actual_valve_idx, control_path))

                # Inner function to execute the actual open requests
                def execute_valves():
                    for v_idx, actual_idx, path in valves_to_process:
                        self.check_more_open_request(v_idx, actual_idx, path, only_close_valve, time_offset, valve_opening_duration)

                # Schedule the execution instead of blocking the thread
                if delay_time_actual > 0:
                    Timer(delay_time_actual, execute_valves).start()
                else:
                    execute_valves()

            except Exception as e:
                cfg.log_xalg(traceback.format_exc())
                cfg.log_xalg(e)

    def run_valve_triger_with_delay_v0(self, nearest_valve_idxes_cur, time_offset, only_close_valve, valve_signal_idx, is_close_nozzles=True, valve_opening_duration=None):
        try: 
            delay_time = -1
            if only_close_valve is False:
                if cfg.end_effector_no_delay:
                    delay_time = 0
                else:
                    if cfg.use_time_offset and time_offset >= 0:
                        delay_time = min(time_offset, cfg.max_time_offset)
                    else:   
                        if is_close_nozzles:
                            delay_time = cfg.delay_time_run_1_to_12
                        else:
                            delay_time = cfg.delay_time_run_13_to_24
                    delay_time_actual = max(0, delay_time-cfg.valve_response_time-cfg.software_response_time-cfg.spray_touch_ground_time)
                if delay_time_actual > 0:
                    delay_time_actual = round(delay_time_actual, 3)
                    sleep(delay_time_actual)
            arduino_idx = 0
            for nearest_valve_idx in nearest_valve_idxes_cur:
                nearest_valve_idx = max(cfg.valid_valve_range[0], nearest_valve_idx)
                nearest_valve_idx = min(cfg.valid_valve_range[-1], nearest_valve_idx)
                if nearest_valve_idx % 2==0:
                    # even 
                    control_dir = self.signal_write_dir_even
                    actual_valve_idx=cfg.nozzle_mapping_all[nearest_valve_idx]
                    arduino_idx = 'close'
                else:
                    #odd
                    control_dir = self.signal_write_dir_odd
                    actual_valve_idx=cfg.nozzle_mapping_all[nearest_valve_idx]
                    arduino_idx = 'far'

                control_file_name = 'signal_nozzle_' + str(nearest_valve_idx) + '.txt'

                if cfg.is_for_Mark_sprayer:
                    control_dir = self.signal_write_dir_Mark
                control_path = os.path.join(control_dir, control_file_name)
                if not os.path.exists(control_path):
                    self.create_signal_file(control_path)

                assert os.path.exists(control_path), f"Control path does not exist: {control_path}"
                cfg.log_xalg(f'valve->pixel:{nearest_valve_idx}, actual:{actual_valve_idx}, delay :{delay_time}, (actual) {delay_time_actual}, arduino_idx:{arduino_idx}', show=False)

                self.check_more_open_request(nearest_valve_idx, actual_valve_idx, control_path, only_close_valve, time_offset, valve_opening_duration)
        except Exception as e:
            cfg.log_xalg(traceback.format_exc())
            cfg.log_xalg(e)
        
# class WorkerSendSignalValve(QObject):
class WorkerSendSignalValve():
    # finished = pyqtSignal()
    # progress = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.last_results_boxes = None
        self.worker = SignalValve()
        self.executor = ThreadPoolExecutor(max_workers=32)

        self.global_idx = 0 
        self.nozzle_idxes = list(cfg.valve_idx_to_text_dict.keys())
        self.need_exit_async_run = False
        self.send_signal_fps = 0
        self.instant_id_track = -1
    
    def calculate_direction(self, box, box_last):
        xmin = int(box[0])
        ymin = int(box[1])
        xmax = int(box[2])
        ymax = int(box[3])
        xmin_last = int(box_last[0])
        ymin_last = int(box_last[1])
        xmax_last = int(box_last[2])
        ymax_last = int(box_last[3])
        if xmin-xmin_last < -10:
            direction = 1
        else:
            direction = -1
        return direction

    def check_box_location(self, bboxes):
        nearest_valve_idxes_new = []
        
        # --- 1. SPATIAL MEMORY INITIALIZATION & CLEANUP ---
        current_time = time()
        if not hasattr(self, 'recently_triggered_boxes'):
            self.recently_triggered_boxes = []  # Stores (bbox_xyxy, timestamp)
            self.duplicate_iou_threshold = 0.4  # Tune this: higher means stricter matching
            self.box_memory_duration = 1.5     # Seconds to remember a box. Set this slightly higher than the time it takes a weed to clear the trigger zone.
            
        # Remove old boxes from memory that have passed the cooldown duration
        self.recently_triggered_boxes = [
            (b, t) for b, t in self.recently_triggered_boxes 
            if current_time - t < self.box_memory_duration
        ]
        # --------------------------------------------------
        if len(bboxes) == 0:
            return nearest_valve_idxes_new

        # --- 2. VECTORIZED DUPLICATE CHECK ---
        current_boxes_xyxy = [[int(box[0]), int(box[1]), int(box[2]), int(box[3])] for box in bboxes]
        current_tensor = torch.tensor(current_boxes_xyxy, dtype=torch.float32)
        
        is_duplicate_mask = torch.zeros(len(bboxes), dtype=torch.bool)
        if self.recently_triggered_boxes:
            recent_tensor = torch.tensor([b for b, t in self.recently_triggered_boxes], dtype=torch.float32)
            # Compute IoU matrix for all current boxes against all memory boxes
            ious = torchvision.ops.box_iou(current_tensor, recent_tensor) 
            is_duplicate_mask = (ious > self.duplicate_iou_threshold).any(dim=1)
        # -------------------------------------
        
        if cfg.is_pad_height:
            x_start_gui = cfg.nozzle_trigger_x_start
            x_end_gui = cfg.nozzle_trigger_x_end
        else:
            x_start_gui = cfg.detect_size_pad_half+cfg.nozzle_trigger_x_start
            x_end_gui = cfg.detect_size_pad_half+cfg.nozzle_trigger_x_end
            
        for i_box, box in enumerate(bboxes):
            x,y,x1,y1, conf, sku_idx = box[:6]
            xmin = int(x)
            ymin = int(y)
            xmax = int(x1)
            ymax = int(y1)
            
            if conf < cfg.spray_conf_t:
                continue

            # --- NEW OVERLAP SUPPRESSION FILTER ---
            # --- UPDATED: Filter out crops AND suppressed fake weeds ---
            if sku_idx == cfg.filtered_sku_id:    # Instantly catch the dummy flag
                continue
            # --------------------------------------
            try:
                # Filter out crops
                sku_label = cfg.global_detector.class_names[int(sku_idx)]
            except Exception as e:
                print(f"Error retrieving SKU label for index {sku_idx}: {e}")
                
            if sku_label in cfg.vegetable_names:
                continue

            if cfg.move_direction_required < 0:
                used_x = xmin
            else:
                used_x = xmax
            used_x = xmin if cfg.move_direction_required < 0 else xmax
            if not (x_start_gui < used_x < x_end_gui):
                continue

            # --- 2. DUPLICATE CHECK AGAINST MEMORY ---
            # Check the vectorized duplicate mask instead of looping
            if is_duplicate_mask[i_box]:
                continue

            # is_duplicate = False
            # current_box_xyxy = [xmin, ymin, xmax, ymax]
            
            # for recent_box, _ in self.recently_triggered_boxes:
            #     iou = cfg.global_detector.bb_intersection_over_union(current_box_xyxy, recent_box)
            #     if iou > self.duplicate_iou_threshold:
            #         is_duplicate = True
            #         break
            
            # # If we've seen this box recently, skip triggering it again
            # if is_duplicate:
            #     continue
            # -----------------------------------------
            
            # If it's a new box, save it to memory to prevent future triggers
            self.recently_triggered_boxes.append((current_boxes_xyxy[i_box], current_time))

            x_ave = (xmin+xmax)/2
            y_ave = (ymin+ymax)/2
            bbox_w = xmax - xmin
            bbox_h = ymax - ymin
            bbox_xy_center_wh = [x_ave, y_ave, bbox_w, bbox_h]
            nearest_valve_idxes_new_one_bbox = cfg.global_detector.find_closest_nozzle_to_weed(bbox_xy_center_wh)
            nearest_valve_idxes_new.extend(nearest_valve_idxes_new_one_bbox)
            
        nearest_valve_idxes_new = np.unique(nearest_valve_idxes_new).tolist()
        return nearest_valve_idxes_new

    def get_current_speed(self):
        speed = -1
        with open('./amiga_control/states.txt', 'r') as f:
            info = f.readlines()
            texts = info[-1].split(' ')
            try:
                idx = texts.index('speed')
                speed = texts[idx+1]
                speed = round(float(speed),3)
            except Exception as e:
                speed = -1
        return speed

    def send_signal(self, nearest_valve_idxes_cur, valve_signal_idx, time_offset, only_close_valve=False, valve_opening_duration=None):
        odd_nozzles = [n for n in nearest_valve_idxes_cur if n % 2 != 0]
        even_nozzles = [n for n in nearest_valve_idxes_cur if n % 2 == 0]
        # msg=f'open valves: {str(nearest_valve_idxes_cur)}, time_offset: {time_offset}'
        # cfg.log_xalg(msg)

        if len(even_nozzles) > 0:
            if cfg.even_idx_is_close_to_camera:
                is_close_nozzles = True
            else:
                is_close_nozzles = False

            # t2 = threading.Thread(target=self.worker.run_valve_triger_with_delay, args=(even_nozzles, time_offset, only_close_valve, valve_signal_idx, is_close_nozzles), daemon=True)  
            # t2.start()  
            # for even_nozzle in even_nozzles:
            #     t2 = threading.Thread(target=self.worker.run_valve_triger_with_delay, args=([even_nozzle], time_offset, only_close_valve, valve_signal_idx, is_close_nozzles, valve_opening_duration), daemon=True)  
            #     t2.start()  

            for even_nozzle in even_nozzles:
                self.executor.submit(
                    self.worker.run_valve_triger_with_delay, 
                    [even_nozzle], time_offset, only_close_valve, valve_signal_idx, is_close_nozzles, valve_opening_duration
                )

        if len(odd_nozzles) > 0:
            if cfg.even_idx_is_close_to_camera:
                is_close_nozzles = False
            else:
                is_close_nozzles = True

            # t1 = threading.Thread(target=self.worker.run_valve_triger_with_delay, args=(odd_nozzles, time_offset, only_close_valve, valve_signal_idx, is_close_nozzles), daemon=True)  
            # t1.start()  
            # for odd_nozzle in odd_nozzles:
            #     t1 = threading.Thread(target=self.worker.run_valve_triger_with_delay, args=([odd_nozzle], time_offset, only_close_valve, valve_signal_idx, is_close_nozzles, valve_opening_duration), daemon=True)  
            #     t1.start()  
            for odd_nozzle in odd_nozzles:
                self.executor.submit(
                    self.worker.run_valve_triger_with_delay, 
                    [odd_nozzle], time_offset, only_close_valve, valve_signal_idx, is_close_nozzles, valve_opening_duration
                )

    def add_valve_idxes_into_queue(self, nearest_valve_idxes_cur, valve_signal_idx, time_offset, id_track, valve_opening_duration):
            now = datetime.now()
            max_time_offset = cfg.max_time_to_close_valve_basic_t + valve_opening_duration + cfg.valve_response_time + cfg.software_response_time + cfg.spray_touch_ground_time
            max_time_to_close_valve = now + timedelta(seconds=max_time_offset)

            valid_valves = []
            estimate_open_time = now + timedelta(seconds=time_offset)

            for valve_idx in nearest_valve_idxes_cur:
                if time_offset < 0:
                    continue

                should_keep = True
                if cfg.filter_out_by_nearest_valve_idxes_in_process:
                    for valve_info in cfg.nearest_valve_idxes_in_process:
                        valve_idx_in = valve_info[0]
                        time_offset_in = valve_info[2]
                        time_triger_in = valve_info[3]
                        
                        if valve_idx == valve_idx_in:
                            estimate_open_time_in = time_triger_in + timedelta(seconds=time_offset_in)
                            time_gap = abs((estimate_open_time_in - estimate_open_time).total_seconds())
                            if time_gap < cfg.nearest_valve_idxes_in_process_time_gap:
                                should_keep = False
                                break # Break early to save cycles
                
                if should_keep:
                    valid_valves.append(valve_idx)

            if len(valid_valves) > 0:
                # cfg.log_xalg(f'------id_track: {id_track}, nozzles {valid_valves} start send_signal------', show=False)
                
                # Update process list in one go
                for valve_idx in valid_valves:
                    cfg.nearest_valve_idxes_in_process.append((valve_idx, max_time_to_close_valve, time_offset, now))
                
                # Send the batched list to send_signal ONCE, instead of looping
                self.send_signal(valid_valves, valve_signal_idx, time_offset, valve_opening_duration=valve_opening_duration)

    def add_valve_idxes_into_queue_v0(self, nearest_valve_idxes_cur, valve_signal_idx, time_offset, id_track, valve_opening_duration):
        # current_set = set(nearest_valve_idxes_cur)
        # processing_set = set(cfg.nearest_valve_idxes_in_process)
        # new_valves = list(current_set - processing_set)
        now = datetime.now()
        max_time_offset = cfg.max_time_to_close_valve_basic_t + valve_opening_duration + cfg.valve_response_time + cfg.software_response_time + cfg.spray_touch_ground_time
        max_time_to_close_valve = now + timedelta(seconds=max_time_offset)

        for valve_idx in nearest_valve_idxes_cur.copy():
            if time_offset < 0:
                nearest_valve_idxes_cur.remove(valve_idx)
            else:
                if cfg.filter_out_by_nearest_valve_idxes_in_process:
                    for valve_info in cfg.nearest_valve_idxes_in_process:
                        valve_idx_in = valve_info[0]
                        time_offset_in = valve_info[2]
                        time_triger_in = valve_info[3]
                        if valve_idx == valve_idx_in:
                            estimate_open_time_in = time_triger_in + timedelta(seconds=time_offset_in)
                            estimate_open_time = now + timedelta(seconds=time_offset)
                            time_gap = abs((estimate_open_time_in - estimate_open_time).total_seconds())
                            if time_gap < cfg.nearest_valve_idxes_in_process_time_gap:
                                nearest_valve_idxes_cur.remove(valve_idx_in)

        if len(nearest_valve_idxes_cur) > 0:
            cfg.log_xalg(f'------id_track: {id_track}, nozzles {nearest_valve_idxes_cur} start send_signal------', show=False)
            for valve_idx in nearest_valve_idxes_cur:
                cfg.nearest_valve_idxes_in_process.append((valve_idx, max_time_to_close_valve, time_offset, now))
                if len(nearest_valve_idxes_cur) > 0:
                    self.send_signal([valve_idx], valve_signal_idx, time_offset, valve_opening_duration=valve_opening_duration)
        
    def get_detection_bbox(self, results_used):
        if results_used is None:
            detected_boxes = None
        elif type(results_used) is list:
            detected_boxes = results_used
        else:
            detected_boxes = results_used.boxes.data.cpu().numpy().copy()
        return detected_boxes
    
    def check_and_add_valve_idxes_into_queue(self, nearest_valve_idxes, time_offset, id_track, valve_opening_duration):
        global valve_signal_idx
        # remove and nearest_valve_idxes != cfg.nearest_valve_idxes_in_process due to time_offset
        need_open_valve = len(nearest_valve_idxes) > 0
        if need_open_valve:
            nearest_valve_idxes_cur = nearest_valve_idxes.copy()
            self.add_valve_idxes_into_queue(nearest_valve_idxes_cur, valve_signal_idx, time_offset, id_track, valve_opening_duration)
            valve_signal_idx += 1
        return valve_signal_idx
        
    def save_track_future_info_to_schedule_table(self, track_future_info):
        for id_track, info in track_future_info.copy().items():
            # Ensure it's a dict, not a list
            if id_track not in cfg.track_schedule_table or id_track < 0:
                cfg.track_schedule_table[id_track] = {}
                cfg.track_schedule_table[id_track]['time_needed'] = []
                cfg.track_schedule_table[id_track]['trigger_timestamp'] = []
                cfg.track_schedule_table[id_track]['nozzles'] = []
                cfg.track_schedule_table[id_track]['current_timestamp'] = []
                cfg.track_schedule_table[id_track]['move_distance'] = []
                cfg.track_schedule_table[id_track]['ave_pixel_speed'] = []
                cfg.track_schedule_table[id_track]['max_x_size'] = 0
                cfg.track_schedule_table[id_track]['triggered'] = False
                cfg.track_schedule_table[id_track]['triggering_plot'] = False
                cfg.triggering_plot_frame_num_t = max(3, int(cfg.global_detector.main_loop_fps//cfg.triggering_plot_frame_num_t_divider))
                cfg.track_schedule_table[id_track]['triggering_plot_frame_num'] = cfg.triggering_plot_frame_num_t
                cfg.track_schedule_table[id_track]['nozzle_time_offsets'] = {}
                cfg.track_schedule_table[id_track]['future_position'] = []
            cfg.track_schedule_table[id_track]['time_needed'].append(info['time_needed'])
            cfg.track_schedule_table[id_track]['trigger_timestamp'].append(info['trigger_timestamp'])
            cfg.track_schedule_table[id_track]['nozzles'].append(info['nozzles'])
            cfg.track_schedule_table[id_track]['current_timestamp'].append(info['current_timestamp'])
            cfg.track_schedule_table[id_track]['move_distance'].append(info['move_distance'])
            cfg.track_schedule_table[id_track]['ave_pixel_speed'].append(info['ave_pixel_speed'])
            cfg.track_schedule_table[id_track]['future_position'].append(info['future_position'])
            if info['x_size'] is not None:
                cfg.track_schedule_table[id_track]['max_x_size'] = max(cfg.track_schedule_table[id_track]['max_x_size'], info['x_size'])

        if len(cfg.track_schedule_table) > cfg.track_trace_max_num:
            track_history_tmp = list(cfg.track_schedule_table.keys())
            track_history_tmp_sorted = sorted(track_history_tmp)
            for i_item, track_id in enumerate(track_history_tmp_sorted):
                if track_id < 0:
                    continue
                if i_item >= int(cfg.track_trace_max_num//2):
                    break
                if track_id in cfg.track_schedule_table:
                    del cfg.track_schedule_table[track_id]

        for id_track in cfg.track_schedule_table:
            if len(cfg.track_schedule_table[id_track]['time_needed']) > cfg.track_schedule_table_each_trace_max_len:
                del_len = int(cfg.track_schedule_table_each_trace_max_len / 2)
                del cfg.track_schedule_table[id_track]['time_needed'][:del_len]
                del cfg.track_schedule_table[id_track]['trigger_timestamp'][:del_len]
                del cfg.track_schedule_table[id_track]['nozzles'][:del_len]
                del cfg.track_schedule_table[id_track]['current_timestamp'][:del_len]
                del cfg.track_schedule_table[id_track]['move_distance'][:del_len]
                del cfg.track_schedule_table[id_track]['ave_pixel_speed'][:del_len]
                del cfg.track_schedule_table[id_track]['future_position'][:del_len]

    def calc_valve_opening_duration_based_on_bbox_size(self, id_track, current_physical_speed):
        if len(cfg.track_schedule_table[id_track]['future_position'])  == 0:
            return cfg.valve_opening_duration 
        future_position = cfg.track_schedule_table[id_track]['future_position'][-1]
        future_position_x, future_position_y, bbox_w, bbox_h = future_position
        bbox_size = bbox_w

        max_x_size = cfg.track_schedule_table[id_track]['max_x_size']

        valve_opening_duration = cfg.valve_opening_duration
        if bbox_size is not None and bbox_size > 0:
            compensation_coeff = cfg.height_compensation_coeff
            height_factor = max(0.8, 1.0 - (max_x_size * compensation_coeff))
            plant_size = bbox_size * cfg.pixel_to_physical_space_resized * height_factor

            valve_opening_duration = plant_size / current_physical_speed
            valve_opening_duration *= cfg.valve_opening_duration_factor
            valve_opening_duration = min(cfg.valve_opening_duration_max, valve_opening_duration)
            valve_opening_duration = max(cfg.valve_opening_duration_min, valve_opening_duration)
        valve_opening_duration = round(valve_opening_duration, 4)
        return valve_opening_duration

    def add_valve_opening_duration_for_large_bbox(self, id_track, valve_opening_duration):
        max_x_size = cfg.track_schedule_table[id_track]['max_x_size']
        if max_x_size >= cfg.large_bbox_max_x_size_t_2:
            if valve_opening_duration is not None:
                large_bbox_valve_long_opening_duration_t_2 = cfg.large_bbox_add_physical_distance_for_duration_2 / cfg.current_physical_speed
                large_bbox_valve_long_opening_duration_t_2 = round(large_bbox_valve_long_opening_duration_t_2, 4)
                valve_opening_duration += large_bbox_valve_long_opening_duration_t_2
        elif max_x_size >= cfg.large_bbox_max_x_size_t_1:
            if valve_opening_duration is not None:
                large_bbox_valve_long_opening_duration_t_1 = cfg.large_bbox_add_physical_distance_for_duration_1 / cfg.current_physical_speed
                large_bbox_valve_long_opening_duration_t_1 = round(large_bbox_valve_long_opening_duration_t_1, 4)
                valve_opening_duration += large_bbox_valve_long_opening_duration_t_1
        return valve_opening_duration
    
    def calc_additional_distance(self, max_x_size):
        compensation_coeff = cfg.height_compensation_coeff
        height_factor = max(0.8, 1.0 - (max_x_size * compensation_coeff))
        actual_pixel_to_physical_space_resized = cfg.pixel_to_physical_space_resized * height_factor
        plant_height = max_x_size * actual_pixel_to_physical_space_resized

        additional_distance = plant_height/cfg.camera_height * cfg.camera_to_sprayed_ground_distance
        additional_distance *= cfg.additional_distance_factor
        #     if max_x_size >= cfg.large_bbox_max_x_size_t_2:
        #         physical_distance += cfg.large_bbox_add_physical_distance_2
        #     elif max_x_size >= cfg.large_bbox_max_x_size_t_1:
        #         physical_distance += cfg.large_bbox_add_physical_distance_1
        return additional_distance

    def calc_time_offset_based_on_valve_pos(self, id_track, nozzle_idx, additional_distance=0, physical_speed=None):
        max_x_size = cfg.track_schedule_table[id_track]['max_x_size']
        nozzle_idx = max(cfg.valid_valve_range[0], nozzle_idx)
        nozzle_idx = min(cfg.valid_valve_range[-1], nozzle_idx)
        if cfg.nozzle_distance_to_view is not None and nozzle_idx in cfg.nozzle_distance_to_view:
            if physical_speed is None or physical_speed <= 0:
                assert False, f"Invalid physical_speed: {physical_speed}"

            physical_distance = cfg.nozzle_distance_to_view[nozzle_idx]

            if cfg.large_bbox_add_distance_before_nozzle_for_parallax:
                physical_distance += additional_distance 

            time_offset = round(physical_distance / physical_speed, 4)
            return time_offset
        else:
            cfg.log_xalg('nozzle_idx out of range, nozzle_idx:', nozzle_idx)
            assert False
        return time_offset

    def calc_physical_speed_from_pixelspace(self):
        pixel_speeds = []

        for id_track, info in cfg.track_schedule_table.items():
            if id_track < 0:
                continue
            if cfg.track_schedule_table[id_track]['triggered'] is True:
                continue
            trigger_timestamp = info['trigger_timestamp'][-1]
            move_distance = info['move_distance'][-1]
            if trigger_timestamp is None:
                continue
            if cfg.move_direction_required < 0:
                if move_distance > cfg.move_direction_required:
                    continue
            else:
                if move_distance < cfg.move_direction_required:
                    continue
            ave_pixel_speed = info['ave_pixel_speed'][-1]
            if ave_pixel_speed is not None and ave_pixel_speed > 0:
                pixel_speeds.append(ave_pixel_speed)
        if len(pixel_speeds) == 0:
            return cfg.current_physical_speed
        pixel_speeds = np.array(pixel_speeds)
        if len(pixel_speeds) == 2:
            ave_pixel_speed = max(pixel_speeds)
        else:
            ave_pixel_speed = np.median(pixel_speeds)
        physical_speed = ave_pixel_speed * cfg.pixel_to_physical_space_resized
        physical_speed = round(physical_speed, 2)
        return pixel_speeds, physical_speed

    def triger_future_open_nozzles(self):
        physical_speed = cfg.current_physical_speed
        if cfg.use_average_speed_from_all_weeds:
            _, physical_speed = self.calc_physical_speed_from_pixelspace()

        now = datetime.now()
        trigger_events = []

        for id_track, track in list(cfg.track_schedule_table.items()):
            # Cancel any pending schedule for a weed that entered a crop safety zone.
            if id_track in cfg.global_detector.unsafe_track_ids:
                track["triggered"] = True
                continue

            if id_track < 0 or track['triggered']:
                continue

            trigger_timestamp = track['trigger_timestamp'][-1]
            if trigger_timestamp is None:
                continue

            move_distance = track['move_distance'][-1]

            if cfg.move_direction_required < 0:
                if move_distance > cfg.move_direction_required:
                    continue
            else:
                if move_distance < cfg.move_direction_required:
                    continue

            if (trigger_timestamp - now).total_seconds() >= 0:
                continue

            if cfg.use_speed_from_each_weed:
                physical_speed = track['ave_pixel_speed'][-1] * cfg.pixel_to_physical_space_resized

            # raw_pixel_speed = track['ave_pixel_speed'][-1]

            # # Estimate height using bbox average size (width and height)
            # if len(track['future_position']) > 0 and track['future_position'][-1]:
            #     future_pos = track['future_position'][-1]
            #     bbox_w = future_pos[2]
            #     bbox_h = future_pos[3]
            #     bbox_avg_size = (bbox_w + bbox_h) / 2.0
            # else:
            #     # Fallback to max_x_size if future_position is empty
            #     bbox_avg_size = track.get('max_x_size', 0)

            # # Parallax correction factor: Taller plants (larger bboxes) appear to move faster in pixels.
            # # We apply a compensation factor to scale down the pixel speed for taller objects.
            # # You will need to add `height_compensation_coeff = 0.001` (or similar) to your config.py.
            # compensation_coeff = cfg.height_compensation_coeff
            
            # # Prevent the factor from dropping below a threshold (e.g., 50%) to avoid stopping the spray
            # height_factor = max(0.5, 1.0 - (bbox_avg_size * compensation_coeff))
            
            # adjusted_pixel_speed = raw_pixel_speed * height_factor
            # physical_speed = adjusted_pixel_speed * cfg.pixel_to_physical_space_resized

            track['triggered'] = True

            if cfg.open_duration_based_on_bbox_size:
                valve_opening_duration = self.calc_valve_opening_duration_based_on_bbox_size(id_track, physical_speed)
                cfg.log_xalg(f'id_track: {id_track}, valve_opening_duration: {valve_opening_duration}', show=False)
            else:
                valve_opening_duration = cfg.valve_opening_duration
                if cfg.large_bbox_valve_long_opening_duration:
                    valve_opening_duration = self.add_valve_opening_duration_for_large_bbox(
                        id_track,
                        valve_opening_duration
                    )

            nozzles = track['nozzles'][-1]

            if not nozzles:
                for nozzle_history in reversed(track['nozzles'][-5:]):
                    if nozzle_history:
                        nozzles = nozzle_history
                        track['nozzles'][-1] = nozzles
                        break

            if not nozzles:
                continue

            nozzle_offsets = {}

            if cfg.use_time_offset:
                cfg.log_xalg(f'physical_speed: {physical_speed}, id_track: {id_track}', show=False)
                additional_distance = self.calc_additional_distance(track['max_x_size'])
                show_info = True
                for nozzle_idx in nozzles:
                    time_offset = self.calc_time_offset_based_on_valve_pos(
                        id_track,
                        nozzle_idx,
                        additional_distance,
                        physical_speed
                    )
                    if show_info:
                        cfg.log_xalg(f'nozzle_idx: {nozzle_idx}, time_offset: {time_offset}', show=False)
                        show_info=False

                    old_offset = nozzle_offsets.get(nozzle_idx)
                    if old_offset is None or time_offset < old_offset:
                        nozzle_offsets[nozzle_idx] = time_offset

                    track['nozzle_time_offsets'][nozzle_idx] = time_offset
            else:
                for nozzle_idx in nozzles:
                    nozzle_offsets[nozzle_idx] = 0

            grouped = defaultdict(list)

            for nozzle_idx, time_offset in nozzle_offsets.items():
                grouped[round(time_offset, 3)].append(nozzle_idx)

            for time_offset, nozzle_group in grouped.items():
                trigger_events.append(
                    (
                        nozzle_group,
                        time_offset,
                        id_track,
                        valve_opening_duration
                    )
                )

        for nozzle_group, time_offset, id_track, valve_opening_duration in trigger_events:
            self.check_and_add_valve_idxes_into_queue(
                nozzle_group,
                time_offset,
                id_track,
                valve_opening_duration
            )
            pass

    def triger_future_open_nozzles_v0(self):
        triggered_nozzles_with_time_offset = []
        physical_speed = -1
        if cfg.use_physical_speed_from_wheel:
            physical_speed = cfg.current_physical_speed
        else:
            if cfg.use_average_speed_from_all_weeds:
                pixel_speeds, physical_speed = self.calc_physical_speed_from_pixelspace()
        # print('physical_speed:', physical_speed)
        cfg.track_schedule_table_copy = cfg.track_schedule_table.copy()
        for id_track, info in cfg.track_schedule_table_copy.items():
            if id_track < 0:
                continue
            if cfg.track_schedule_table[id_track]['triggered'] is True:
                continue
            trigger_timestamp = info['trigger_timestamp'][-1]
            move_distance = info['move_distance'][-1]
            if trigger_timestamp is None:
                continue
            if cfg.move_direction_required < 0:
                if move_distance > cfg.move_direction_required:
                    continue
            else:
                if move_distance < cfg.move_direction_required:
                    continue
            if cfg.use_speed_from_each_weed:
                physical_speed = info['ave_pixel_speed'][-1] * cfg.pixel_to_physical_space_resized
            assert physical_speed >=0
            nozzles = info['nozzles'][-1]
            if len(nozzles) == 0:
                track_len = len(info['nozzles'])
                for i_bbox in range(track_len-1, track_len-5, -1):
                    if i_bbox < 0:
                        break
                    nozzle_info = info['nozzles'][i_bbox]
                    if len(nozzle_info) > 0:
                        nozzles = nozzle_info
                        cfg.track_schedule_table[id_track]['nozzles'][-1] = nozzles
                        break

            current_timestamp = datetime.now()
            time_gap = (trigger_timestamp-current_timestamp).total_seconds()
            if time_gap < 0:
                cfg.track_schedule_table[id_track]['triggered'] = True
                valve_opening_duration = cfg.valve_opening_duration
                if cfg.open_duration_based_on_bbox_size:
                    valve_opening_duration = self.calc_valve_opening_duration_based_on_bbox_size(id_track)
                else:
                    if cfg.large_bbox_valve_long_opening_duration:
                        valve_opening_duration = self.add_valve_opening_duration_for_large_bbox(id_track, valve_opening_duration)
                if cfg.use_time_offset:
                    # time_offset_front = cfg.delay_time_run_1_to_12
                    # time_offset_back = cfg.delay_time_run_13_to_24
                    """
                    if the detection fps much less than the camera fps, the time offset maybe not accurate
                    """
                    for nozzle_idx in nozzles:
                        time_offset = self.calc_time_offset_based_on_valve_pos(id_track, nozzle_idx, physical_speed=physical_speed)
                        triggered_nozzles_with_time_offset.append(([nozzle_idx], time_offset))

                        cfg.track_schedule_table[id_track]['nozzle_time_offsets'][nozzle_idx] = time_offset
                else:
                    time_offset = 0
                    triggered_nozzles_with_time_offset.append((nozzles, time_offset))
                # if id_track in cfg.global_detector.track_history:
                #     del cfg.global_detector.track_history[id_track]
                # del cfg.track_schedule_table[id_track]

                # For each nozzle idx, select the smallest time offset
                nozzle_to_time_offset = {}
                for nozzle_list, time_offset in triggered_nozzles_with_time_offset:
                    for nozzle in nozzle_list:
                        if (nozzle not in nozzle_to_time_offset) or (time_offset < nozzle_to_time_offset[nozzle]):
                            nozzle_to_time_offset[nozzle] = time_offset
                # Group by time_offset (within 1e-6 for float equality)
                time_offset_to_nozzles = {}
                for nozzle, time_offset in nozzle_to_time_offset.items():
                    key = round(time_offset, 3)
                    if key not in time_offset_to_nozzles:
                        time_offset_to_nozzles[key] = []
                    time_offset_to_nozzles[key].append(nozzle)
                final_triggered = [(nozzles, time_offset) for time_offset, nozzles in time_offset_to_nozzles.items()]
                for nozzles, time_offset in final_triggered:
                    # self.check_and_add_valve_idxes_into_queue(nozzles, time_offset, id_track)
                    t_track = threading.Thread(target=self.check_and_add_valve_idxes_into_queue, args=(nozzles, time_offset, id_track, valve_opening_duration), daemon=True)
                    t_track.start()

    def check_overdue_valve_idx(self):
        now = datetime.now()
        for valve_info in cfg.nearest_valve_idxes_in_process.copy():
            assert isinstance(valve_info, tuple)
            valve_idx, max_time_to_close_valve = valve_info[:2]
            time_gap = (max_time_to_close_valve - now).total_seconds()
            if time_gap < 0:
                self.send_signal([valve_idx], valve_signal_idx, time_offset=0, only_close_valve=True)
                if valve_info in cfg.nearest_valve_idxes_in_process:
                    cfg.nearest_valve_idxes_in_process.remove(valve_info)

    def add_mannual_valve_idxes_into_queue(self, valve_signal_idx):
        need_open_valve_by_mannual = len(cfg.nearest_valve_idxes_manual_control) > 0 
        if need_open_valve_by_mannual:
            for valve_idx in cfg.nearest_valve_idxes_manual_control:
                already_has_the_valve = False
                for valve_info in cfg.nearest_valve_idxes_in_process.copy():
                    valve_info_idx = valve_info[0]
                    if valve_idx == valve_info_idx:
                        already_has_the_valve = True
                        break
                if not already_has_the_valve:
                    self.add_valve_idxes_into_queue([valve_idx], valve_signal_idx=valve_signal_idx, time_offset=0.1, id_track='manual', valve_opening_duration=cfg.valve_opening_duration)
                    cfg.nearest_valve_idxes_manual_control = []
                    valve_signal_idx += 1
        return valve_signal_idx

    def triger_bbox_under_nozzle(self, detected_boxes=None, track_future_info=None):
        physical_speed = cfg.current_physical_speed

        nearest_valve_idxes = []
        current_bboxes = []
        if detected_boxes is not None:
            for detected_box in detected_boxes:
                current_bbox = detected_box
                current_bboxes.append(current_bbox)
        elif track_future_info is not None:
            for i_track in track_future_info:
                current_bbox = track_future_info[i_track]['current_bbox']
                current_bboxes.append(current_bbox)
        if len(current_bboxes) > 0:
            nearest_valve_idxes = self.check_box_location(current_bboxes)

        if len(nearest_valve_idxes) > 0:
            id_track = self.instant_id_track
        #     # id_track_prev = id_track + 1
        #     for nearest_valve_idx in nearest_valve_idxes.copy():
        #         for id_track_prev in cfg.track_schedule_table:
        #             # if id_track_prev >= 0:
        #             #     continue
        #             if nearest_valve_idx in cfg.track_schedule_table[id_track_prev]['nozzles'][-1]:
        #                 nearest_valve_idxes.remove(nearest_valve_idx)
        #                 break
        #             pass

        if len(nearest_valve_idxes) > 0:
            track_future_info = {}
            track_future_info[id_track] = {}
            track_future_info[id_track]['ave_pixel_speed'] = -1
            track_future_info[id_track]['move_distance'] = cfg.move_direction_required
            track_future_info[id_track]['current_timestamp'] = -1
            track_future_info[id_track]['trigger_timestamp'] = -1
            track_future_info[id_track]['time_needed'] = 0
            track_future_info[id_track]['nozzles'] = nearest_valve_idxes
            track_future_info[id_track]['x_size'] = None

            track_future_info[id_track]['future_position'] = (0, 0, 0, 0)

            self.save_track_future_info_to_schedule_table(track_future_info)
            cfg.track_schedule_table[id_track]['triggered'] = True
            for nozzle_idx in nearest_valve_idxes:
                valve_opening_duration = cfg.valve_opening_duration
                time_offset = self.calc_time_offset_based_on_valve_pos(id_track, nozzle_idx, physical_speed=physical_speed)
                cfg.track_schedule_table[id_track]['nozzle_time_offsets'][nozzle_idx] = time_offset
                # self.check_and_add_valve_idxes_into_queue([nozzle_idx], time_offset=time_offset, id_track=id_track)
                t_under_nozzle = threading.Thread(target=self.check_and_add_valve_idxes_into_queue, args=([nozzle_idx], time_offset, id_track, valve_opening_duration), daemon=True)
                t_under_nozzle.start()

            self.instant_id_track -= 1
            if self.instant_id_track <= -cfg.allowed_instant_id_track_num:
                self.instant_id_track = -1

    def run_sync(self):
        """
        Directly send through seriel port cannot work in Qt framwork? 
        To bypass it, using valve_control/valve_serial_send_port2.py
        """
        #memory increasing?
        try:
            global valve_signal_idx
            if True:
                if not hasattr(self, '_fps_frame_count'):
                    self._fps_frame_count = 0
                    self._fps_last_time = time()
                    self._fps_value = 0

                self.global_idx += 1

                valve_signal_idx = self.add_mannual_valve_idxes_into_queue(valve_signal_idx)

                if not cfg.use_track:
                    try:
                        frame_result_info = cfg.frame_result_for_signal_queue.get(timeout=0.01)  # wait briefly for new frame
                    except Exception as e:
                        return
                    results_used = frame_result_info['results']
                    detected_boxes = self.get_detection_bbox(results_used)
                    if detected_boxes is None:
                        sleep(0.001)
                        return 
                    self.triger_bbox_under_nozzle(detected_boxes=detected_boxes)
                else:
                    # test it solely
                    pass
                    try:
                        track_future_info = cfg.track_future_info_queue.get(timeout=0.01)
                    except Exception as e:
                        track_future_info = None
                    if track_future_info is not None:
                        self.save_track_future_info_to_schedule_table(track_future_info)
                        self.triger_future_open_nozzles()

                        # self.check_overdue_valve_idx()
                        # test it solely
                        # self.triger_bbox_under_nozzle(track_future_info=track_future_info)

                self._fps_frame_count += 1
                now = time()
                if now - self._fps_last_time >= 1.0:
                    self._fps_value = np.round(self._fps_frame_count / (now - self._fps_last_time), 2)
                    self._fps_last_time = now
                    self._fps_frame_count = 0
                self.send_signal_fps = self._fps_value
        except Exception as e:
            cfg.log_xalg(traceback.format_exc())
            cfg.log_xalg(e)


    def run(self):
        try:
            while self.need_exit_async_run is False:
                self.run_sync()
                pass
        except Exception as e:
            cfg.log_xalg(traceback.format_exc())
            cfg.log_xalg(e)
        # self.finished.emit()

# class WorkerSendSignalSpeed(QObject):
class WorkerSendSignalSpeed():
    # finished = pyqtSignal()
    # progress = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        # self.signal_dir = '/home/agfoodsensinglab/Desktop/Arduino/signal.txt'
        self.signal_read_path = 'sensing/signal_read.txt'
        self.signal_write_path = 'sensing/signal_write.txt'
        self.signal_new_sample_path = 'sensing/signal_new_sample.txt'
        with open(self.signal_new_sample_path, 'w') as f:
            f.write('')
        with open(self.signal_read_path, 'w') as f:
            f.write('')
        with open(self.signal_write_path, 'w') as f:
            f.write('0')

    def run(self):
        pass

