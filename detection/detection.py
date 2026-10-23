import os
import sys

import torchvision
if __name__ == '__main__':
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QVBoxLayout, QSlider, QLabel, QFileDialog, QMainWindow, QButtonGroup, QApplication
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QCoreApplication, QObject, QRunnable, QThread, QThreadPool, pyqtSignal, QTimer, QRect
# from  PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
# from  PyQt5.QtMultimediaWidgets import QVideoWidget
import torch
import json
import threading
from typing import Optional
from time import time, sleep
from datetime import datetime, timedelta
import numpy as np
from queue import SimpleQueue, Queue
from itertools import count
from loguru import logger
import random
from collections import defaultdict
import traceback
from ultralytics.engine.results import Boxes  # or wherever Boxes is defined


if __name__ != '__main__':
    import config as cfg
    from detection.plot import Plotter

g_frame_raw_lock = threading.Lock()

g_pause_event = threading.Event()
g_pause_event_by_detection = threading.Event()

g_resize_frame_for_display_raw = None
g_resize_frame_for_display_raw_frame_idx = 0
g_frame_raw = None

g_frame_info_queue = Queue(maxsize=1)
g_frame_info_for_save_queue = Queue(maxsize=1)

g_first_detection = True
g_use_detection = True

def get_frame_from_camera():
    frame = None
    frame_timestamp = None
    frame_idx = 0

    if cfg.async_camera_stream:
        assert cfg.camera_vendor == 'vimba' or cfg.camera_vendor == 'JAI', 'only available for vimba and JAI'
        # frame = cfg.camera_latest_frame
        # frame_timestamp = cfg.camera_latest_frame_timestamp
        # frame_idx = cfg.camera_latest_frame_idx

        frame_info = cfg.global_camera_handler.frame_queue.get()
        frame, frame_timestamp, frame_idx = frame_info
        if type(frame) is list: # include infrared channel
            if cfg.has_infrared_channel:
                frame = np.concatenate([frame[0], frame[1][:,:,None]], -1)
            else:
                frame = frame[0]
        # if not cfg.global_camera_handler.frame_queue.empty():
        #     pass
        #     # frame_info = cfg.global_camera_handler.frame_queue.get_nowait()
        # else:
        #     print('cfg.global_camera_handler.frame_queue.empty()')
    else:
        if cfg.camera_vendor == 'vimba':
            frame = cfg.camera_latest_frame
            frame_timestamp = cfg.camera_latest_frame_timestamp
            frame_idx = cfg.camera_latest_frame_idx
        elif cfg.camera_vendor == 'opencv':
            frame_read_status, frame_info = cfg.global_camera_handler.capture()
            frame = frame_info['img']
            frame_ann = frame_info['ann']
            frame_status = frame_info['status']
            frame_timestamp = frame_info['timestamp']
            frame_idx = frame_info['frame_idx']
        else:
            assert False

    return frame, frame_timestamp, frame_idx


def resize_origin_to_detect_im(frame):
    if cfg.camera_vendor == 'opencv':
        is_bgr = True
    elif cfg.camera_vendor == 'vimba':
        is_bgr = True
    elif cfg.camera_vendor == 'JAI':
        is_bgr = True
    else:
        raise ValueError("Unsupported camera vendor: {}".format(cfg.camera_vendor))
    frame_resize = cv2.resize(frame, cfg.detect_size_no_pad)

    if cfg. detect_size_with_pad != cfg.detect_size_no_pad:
        if cfg.is_pad_height:
            frame_resize = cv2.copyMakeBorder(
                frame_resize,
                top=cfg.detect_size_pad_half,
                bottom=cfg.detect_size_pad_half,
                left=0,
                right=0,
                borderType=cv2.BORDER_CONSTANT,
                value=(128, 128, 128) 
            )
        else:
            frame_resize = cv2.copyMakeBorder(
                frame_resize,
                top=0,
                bottom=0,
                left=cfg.detect_size_pad_half,
                right=cfg.detect_size_pad_half,
                borderType=cv2.BORDER_CONSTANT,
                value=(128, 128, 128)  
            )
    if is_bgr:
        frame_bgr_resize = frame_resize
        if frame_resize.shape[-1] ==4:
            frame_rgb_resize = cv2.cvtColor(frame_resize, cv2.COLOR_BGRA2RGBA)
        else:
            frame_rgb_resize = cv2.cvtColor(frame_resize, cv2.COLOR_BGR2RGB)
    else:
        frame_rgb_resize = frame_resize
        if frame_resize.shape[-1] ==4:
            frame_bgr_resize = cv2.cvtColor(frame_resize, cv2.COLOR_RGBA2BGRA)
        else:
            frame_bgr_resize = cv2.cvtColor(frame_resize, cv2.COLOR_RGB2BGR)
    return frame_bgr_resize, frame_rgb_resize


def retrive_frame_from_camera():
    frame, frame_timestamp, frame_idx = get_frame_from_camera()
    if frame is None:
        return None, None, None, None, None
    # if frame is None:
    #     if self.frame_last is None:
    #         return
    #     else:
    #         frame = self.frame_last
    #         frame_timestamp = self.frame_last_timestamp
    # else:
    #     self.frame_last = frame
    #     self.frame_last_timestamp = frame_timestamp
    frame_copy = frame.copy()
    frame_bgr_resize, frame_rgb_resize = resize_origin_to_detect_im(frame_copy)
    return frame_copy, frame_timestamp, frame_idx, frame_bgr_resize, frame_rgb_resize


class LettuceTracker:
    def __init__(self, max_distance_px=80.0, max_age=30, move_direction=-1):
        self.next_id = 1
        # track_id -> {'box': [x1, y1, x2, y2], 'age': int, 'velocity_px': float (optional)}
        self.tracks = {}  
        self.max_distance_px = max_distance_px
        self.max_age = max_age
        self.move_direction = move_direction
        self.last_timestamp = None 
        self.global_velocity_px = None 

    def update(self, bboxes, current_timestamp, pixel_to_physical_space_resized, fallback_physical_speed):
        num_dets = len(bboxes)

        # 1. Calculate precise elapsed time
        time_delta = 0
        if self.last_timestamp is not None and current_timestamp is not None:
            time_delta = (current_timestamp - self.last_timestamp).total_seconds()
            # Cap delta at 0.3 seconds to prevent massive box jumps if the camera stutters heavily
            time_delta = min(max(time_delta, 0.001), 0.3) 
        self.last_timestamp = current_timestamp

        # 2. Determine Dynamic Pixel Velocity from active tracks
        move_px = 0
        if time_delta > 0:
            valid_velocities = [t['velocity_px'] for t in self.tracks.values() if t.get('velocity_px') is not None]
            
            if len(valid_velocities) > 0:
                # Use the median real-time velocity of tracked lettuces (ignores tracking errors/outliers)
                self.global_velocity_px = np.median(valid_velocities)
            elif self.global_velocity_px is None and pixel_to_physical_space_resized > 0:
                # Fallback to the physical config speed only if no tracks have been established yet
                self.global_velocity_px = fallback_physical_speed / pixel_to_physical_space_resized

            if self.global_velocity_px is not None:
                move_px = self.global_velocity_px * time_delta

        # 3. Handle frames with no detections
        if num_dets == 0:
            dead_ids = [tid for tid, t in self.tracks.items() if t['age'] + 1 > self.max_age]
            for tid in dead_ids:
                del self.tracks[tid]
            # Accumulate virtual shift so boxes keep moving in memory
            for tid in self.tracks:
                self.tracks[tid]['age'] += 1
                shift_x = move_px * self.move_direction
                self.tracks[tid]['box'][0] += shift_x
                self.tracks[tid]['box'][2] += shift_x
            return []

        track_ids = list(self.tracks.keys())
        
        # 4. Shift tracks into a temporary array strictly for nearest-neighbor prediction
        old_boxes = np.array([self.tracks[tid]['box'] for tid in track_ids], dtype=np.float32)
        shifted_boxes = old_boxes.copy()
        if len(shifted_boxes) > 0:
            shift_x = move_px * self.move_direction
            shifted_boxes[:, 0] += shift_x
            shifted_boxes[:, 2] += shift_x

        new_boxes = bboxes[:, :4].astype(np.float32)

        old_cx = (shifted_boxes[:, 0] + shifted_boxes[:, 2]) * 0.5 if len(shifted_boxes) > 0 else np.array([])
        old_cy = (shifted_boxes[:, 1] + shifted_boxes[:, 3]) * 0.5 if len(shifted_boxes) > 0 else np.array([])
        new_cx = (new_boxes[:, 0] + new_boxes[:, 2]) * 0.5
        new_cy = (new_boxes[:, 1] + new_boxes[:, 3]) * 0.5

        dist_matrix = np.zeros((len(shifted_boxes), len(new_boxes)))
        if len(shifted_boxes) > 0:
            vx = new_cx[None, :] - old_cx[:, None]
            vy = new_cy[None, :] - old_cy[:, None]
            dist_matrix = np.hypot(vx, vy)

        assigned_ids = [-1] * num_dets
        matched_tracks = set()

        if dist_matrix.size > 0:
            flat_indices = np.argsort(dist_matrix, axis=None)
            for flat_idx in flat_indices:
                r, c = np.unravel_index(flat_idx, dist_matrix.shape)
                if dist_matrix[r, c] > self.max_distance_px:
                    break
                if r in matched_tracks or assigned_ids[c] != -1:
                    continue

                tid = track_ids[r]
                assigned_ids[c] = tid
                matched_tracks.add(r)
                
                # 5. Calculate real velocity based on the raw match (before updating the anchor)
                if time_delta > 0:
                    real_old_cx = (old_boxes[r][0] + old_boxes[r][2]) / 2.0
                    real_vel_px = abs(new_cx[c] - real_old_cx) / time_delta
                    
                    # Sanity gate: Cap speed spikes (e.g., if a box rapidly flickers to a different crop)
                    max_allowed_vel = (fallback_physical_speed / pixel_to_physical_space_resized) * 3
                    if real_vel_px < max_allowed_vel:
                        self.tracks[tid]['velocity_px'] = real_vel_px

                self.tracks[tid]['box'] = new_boxes[c].copy()
                self.tracks[tid]['age'] = 0

        # 6. Unassigned detections become new tracks
        for c in range(num_dets):
            if assigned_ids[c] == -1:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {'box': new_boxes[c].copy(), 'age': 0, 'velocity_px': None}
                assigned_ids[c] = tid

        # 7. Age unmatched tracks and apply the virtual shift so they don't fall behind
        for r, tid in enumerate(track_ids):
            if r not in matched_tracks:
                self.tracks[tid]['age'] += 1
                shift_x = move_px * self.move_direction
                self.tracks[tid]['box'][0] += shift_x
                self.tracks[tid]['box'][2] += shift_x
                if self.tracks[tid]['age'] > self.max_age:
                    del self.tracks[tid]

        return assigned_ids

    
class FastSimpleTracker:
    def __init__(self, max_distance_px=80.0, max_age=3, move_direction=-1, max_angle_degree=10.0, jitter_allowance_px=10.0):
        """
        Ultra-fast vectorized tracker for linear platform motion with strict angle constraints.
        
        :param max_distance_px: Max allowed pixel distance for matching between frames.
        :param max_age: Number of consecutive missing frames before dropping a track.
        :param move_direction: -1 if weeds move Right-to-Left, +1 if Left-to-Right.
        :param max_angle_degree: Max allowed movement angle deviation from the horizontal axis (prevents vertical jumping).
        :param jitter_allowance_px: Radius inside which angle constraints are ignored to allow sub-pixel box vibration.
        """
        self.next_id = 1
        self.tracks = {}  # track_id -> {'box': [x1, y1, x2, y2], 'age': int}
        self.max_distance_px = max_distance_px
        self.max_age = max_age
        self.move_direction = move_direction
        self.max_angle_degree = max_angle_degree
        self.jitter_allowance_px = jitter_allowance_px

    def update(self, bboxes):
        """
        :param bboxes: np.ndarray of shape (N, 4) or (N, 6) -> [x1, y1, x2, y2, ...]
        :return: List of assigned track_ids of length N
        """
        num_dets = len(bboxes)
        if num_dets == 0:
            # Increment age for missing tracks and purge expired ones
            dead_ids = [tid for tid, t in self.tracks.items() if t['age'] + 1 > self.max_age]
            for tid in dead_ids:
                del self.tracks[tid]
            for tid in self.tracks:
                self.tracks[tid]['age'] += 1
            return []

        # Initialize if no active tracks exist
        if len(self.tracks) == 0:
            assigned_ids = []
            for box in bboxes:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {'box': box[:4].copy(), 'age': 0}
                assigned_ids.append(tid)
            return assigned_ids

        # Extract track IDs and centroids
        track_ids = list(self.tracks.keys())
        old_boxes = np.array([self.tracks[tid]['box'] for tid in track_ids], dtype=np.float32) # (M, 4)
        new_boxes = bboxes[:, :4].astype(np.float32) # (N, 4)

        old_cx = (old_boxes[:, 0] + old_boxes[:, 2]) * 0.5
        old_cy = (old_boxes[:, 1] + old_boxes[:, 3]) * 0.5
        new_cx = (new_boxes[:, 0] + new_boxes[:, 2]) * 0.5
        new_cy = (new_boxes[:, 1] + new_boxes[:, 3]) * 0.5

        # 1. Vectorized Kinematics (M x N)
        # Calculate velocity vectors (vx, vy)
        vx = new_cx[None, :] - old_cx[:, None]
        vy = new_cy[None, :] - old_cy[:, None]
        
        # Calculate scalar distance
        dist_matrix = np.hypot(vx, vy)
        
        # Calculate angle of movement in degrees (0 to 180)
        # 0 = moving perfectly right, 180 = moving perfectly left
        angles_deg = np.rad2deg(np.abs(np.arctan2(vy, vx)))

        # 2. Apply Kinematic Directional & Angle Mask
        if self.move_direction < 0: # Right-to-Left
            # Target angle is 180. Allowed range is (180 - max_angle) to 180
            valid_angle_mask = angles_deg >= (180.0 - self.max_angle_degree)
            invalid_mask = ~(valid_angle_mask | (dist_matrix < self.jitter_allowance_px))
            # Also explicitly forbid backward jumping beyond jitter
            invalid_mask |= (vx > self.jitter_allowance_px) 
        else: # Left-to-Right
            # Target angle is 0. Allowed range is 0 to max_angle
            valid_angle_mask = angles_deg <= self.max_angle_degree
            invalid_mask = ~(valid_angle_mask | (dist_matrix < self.jitter_allowance_px))
            # Explicitly forbid backward jumping beyond jitter
            invalid_mask |= (vx < -self.jitter_allowance_px)

        dist_matrix[invalid_mask] = 1e6

        # 3. Fast Greedy Matching (Sub-millisecond for N ~ 100)
        assigned_ids = [-1] * num_dets
        matched_tracks = set()

        if dist_matrix.size > 0:
            flat_indices = np.argsort(dist_matrix, axis=None)
            for flat_idx in flat_indices:
                r, c = np.unravel_index(flat_idx, dist_matrix.shape)
                
                # Stop if distance exceeds threshold
                if dist_matrix[r, c] > self.max_distance_px:
                    break
                if r in matched_tracks or assigned_ids[c] != -1:
                    continue

                tid = track_ids[r]
                assigned_ids[c] = tid
                matched_tracks.add(r)
                self.tracks[tid] = {'box': new_boxes[c], 'age': 0}

        # 4. Assign new IDs to unassigned detections
        for c in range(num_dets):
            if assigned_ids[c] == -1:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {'box': new_boxes[c], 'age': 0}
                assigned_ids[c] = tid

        # 5. Age and purge unmatched tracks
        for r, tid in enumerate(track_ids):
            if r not in matched_tracks:
                self.tracks[tid]['age'] += 1
                if self.tracks[tid]['age'] > self.max_age:
                    del self.tracks[tid]

        return assigned_ids
    
class WorkerCameraRetrieve():
    def __init__(self):
        self.skip_frame_num = 0
        pass

    def run(self):
        print('WorkerCameraRetrieve: start running')
        global g_resize_frame_for_display_raw, g_resize_frame_for_display_raw_frame_idx, g_frame_raw, g_frame_info_queue, g_first_detection
        try:
            camera_retrieve_loop_idx = 0
            while True:
                if cfg.global_camera_handler.camera_is_opened is False:
                    sleep(0.001)
                    continue

                frame, frame_timestamp, frame_idx, frame_bgr_resize, frame_rgb_resize = retrive_frame_from_camera()
                if frame is None:
                    sleep(0.001)
                    continue
                with g_frame_raw_lock:
                    g_frame_raw = frame
                    if cfg.detect_size_with_pad == cfg.display_size:
                        g_resize_frame_for_display_raw = frame_rgb_resize
                        # g_resize_frame_for_display_raw = frame_bgr_resize
                    else:
                        g_resize_frame_for_display_raw = cv2.resize(frame_rgb_resize, cfg.display_size)
                g_resize_frame_for_display_raw_frame_idx = frame_idx

                if not hasattr(self, '_fps_frame_count'):
                    self._fps_frame_count = 0
                    self._fps_last_time = time()
                    self._fps_value = 0
                    self.skip_frame_num = 0

                if cfg.global_detector is not None:
                    if cfg.global_window.detect_check.isChecked():
                        frame_info = {}
                        frame_info['frame_idx'] = frame_idx
                        # frame_info['frame'] = np.copy(frame_bgr_resize)  # Ensures thread safety
                        frame_info['frame'] = frame_bgr_resize.copy()  # Ensures thread safety
                        frame_info['timestamp'] = frame_timestamp
                        frame_info['frame_raw'] = frame
                        if 'frame' not in frame_info or 'timestamp' not in frame_info:
                            cfg.log_xalg('frame_info is empty')
                            self.skip_frame_num += 1
                            continue
                        try:
                            # Non-blocking put, discard old frame if queue is full
                            g_frame_info_queue.get_nowait()  # drop stale frame
                        except Exception as e:
                            pass
                        g_frame_info_queue.put_nowait(frame_info)

                        if cfg.show_step_by_step:
                            global g_pause_event
                            # if frame_idx < 10:
                            #     g_pause_event.set()
                            g_pause_event.wait()
                            g_pause_event.clear()

                        global g_pause_event_by_detection, g_use_detection
                        if g_first_detection:
                            if g_use_detection and cfg.camera_vendor == 'opencv':
                                g_pause_event_by_detection.wait()

                self._fps_frame_count += 1
                now = time()
                if now - self._fps_last_time >= 1.0:
                    self._fps_value = np.round(self._fps_frame_count / (now - self._fps_last_time), 2)
                    self._fps_last_time = now
                    self._fps_frame_count = 0
                    if self.skip_frame_num > 0:
                        cfg.log_xalg('WorkerCameraRetrieve thread -> self.skip_frame_num:', self.skip_frame_num)
                    self.skip_frame_num = 0
                    # cfg.log_xalg('cur_camera_retrieve frame rate:', cfg.global_detector.cur_camera_retrieve_fps)
                camera_retrieve_loop_idx += 1
                cfg.global_detector.cur_camera_retrieve_fps = self._fps_value
        except Exception as e:
            cfg.log_xalg(traceback.format_exc())
            cfg.log_xalg(e)


class WorkerDetection():
    def __init__(self):
        self.skip_frame_num = 0
        self.results_history_list = []
        pass

    def put_results_into_queues(self, results, frame_info):
        results_grade, results_track, timestamp = results
        frame_result_info = {}
        frame_result_info['timestamp'] = timestamp
        if results_track is not None:
            frame_result_info['results'] = results_track
        else:
            frame_result_info['results'] = results_grade
        try:
            cfg.frame_result_for_signal_queue.get_nowait()  
        except Exception as e:
            pass
        cfg.frame_result_for_signal_queue.put_nowait(frame_result_info)

        frame_result_info['results_grade'] = results_grade
        frame_result_info['results_track'] = results_track
        try:
            cfg.frame_result_for_track_queue.get_nowait()  
        except Exception as e:
            pass
        cfg.frame_result_for_track_queue.put_nowait(frame_result_info)

        try:
            cfg.frame_result_for_display_queue.get_nowait()  
        except Exception as e:
            pass
        cfg.frame_result_for_display_queue.put_nowait(frame_result_info)

        if cfg.save_newest_frame is False:
            global g_frame_info_for_save_queue
            frame_save_info = {}
            frame_save_info['frame'] = frame_info['frame']
            frame_save_info['results_grade'] = results_grade
            frame_save_info['results_track'] = results_track
            try:
                g_frame_info_for_save_queue.get_nowait()  
            except Exception as e:
                pass
            g_frame_info_for_save_queue.put_nowait(frame_save_info)

    def result_postprocess(self, results, frame_info):
        frame = frame_info['frame']

        results_grade, results_track, timestamp = results
        results_ = results_grade
        if cfg.use_track:
            results_ = results_track    

        if cfg.only_binary_classes_crop_and_weed:
            results_ = cfg.global_detector.merge_bboxes_to_binary_classes(results_)

        if cfg.filter_out_obj_for_reasons:
            if cfg.filter_out_obj_not_in_roi:
                results_ = cfg.global_detector.filter_out_obj_not_in_roi(results_)

            if cfg.filter_out_overlapped_plants:
                results_ = cfg.global_detector.filter_out_overlapped_plants(results_) 
                # results_ = cfg.global_detector.filter_out_overlapped_plants_v0(results_) 

            if cfg.filter_out_weeds_close_to_crops:
                # results_ = cfg.global_detector.filter_out_weeds_close_to_crops_v0(results_, cfg.safety_zone_size_pixel)
                results_ = cfg.global_detector.filter_out_weeds_close_to_crops(results_, cfg.safety_zone_size_pixel)

            if cfg.filter_out_overlapped_plants_old:
                for old_results in self.results_history_list:
                    old_results_grade, old_results_track, old_timestamp = old_results
                    old_results_ = old_results_grade
                    if cfg.use_track:
                        old_results_ = old_results_track
                    results_ = cfg.global_detector.filter_out_obj_overlap_with_crop(results_, ref_results=old_results_, ref_is_history=True)
                results_ = cfg.global_detector.filter_out_obj_overlap_with_crop(results_, ref_results=results_, ref_is_history=False)

            if cfg.filter_out_by_object_not_green:
                results_, frame = cfg.global_detector.filter_out_obj_not_green(results_, frame)
                
        if cfg.only_detection:
            results_grade = results_
        else:
            results_track = results_

        results = [results_grade, results_track, timestamp]
        return results, frame

    def run(self):
        print('WorkerDetection: start running')
        global g_frame_info_queue, g_first_detection, g_pause_event_by_detection

        cfg.global_detector.cur_model = cfg.global_window.comboBox_pt.currentText()
        cfg.global_detector.cur_model_version = cfg.global_window.comboBox_model_name.currentText()
        sleep(0.5)
        has_exception = False
        try:
            print('WorkerDetection: model_init')
            cfg.global_detector.model_init()

            cfg.global_detector.model_is_changed = False
            detection_loop_idx = 0
            while True:
                if cfg.global_camera_handler.camera_is_opened is False:
                    break
                if cfg.global_detector.model_is_changed is True:
                    break

                if cfg.global_detector is not None:
                    if cfg.global_window.detect_check.isChecked():
                        try:
                            frame_info = g_frame_info_queue.get(timeout=0.01)  # wait briefly for new frame
                        except Exception as e:
                            self.skip_frame_num += 1
                            continue

                        if 'frame' not in frame_info or 'timestamp' not in frame_info:
                            cfg.log_xalg('frame_info is empty')
                            self.skip_frame_num += 1
                            continue

                        if not hasattr(self, '_fps_frame_count'):
                            self._fps_frame_count = 0
                            self._fps_last_time = time()
                            self._fps_value = 0
                            self.skip_frame_num = 0


                        if not hasattr(self, '_fps_frame_count_model'):
                            self._fps_frame_count_model = 0
                            self._fps_last_time_model = time()
                            self._fps_value_model = 0

                        try:
                            results = cfg.global_detector.model_detect_one_frame(frame_info)
                            if cfg.is_debug:
                                # --- COUNT RAW DETECTIONS (BEFORE FILTERS) ---
                                results_grade_raw, results_track_raw, _ = results
                                active_raw = results_track_raw if (cfg.use_track and results_track_raw is not None) else results_grade_raw
                                raw_count = len(active_raw.boxes) if (active_raw is not None and hasattr(active_raw, 'boxes') and active_raw.boxes is not None) else 0
                                
                            assert len(results) == 3
                            # frame = frame_info['frame']
                            results, frame = self.result_postprocess(results, frame_info)

                            if cfg.is_debug:
                                # --- COUNT FILTERED DETECTIONS (AFTER FILTERS) ---
                                results_grade_post, results_track_post, _ = results
                                active_post = results_track_post if (cfg.use_track and results_track_post is not None) else results_grade_post
                                post_count = len(active_post.boxes) if (active_post is not None and hasattr(active_post, 'boxes') and active_post.boxes is not None) else 0

                                # 3. Log frame-by-frame comparison
                                frame_idx = frame_info.get('frame_idx', 0)
                                cfg.log_xalg(f"[Frame {frame_idx:05d}] Raw Model Bboxes: {raw_count:2d} | Post-Filter Bboxes: {post_count:2d}", show=True)

                            self.results_history_list.append(results)

                            if cfg.use_g_green_mask_frame:
                                cfg.g_green_mask_frame = frame

                            self._fps_frame_count_model += 1
                            now = time()
                            if now - self._fps_last_time_model >= 1.0:
                                self._fps_value_model = np.round(self._fps_frame_count_model / (now - self._fps_last_time_model), 2)
                                self._fps_last_time_model = now
                                self._fps_frame_count_model = 0
                            cfg.global_detector.cur_model_fps_purely = self._fps_value_model

                            move_cm_per_frame = cfg.current_physical_speed/max(self._fps_value, 1)
                            small_move_t = 3  # cm
                            detection_results_history_len = int(small_move_t/(move_cm_per_frame)*self._fps_value)  # 3 cm
                            detection_results_history_len = min(cfg.detection_results_history_len, self._fps_value//2)
                            # if detection_loop_idx % 100 == 0:
                            #     cfg.log_xalg('detection_results_history_len:', detection_results_history_len, show=False)

                            if len(self.results_history_list) > detection_results_history_len:
                                self.results_history_list.pop(0)
                            if g_first_detection:
                                g_pause_event_by_detection.set()
                            g_first_detection = False

                            self.put_results_into_queues(results, frame_info)

                        except Exception as e:
                            self.skip_frame_num += 1
                            cfg.log_xalg(traceback.format_exc())
                            cfg.log_xalg(e)
                            has_exception = True
                            # self.finished.emit()
                            break

                        self._fps_frame_count += 1
                        now = time()
                        if now - self._fps_last_time >= 1.0:
                            self._fps_value = np.round(self._fps_frame_count / (now - self._fps_last_time), 2)
                            self._fps_last_time = now
                            self._fps_frame_count = 0
                            if self.skip_frame_num > 0:
                                # cfg.log_xalg('detection thread -> self.skip_frame_num:', self.skip_frame_num)
                                pass
                            self.skip_frame_num = 0
                            # cfg.log_xalg('detection frame rate:', cfg.global_detector.cur_model_fps)
                        detection_loop_idx += 1
                        cfg.global_detector.cur_model_fps = self._fps_value
                        # print('detected obj num:', len(cfg.global_detector.results_grade.boxes), len(cfg.global_detector.results_track.boxes))
                    else:
                        sleep(0.001)
                        g_first_detection = True
        except Exception as e:
            cfg.log_xalg(traceback.format_exc())
            cfg.log_xalg(e)
        if not has_exception:
            # self.finished.emit()
            pass
        cfg.global_detector.model_is_changed = False


class WorkerDetectionTrack():
    def __init__(self):
        self.skip_frame_num = 0
        pass

    def run(self):
        print('WorkerDetectionTrack: start running')
        sleep(0.5)
        has_exception = False
        try:
            while True:
                if cfg.global_camera_handler.camera_is_opened is False:
                    break
                if cfg.global_detector.model_is_changed is True:
                    break

                if cfg.global_detector is not None:
                    if cfg.global_window.detect_check.isChecked():
                        try:
                            cfg.global_detector.track_after_detection()
                        except Exception as e:
                            print('track_after_detection:', e)
                            self.skip_frame_num += 1
                            continue
                        
                        sleep(0.001)
                        
                        if not hasattr(self, '_fps_frame_count'):
                            self._fps_frame_count = 0
                            self._fps_last_time = time()
                            self._fps_value = 0
                            self.skip_frame_num = 0

                        self._fps_frame_count += 1
                        now = time()
                        if now - self._fps_last_time >= 1.0:
                            self._fps_value = np.round(self._fps_frame_count / (now - self._fps_last_time), 2)
                            self._fps_last_time = now
                            self._fps_frame_count = 0
                            if self.skip_frame_num > 0:
                                cfg.log_xalg('detection track thread -> self.skip_frame_num:', self.skip_frame_num)
                            self.skip_frame_num = 0
                            cfg.global_detector.cur_model_track_fps = self._fps_value

                    else:
                        sleep(0.001)
        except Exception as e:
            cfg.log_xalg(traceback.format_exc())
            cfg.log_xalg(e)
        if not has_exception:
            # self.finished.emit()
            pass


class Detector():
    def __init__(self) -> None:
        self.loop_idx = 0
        self.plotter = Plotter()
        self.graded_sps = {}

        self.cur_model = None
        self.cur_model_version = None
        self.cur_model_fps = 0
        self.cur_model_fps_purely = 0
        self.cur_model_track_fps = 0
        self.cur_camera_retrieve_fps = 0
        self.model_is_changed = False

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model_added = False
        self.model_loading = False
        self.json_save_dir = []

        self.results_track = None
        self.results_track_last = None
        self.results_grade = None
        self.results_frame_timestamp = None
        
        self.frame_last = None
        self.frame_last_timestamp = None

        self.track_history = defaultdict(lambda: [])
        self.track_future_info = {}

        # ADD THIS: Set to permanently tag weeds that enter the safety zone
        self.unsafe_track_ids = set()

        self.main_loop_fps = 0

        if cfg.fast_tracker_for_only_detection:
            self.fast_tracker = FastSimpleTracker(
                max_distance_px=cfg.target_detection_size * 0.1, 
                max_age=cfg.track_length_max, 
                move_direction=cfg.move_direction_required
            )

        # ADD THIS NEW BLOCK:
        if cfg.lettuce_tracker_for_only_detection:
            self.lettuce_tracker = LettuceTracker(
                max_distance_px=cfg.target_detection_size * 0.1, 
                max_age=30,     # Allows holding the estimate for up to 30 frames
                move_direction=cfg.move_direction_required
            )

    def model_init(self):
        self.model_added = False
        self.model_loading = True
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(self.device)
        self.model_track = None
        self.model_grade = None
        if not cfg.pattern_detection_by_morphology:
            while not cfg.global_window.model_selection_finished:
                print('wait for model_selection_finished')
                sleep(0.1)
            if cfg.only_detection:
                self.model_grade = cfg.YOLO(cfg.model_paths['detection'])
                print("self.model_grade = YOLO(cfg.model_paths['detection'])")
            elif cfg.only_track:
                self.model_track = cfg.YOLO(cfg.model_paths['track'])
                print("self.model_track = YOLO(cfg.model_paths['track'])")
            else:
                self.model_grade = cfg.YOLO(cfg.model_paths['detection'])
                self.model_track = cfg.YOLO(cfg.model_paths['track'])
                print("self.model_grade = YOLO(cfg.model_paths['detection'])")
                print("self.model_track = YOLO(cfg.model_paths['track'])")
        else:
            print('pattern_detection_by_morphology is enabled, no model loaded')
        # self.class_names = self.model_grade.names
        self.class_names = cfg.sku_names
        if cfg.exp_type == 'indoor':
            self.class_names = self.class_names * cfg.color_num
        print('Class Names: ', self.class_names)

        print("model is added")
        self.model_added = True
        self.model_loading = False
        if self.model_track:
            self.model_track.to(self.device)
        if self.model_grade:
            self.model_grade.to(self.device)

    def results(self, results, frame):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        self.show_statistic(results)
        return frame

    def release_model(self):
        cfg.global_window.statusBar().showMessage("Release the current model.")

    def save_json(self, results, json_save_dir, Jnum):
        f = open(json_save_dir + "data%s.json" % Jnum, 'w')
        json.dump(results, f)
        f.close()

    def idx_to_label(self, idx):
        """
        For a given label value, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        """
        return self.class_names[int(idx)]

    def show_frame_rates(self, frame):
        model_fps = round(cfg.global_detector.cur_model_fps, 1)
        if model_fps > 120:
            model_fps = '120.0+'
        cv2.putText(frame, f'Model FPS: {model_fps}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)

        camera_fps = round(cfg.global_camera_handler.camera_fps, 1)
        if camera_fps > 120:
            camera_fps = '120.0+'
        cv2.putText(frame, f'Camera FPS: {camera_fps}', (250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 0, 0), 2)
        
        # if not cfg.is_for_Mark_sprayer:
        if cfg.async_valve_control:
            send_signal_fps = round(cfg.global_relay_monitor.worker_send_signal_valve.send_signal_fps, 1)
            if send_signal_fps > 120:
                send_signal_fps = '120.0+'
            cv2.putText(frame, f'Signal FPS: {send_signal_fps}', (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 200), 2)
        return frame

    def sync_detection(self):
        global g_frame_info_queue
        try:
            frame_info = g_frame_info_queue.get()  # wait briefly for new frame
        except Exception as e:
            self.skip_frame_num += 1
            return
        if not self.model_added:
            self.model_init()
        if not hasattr(self, '_fps_frame_count_detection'):
            self._fps_frame_count_detection = 0
            self._fps_last_time_detection = time()
            self._fps_value_detection = 0
        results_grade, results_track, results_frame_timestamp= self.model_detect_one_frame(frame_info)
        self._fps_frame_count_detection +=1
        end_time = time()
        now = end_time
        if now - self._fps_last_time_detection >= 1.0:
            self._fps_value_detection = self._fps_frame_count_detection / (now - self._fps_last_time_detection)
            self._fps_last_time_detection = now
            self._fps_frame_count_detection = 0
        cfg.global_detector.cur_model_fps = np.round(self._fps_value_detection, 2)
        return results_grade, results_track, results_frame_timestamp

    def assess_metrics(self):
        """
        ref to YOLOv8 code:
        ap_per_class(), compute_ap()
        only one class for indoor test 
        """
        global TPs, FPs, FNs, GTs

        TPs_np = np.array(TPs)
        FPs_np = np.array(FPs)
        FNs_np = np.array(FNs)
        GTs_np = np.array(GTs)

        precision_tot = sum(TPs)/(sum(TPs)+sum(FPs))
        recall_tot = sum(TPs)/(sum(TPs)+sum(FNs))

        tpc = TPs_np.cumsum(0)
        gt_num = np.sum(GTs_np)
        gt = GTs_np.cumsum(0)
        recall = tpc / gt

        fpc = FPs_np.cumsum(0)
        precision = tpc / (tpc + fpc)

        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([1.0], precision, [0.0]))

        mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
        method = 'interp'  # methods: 'continuous', 'interp'
        if method == 'interp':
            x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
            ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
        return precision_tot, recall_tot, ap

    def color_classify_pattern(self, img):
        config_indoor = False
        if config_indoor:
            lower_H = np.array([75])
            upper_H = np.array([90])

            lower_S = np.array([100])
            upper_S = np.array([120])
        else:
            # green mark
            lower_H = np.array([65])
            upper_H = np.array([100])

            lower_S = np.array([90])
            upper_S = np.array([130])

            # white 
            lower_H = np.array([90])
            upper_H = np.array([130])

            lower_S = np.array([80])
            upper_S = np.array([200])

            lower_V = np.array([100])
            upper_V = np.array([255])

            lower_V1 = np.array([200])
            upper_V1 = np.array([255])

        h = img.shape[0]
        w = img.shape[1]
        area = h*w

        obj_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        (H, S, V) = cv2.split(obj_hsv)

        mask_H = cv2.inRange(H, lower_H, upper_H)
        output_H = cv2.bitwise_and(H, H, mask=mask_H)

        mask_S = cv2.inRange(S, lower_S, upper_S)
        output_S = cv2.bitwise_and(S, S, mask=mask_S)

        mask_V = cv2.inRange(V, lower_V, upper_V)
        output_V = cv2.bitwise_and(V, V, mask=mask_S)

        mask_V1 = cv2.inRange(V, lower_V1, upper_V1)
        output_V1 = cv2.bitwise_and(V, V, mask=mask_S)

        mask_tot = mask_H & mask_S & mask_V 
        mask_tot = mask_tot | mask_V1
        
        output_tot = cv2.bitwise_and(S, S, mask=mask_tot)
        if config_indoor:
            output = output_S
        else:
            # output = output_tot
            output = mask_tot

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        iterations = 4
        output = cv2.dilate(output, kernel, iterations=iterations)
        output = cv2.erode(output, kernel, iterations=iterations)
        #  scikit-image
        from skimage.filters import threshold_yen, threshold_otsu, try_all_threshold
        # thresh = threshold_yen(output)
        ret, output_binary = cv2.threshold(output, 100, 255, cv2.THRESH_BINARY)
        if len(output_binary.shape) ==3:
            output_binary = output_binary[:, :, 1]
        return output, output_binary

    def detect_pattern(self, binary_img):
        """
        binary image
        """
        import cv2
        img_circle = np.stack([binary_img, binary_img, binary_img], axis=-1)
        # binary_img_filtered = cv2.medianBlur(binary_img, 5)

        # circles = cv2.HoughCircles(binary_img, cv2.HOUGH_GRADIENT, 1, 50,
        #                         param1=200, param2=15, minRadius=15, maxRadius=25)
        # circles = np.uint16(np.around(circles))
        # find_circle_list = []
        # for i_circle in range(circles.shape[1]):
        #     (x, y, r) = circles[0][i_circle]
        #     cv2.circle(img_circle, (x, y), r, (0, 0, 255), 2)
        #     cv2.circle(img_circle, (x, y), 2, (0, 255, 0), 3)
        #     find_circle_list.append((x, y, r))

        # use contour? if the circle is filled.
        contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_filtered = []
        for contour in contours:
            area = cv2.contourArea(contour)
            min_area = 15*15*3.1415
            if area > min_area:
                contours_filtered.append(contour)
        # print(circles.shape)
        cv2.drawContours(img_circle, contours_filtered, -1, (0,255,0), 3)
        return contours_filtered, img_circle

    def detect_pattern_by_morphology(self, frame):
        results_grade = [None]
        output, output_binary = self.color_classify_pattern(frame)
        contours, img_circle = self.detect_pattern(output_binary)
        boxes = []
        #Let (x,y) be the top-left coordinate of the rectangle and (w,h) be its width and height.
        for contour in contours:
            x,y,w,h = cv2.boundingRect(contour)
            box = [x, y, x+w, y+h, 'Target']
            boxes.append(box)
        results_grade = boxes
        return results_grade

    def remove_passed_tracks(self):
        track_history_tmp = self.track_history.copy()
        for track_id, track in track_history_tmp.items():
            # print(track_id, len(self.track_history[track_id]))
            if len(self.track_history[track_id]) > cfg.track_length_max:
                half_length = int(cfg.track_length_max // 2)
                self.track_history[track_id] = self.track_history[track_id][half_length:]

        if len(self.track_history) > cfg.track_trace_max_num:
            track_history_tmp = self.track_history.copy()
            for track_id, track in track_history_tmp.items():
                box_latest = track[-1]
                x, y, x1, y1, datetime_latest = box_latest[:5]
                # bbox move from right to left
                time_gap = datetime.now() - datetime_latest
                time_gap = time_gap.total_seconds()
                if cfg.track_remove_time_gap < time_gap:
                    del self.track_history[track_id]

                    # ADD THIS: Cleanup unsafe tracks
                    if track_id in self.unsafe_track_ids:
                        self.unsafe_track_ids.remove(track_id)
            if len(self.track_history) > cfg.track_trace_max_num:
                track_history_tmp = list(self.track_history.keys())
                track_history_tmp_sorted = sorted(track_history_tmp)
                for i_item, track_id in enumerate(track_history_tmp_sorted):
                    if i_item >= int(cfg.track_trace_max_num//2):
                        break
                    if track_id in self.track_history:
                        del self.track_history[track_id]
                        # ADD THIS: Cleanup unsafe tracks
                        if track_id in self.unsafe_track_ids:
                            self.unsafe_track_ids.remove(track_id)

    def bb_intersection_over_union(self, boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
        # return the intersection over union value
        return iou

    def largely_overlap_with_crop(self, box, bboxes, ref_is_history):
        x, y, x1, y1, conf, sku_idx, sku_label = box[:8]
        is_crop = False
        if sku_label in cfg.vegetable_names:
            return is_crop
        for i, box_ref in enumerate(bboxes):
            x_t = int(box_ref[0])
            y_t = int(box_ref[1])
            x1_t = int(box_ref[2])
            y1_t = int(box_ref[3])
            if len(box_ref) == 7:
                sku_idx_t = int(box_ref[6])
            else:
                assert len(box_ref) == 6
                sku_idx_t = int(box_ref[5])
            sku_label_t = cfg.global_detector.class_names[int(sku_idx_t)]
            if sku_label_t not in cfg.vegetable_names:
                continue
            iou = self.bb_intersection_over_union([x, y, x1, y1], [x_t, y_t, x1_t, y1_t])
            if iou > cfg.largely_overlap_with_crop_iou:
                is_crop = True
                break
        return is_crop

    def add_bbox_into_track(self, results_track, timestamp):
        lastest_track_ids = []
        if results_track.boxes and results_track.boxes.is_track:
            # boxes = results_track.boxes.xywh.cpu().numpy()
            boxes_latest_frame = results_track.boxes.cpu().numpy()
            track_ids_latest_frame = results_track.boxes.id.int().cpu().numpy().tolist()
            bboxes = results_track.boxes.data.cpu().numpy().copy()
            # masks = results.masks.data.cpu()
            for box, track_id in zip(boxes_latest_frame, track_ids_latest_frame):
                #cls conf xyxy
                lastest_track_ids.append(track_id)
                x, y, x1, y1 = box.xyxy[0]  
                conf = box.conf[0]
                sku_idx = box.cls[0]

                if sku_idx == cfg.filtered_sku_id:    # Instantly catch the dummy flag
                    self.unsafe_track_ids.add(track_id)
                    # COMMENT OUT OR REMOVE `continue`
                    # We must NOT skip appending here. We want to track its movement 
                    # so it retains its original track_id.
                    # continue
                    if track_id in cfg.track_schedule_table:
                        cfg.track_schedule_table[track_id]["triggered"] = True
                    sku_label = 'Weed in Safty Zone'
                else:
                    sku_label = cfg.global_detector.class_names[int(sku_idx)]
                if sku_label in cfg.vegetable_names:
                    continue
                # is_crop = self.largely_overlap_with_crop([x, y, x1, y1, conf, sku_idx, sku_label], bboxes)
                # if is_crop:
                #     continue
                track = self.track_history[track_id]
                if cfg.use_fake_bbox_in_track:
                    track_real = self.get_real_track_from_track(track)
                    if track_real != track:
                        self.track_history[track_id] = track_real
                is_fake = False
                self.track_history[track_id].append((float(x), float(y), float(x1), float(y1), timestamp, conf, sku_idx, sku_label, is_fake))  # x, y center point
        return lastest_track_ids

    def get_real_track_from_track(self, track):
        track_real = track.copy()
        for i_trace in range(-1, -len(track), -1):
            newest_box = track[i_trace]
            is_fake_trace_idx = 8
            if newest_box[is_fake_trace_idx] is False:
                if i_trace != -1:
                    track_real = track[:i_trace+1]
                break
        return track_real

    def compensate_tracks(self, results_track):
        track_ids_latest_frame = results_track.boxes.id.int().cpu().numpy().tolist()
        now = datetime.now()
        for track_id in self.track_history:
            track = self.track_history[track_id]
            if len(track) < cfg.track_length_min:
                continue
            track_real = self.get_real_track_from_track(track)
            if track_id not in track_ids_latest_frame:
                move_distance, move_time_used, ave_pixel_speed, move_angle, move_distance = self.calc_track_average_speed(track_real)
                newest_box = track_real[-1]
                x_b,y_b,x1_b,y1_b, timestamp_b, conf_b, sku_idx_b, sku_label_b = newest_box[:8]
                if cfg.move_direction_required < 0:
                    if move_distance > 0:
                        continue
                    bbox_represent_x = x_b
                    pass_target = bbox_represent_x - cfg.target_closest_trigger_x
                else:
                    if move_distance < 0:
                        continue
                    bbox_represent_x = x1_b
                    pass_target = cfg.target_closest_trigger_x - bbox_represent_x
                if pass_target < 0:
                    continue
                time_gap = (now - timestamp_b).total_seconds()
                x_move = ave_pixel_speed * time_gap * cfg.move_direction_required
                y_move = x_move * np.tan(move_angle)
                x_estimate = x_b + x_move
                y_estimate = y_b + y_move
                x1_estimate = x1_b + x_move
                y1_estimate = y1_b + y_move
                is_fake = True
                track.append((float(x_estimate), float(y_estimate), float(x1_estimate), float(y1_estimate), now, conf_b, sku_idx_b, sku_label_b, is_fake))  # x, y center point
                cfg.log_xalg(f'------use_fake_bbox_in_track: {track_id}, x_estimate: {x_estimate}, y_estimate: {y_estimate}------', show=True)
            if len(track_real) > cfg.track_length_max:  # retain 30 tracks for 30 frames
                track.pop(0)

    def track_after_detection(self):
        try:
            frame_result_info = cfg.frame_result_for_track_queue.get(timeout=0.01)  # wait briefly for new frame
        except Exception as e:
            return
        results_grade = frame_result_info['results_grade']
        results_track = frame_result_info['results_track']
        results_frame_timestamp = frame_result_info['timestamp']
        if cfg.only_detection:
            if results_grade is None:
                return
        else:
            if results_track is None:
                return

        if cfg.use_track:    
            if results_track.boxes.id is not None:
                lastest_track_ids = self.add_bbox_into_track(results_track, results_frame_timestamp)

                if cfg.use_fake_bbox_in_track:
                    self.compensate_tracks(results_track)

                self.check_box_location_from_track(lastest_track_ids)

                self.remove_passed_tracks()

    def model_detect_one_frame(self, frame_info):
        frame = frame_info['frame']
        if cfg.has_infrared_channel:
            frame = frame[:, :, :3]
        timestamp = frame_info['timestamp']
        frame_idx = frame_info['frame_idx']
        if cfg.pattern_detection_by_morphology:
            results_grade = self.detect_pattern_by_morphology(frame)
            results_track = [None]
        elif cfg.only_detection:
            results_grade = self.model_grade.predict(frame, show=False, save=False, show_labels=False, show_conf=False, conf=cfg.detection_conf_t, iou=cfg.detection_iou_t, save_txt=False, verbose=False)
            results_grade = results_grade[0]

            results_track = None

            if cfg.lettuce_tracker_for_only_detection:
                # --- ADD LETTUCE TRACKING LOGIC ---
                bboxes_np = results_grade.boxes.data.cpu().numpy() if (results_grade.boxes is not None and len(results_grade.boxes) > 0) else np.empty((0, 6))

                if len(bboxes_np) > 0:
                    is_crop_mask = np.array([self.class_names[int(cls)] in cfg.vegetable_names for cls in bboxes_np[:, 5]])
                    crop_bboxes = bboxes_np[is_crop_mask]
                else:
                    crop_bboxes = np.empty((0, 6))

                self.lettuce_tracker.update(crop_bboxes, timestamp, cfg.pixel_to_physical_space_resized, cfg.current_physical_speed)
                # ----------------------------------

            if cfg.fast_tracker_for_only_detection:
                # 2. High-speed custom tracking on predictions
                if results_grade.boxes is not None and len(results_grade.boxes) > 0:
                    bboxes_np = results_grade.boxes.data.cpu().numpy()  # [x1, y1, x2, y2, conf, cls]
                    track_ids = self.fast_tracker.update(bboxes_np)
                    
                    # Attach IDs to Ultralytics Boxes object so downstream functions work automatically
                    ids_tensor = torch.tensor(track_ids, dtype=results_grade.boxes.data.dtype, device=results_grade.boxes.data.device).unsqueeze(1)
                    # results_grade.boxes.id = ids_tensor
                    xyxy = results_grade.boxes.data[:, :4]
                    conf_cls = results_grade.boxes.data[:, 4:]
                    new_track_data = torch.cat((xyxy, ids_tensor, conf_cls), dim=1)
                    
                    # Replace the Boxes object entirely so Ultralytics recognizes it as a tracked frame
                    from ultralytics.engine.results import Boxes
                    results_grade.boxes = Boxes(new_track_data, orig_shape=results_grade.orig_shape)
                results_track = results_grade  # Pass as results_track so trigger/signal pipeline receives it
        elif cfg.only_track:
            results_track = self.model_track.track(frame, persist=True, conf=cfg.detection_conf_t, iou=cfg.detection_iou_t, save_txt=False, tracker=cfg.track_alg, verbose=False)
            results_grade = None
            results_track = results_track[0]
        else:
            results_grade = self.model_grade.predict(frame, show=False, save=False, show_labels=False, show_conf=False, conf=cfg.detection_conf_t, iou=cfg.detection_iou_t, save_txt=False, verbose=False)
            results_track = self.model_track.track(frame, persist=True, conf=cfg.detection_conf_t, iou=cfg.detection_iou_t, save_txt=False, tracker=cfg.track_alg, verbose=False)
            results_track = results_track[0]
            results_grade = results_grade[0]
        return results_grade, results_track, timestamp

    def calc_track_average_speed(self, track):
        # oldest_box = track[0]
        if len(track) >= cfg.track_ref_frame_gap:
            oldest_box = track[-cfg.track_ref_frame_gap]
        else:
            oldest_box = track[-cfg.track_length_min]
        newest_box = track[-1]

        x_old,y_old,x1_old,y1_old, timestamp_old = oldest_box[:5]
        x,y,x1,y1, timestamp = newest_box[:5]

        x_ave_old = (x1_old + x_old)/2
        y_ave_old = (y1_old + y_old)/2
        x_ave = (x+x1)/2
        y_ave = (y+y1)/2

        if cfg.move_direction_required < 0:
            move_distance_abs = abs(x1 - x1_old)
            move_distance = x1 - x1_old
        else:
            move_distance_abs = abs(x-x_old)
            move_distance = x - x_old
        move_time_used = (timestamp - timestamp_old).total_seconds()
        if move_distance_abs < cfg.track_move_distance_min or move_time_used <= 0:
            ave_pixel_speed = 0
            move_angle = 0
        else:
            ave_pixel_speed = move_distance_abs / move_time_used
            move_angle = np.arctan2(y_ave - y_ave_old, x_ave - x_ave_old)
        return move_distance_abs, move_time_used, ave_pixel_speed, move_angle, move_distance

    def calc_time_needed_to_reach_target_x(self, track, target_x):
        newest_box = track[-1]
        x,y,x1,y1, timestamp = newest_box[:5]
        y_ave = (y+y1)/2
        bbox_w = x1-x
        bbox_h = y1-y
        
        if cfg.move_direction_required < 0:
            bbox_represent_x = x
            gap_reach_triger_area = bbox_represent_x - target_x
        else:
            bbox_represent_x = x1
            gap_reach_triger_area = target_x - bbox_represent_x
        move_distance_abs, move_time_used, ave_pixel_speed, move_angle, move_distance = self.calc_track_average_speed(track)
        if move_distance_abs < cfg.track_move_distance_min or move_time_used <= 0:
            time_needed = float('inf')
            trigger_timestamp = None
            future_position_y = y_ave
        else:
            time_needed = gap_reach_triger_area / ave_pixel_speed
            if time_needed > 0:
                trigger_timestamp = timedelta(seconds=time_needed)+timestamp
            else:
                trigger_timestamp = None
            future_position_y = y_ave + gap_reach_triger_area * np.tan(move_angle)
            # future_position_y = np.clip(future_position_y, future_position_y-100, future_position_y+100) 
        future_position_x = target_x
        future_position = (future_position_x, future_position_y, bbox_w, bbox_h)
        return trigger_timestamp, time_needed, ave_pixel_speed, move_distance, future_position

    def calc_time_needed_to_reach_triger_area(self, track):
        trigger_timestamp, time_needed, ave_pixel_speed, move_distance, future_position = self.calc_time_needed_to_reach_target_x(track, cfg.target_closest_trigger_x)
        return trigger_timestamp, time_needed, ave_pixel_speed, move_distance, future_position

    def find_closest_index(self, nozzle_center_locations, value):
        """
        Finds the closest physical nozzle index (1-based) to a given spatial coordinate.
        Optimized to eliminate dynamic offset allocations and branching overhead.
        """
        # Convert list to array if it isn't one, to utilize vectorized math
        locations = np.asarray(nozzle_center_locations)
        
        # 1. Calculate absolute spatial distances to all nozzles instantly
        distances = np.abs(locations - value)
        
        # 2. Extract the closest 0-based array slot index
        closest_arr_idx = np.argmin(distances)
        
        # 3. Convert to your 1-based valve number notation
        closest_real_idx = int(closest_arr_idx + 1)
        
        # 4. Perform a unified out-of-bounds check against your physical nozzle gap rules
        # Handles edge-of-boom out-of-bound targets cleanly
        if closest_real_idx == cfg.valid_valve_range[0] or closest_real_idx == cfg.valid_valve_range[-1]:
            if distances[closest_arr_idx] >= (cfg.nozzle_gap * 2.0):
                return -1

        # 5. Return index if it falls within valid operational boundaries
        if cfg.valid_valve_range[0] <= closest_real_idx <= cfg.valid_valve_range[-1]:
            return closest_real_idx
            
        return -1

    def find_closest_index_v0(self, nozzle_center_locations, value):
        idx_in_array = np.searchsorted(nozzle_center_locations, value)
        idx_in_array_less_than_valve = idx_in_array - 1  # to actual nozzle idx instead of idx in nozzle_center_locations
        idx_in_array_equal_or_bigger_than_valve = idx_in_array  # to actual nozzle idx instead of idx in nozzle_center_locations
        idx_in_array_less_than_valve_nozzle_idx = idx_in_array_less_than_valve + 1 
        idx_in_array_equal_or_bigger_than_valve_nozzle_idx = idx_in_array_equal_or_bigger_than_valve + 1 
        if idx_in_array_equal_or_bigger_than_valve_nozzle_idx <= cfg.valid_valve_range[0]:
            if abs(nozzle_center_locations[cfg.valid_valve_range[0]-1] - value) < cfg.nozzle_gap*2:
                closest_real_idx = cfg.valid_valve_range[0]
            else:
                return -1
        elif idx_in_array_less_than_valve_nozzle_idx >= cfg.valid_valve_range[-1]:
            if abs(nozzle_center_locations[cfg.valid_valve_range[-1]-1] - value) < cfg.nozzle_gap*2:
                closest_real_idx = cfg.valid_valve_range[-1]
            else:
                return -1
        else:
            before_idx = max(idx_in_array_less_than_valve_nozzle_idx, cfg.valid_valve_range[0])
            after_idx = min(idx_in_array_equal_or_bigger_than_valve_nozzle_idx, cfg.valid_valve_range[-1])
            before_value = nozzle_center_locations[before_idx-1]
            after_value = nozzle_center_locations[after_idx-1]
            if before_value == after_value:
                closest_real_idx = idx_in_array_equal_or_bigger_than_valve_nozzle_idx
            elif abs(value - before_value) < abs(value - after_value):
                closest_real_idx = idx_in_array_less_than_valve_nozzle_idx
            else:
                closest_real_idx = idx_in_array_equal_or_bigger_than_valve_nozzle_idx
        if closest_real_idx in cfg.valid_valve_range:
            return closest_real_idx
        else:
            return -1

    def add_valve_idx_to_array(self, valve_idx, nearest_valve_idxes_tmp):
        """Safely appends valid valve indexes within config boundaries."""
        if cfg.valid_valve_range[0] <= valve_idx <= cfg.valid_valve_range[-1]:
            if valve_idx not in nearest_valve_idxes_tmp:
                nearest_valve_idxes_tmp.append(valve_idx)

    def find_closest_nozzle_to_weed(self, bbox):
        """
        Determines which nozzle spray indexes span the weed bounding box area.
        Avoids tracking errors by utilizing a vector boundary lookup framework.
        """
        x_ave, y_ave, bbox_w, bbox_h = bbox
        nearest_valve_idxes_tmp = []

        ymin = y_ave - (bbox_h / 2.0)
        ymax = y_ave + (bbox_h / 2.0)

        # Catch-all override: If weed covers the entire spray boom span
        if ymin <= cfg.nozzle_center_locations[0] and ymax >= cfg.nozzle_center_locations[-1]:
            nearest_valve_idxes_future = cfg.valid_valve_range.tolist()
            # cfg.log_xalg('nearest_valve_idxes_future:', nearest_valve_idxes_future, show=False)
            return nearest_valve_idxes_future

        # Step 1: Always include the center point projection nozzle
        ave_valve_idx = self.find_closest_index(cfg.nozzle_center_locations, y_ave)
        self.add_valve_idx_to_array(ave_valve_idx, nearest_valve_idxes_tmp)

        # Step 2: Extract boundary nozzle index values
        min_valve_idx = self.find_closest_index(cfg.nozzle_center_locations, ymin)
        max_valve_idx = self.find_closest_index(cfg.nozzle_center_locations, ymax)

        # Step 3: Handle lower boundary checks gracefully (Fixes the index -1 bug)
        if min_valve_idx >= cfg.valid_valve_range[0]:
            if bbox_h < cfg.nozzle_gap:
                self.add_valve_idx_to_array(min_valve_idx, nearest_valve_idxes_tmp)
            else:
                # Direct safe indexing lookup protection
                lookup_idx = max(0, min_valve_idx - 1)
                if cfg.nozzle_center_locations[lookup_idx] >= ymin:
                    self.add_valve_idx_to_array(min_valve_idx, nearest_valve_idxes_tmp)
                else:
                    target_idx = min_valve_idx + 1
                    if target_idx <= cfg.valid_valve_range[-1]:
                        self.add_valve_idx_to_array(target_idx, nearest_valve_idxes_tmp)

        # Step 4: Handle upper boundary checks gracefully
        if max_valve_idx >= cfg.valid_valve_range[0]:
            if bbox_h < cfg.nozzle_gap:
                self.add_valve_idx_to_array(max_valve_idx, nearest_valve_idxes_tmp)
            else:
                lookup_idx = max(0, max_valve_idx - 1)
                if cfg.nozzle_center_locations[lookup_idx] <= ymax:
                    self.add_valve_idx_to_array(max_valve_idx, nearest_valve_idxes_tmp)
                else:
                    target_idx = max_valve_idx - 1
                    if target_idx >= cfg.valid_valve_range[0]:
                        self.add_valve_idx_to_array(target_idx, nearest_valve_idxes_tmp)

        # Step 5: Fallback error checking for missing extreme triggers (-1 returns)
        if max_valve_idx == -1 and min_valve_idx >= cfg.valid_valve_range[0]:
            fallback_max = cfg.valid_valve_range[-1]
            if cfg.nozzle_center_locations[fallback_max - 1] <= ymax:
                self.add_valve_idx_to_array(fallback_max, nearest_valve_idxes_tmp)
                
        if min_valve_idx == -1 and (cfg.valid_valve_range[0] <= max_valve_idx <= cfg.valid_valve_range[-1]):
            fallback_min = cfg.valid_valve_range[0]
            if cfg.nozzle_center_locations[fallback_min - 1] >= ymin:
                self.add_valve_idx_to_array(fallback_min, nearest_valve_idxes_tmp)

        # Step 6: Bridge the continuous spray span without array performance degradation
        if nearest_valve_idxes_tmp:
            min_idx = int(max(cfg.valid_valve_range[0], min(nearest_valve_idxes_tmp)))
            max_idx = int(min(cfg.valid_valve_range[-1], max(nearest_valve_idxes_tmp)))
            
            # Generate continuous range values directly, avoiding multiple appends
            span_range = list(range(min_idx, max_idx + 1))
            nearest_valve_idxes_tmp.extend(span_range)

        # Step 7: Clean and return data arrays
        unique_nozzles = sorted(list(set(nearest_valve_idxes_tmp)))
        nearest_valve_idxes_future = [idx for idx in unique_nozzles if cfg.valid_valve_range[0] <= idx <= cfg.valid_valve_range[-1]]

        # cfg.log_xalg('nearest_valve_idxes_future:', nearest_valve_idxes_future, show=False)
        return nearest_valve_idxes_future

    def add_valve_idx_to_array_v0(self, valve_idx, nearest_valve_idxes_tmp):
        smallest_valve_idx = cfg.valid_valve_range[0]
        if valve_idx >= smallest_valve_idx:
            if valve_idx not in nearest_valve_idxes_tmp:
                nearest_valve_idxes_tmp += [valve_idx]

    def find_closest_nozzle_to_weed_v0(self, bbox):
        x_ave, y_ave, bbox_w, bbox_h = bbox
        nearest_valve_idxes_future = []
        nearest_valve_idxes_tmp = []
        ave_valve_idx = self.find_closest_index(cfg.nozzle_center_locations, y_ave)
        self.add_valve_idx_to_array(ave_valve_idx, nearest_valve_idxes_tmp)

        ymin = y_ave - bbox_h / 2
        min_valve_idx = self.find_closest_index(cfg.nozzle_center_locations, ymin)
        if min_valve_idx >= cfg.valid_valve_range[0]:
            if bbox_h < cfg.nozzle_gap:
                self.add_valve_idx_to_array(min_valve_idx, nearest_valve_idxes_tmp)
            else:
                if cfg.nozzle_center_locations[min_valve_idx-1] >= ymin:
                    self.add_valve_idx_to_array(min_valve_idx, nearest_valve_idxes_tmp)
                else:
                    min_valve_idx += 1
                    if min_valve_idx <= cfg.valid_valve_range[-1]:
                        self.add_valve_idx_to_array(min_valve_idx, nearest_valve_idxes_tmp)
                    else:
                        self.add_valve_idx_to_array(min_valve_idx, nearest_valve_idxes_tmp)
                        # print(f'min_valve_idx: {min_valve_idx} > cfg.valid_valve_range[-1]')

        ymax = y_ave + bbox_h / 2
        max_valve_idx = self.find_closest_index(cfg.nozzle_center_locations, ymax)
        if max_valve_idx >= cfg.valid_valve_range[0]:
            if bbox_h < cfg.nozzle_gap:
                self.add_valve_idx_to_array(max_valve_idx, nearest_valve_idxes_tmp)
            else:
                if cfg.nozzle_center_locations[max_valve_idx-1] <= ymax:
                    self.add_valve_idx_to_array(max_valve_idx, nearest_valve_idxes_tmp)
                else:
                    max_valve_idx -= 1
                    if max_valve_idx >= cfg.valid_valve_range[0]:
                        self.add_valve_idx_to_array(max_valve_idx, nearest_valve_idxes_tmp)
                    else:
                        self.add_valve_idx_to_array(max_valve_idx, nearest_valve_idxes_tmp)
                        # print(f'max_valve_idx: {max_valve_idx} < cfg.valid_valve_range[0]')

        if max_valve_idx == -1 and min_valve_idx >= cfg.valid_valve_range[0]:
            max_valve_idx = cfg.valid_valve_range[-1]
            if cfg.nozzle_center_locations[max_valve_idx-1] <= ymax:
                self.add_valve_idx_to_array(max_valve_idx, nearest_valve_idxes_tmp)
        if min_valve_idx == -1 and cfg.valid_valve_range[0] <= max_valve_idx <= cfg.valid_valve_range[-1]:
            min_valve_idx = cfg.valid_valve_range[0]
            if cfg.nozzle_center_locations[min_valve_idx-1] >= ymin:
                self.add_valve_idx_to_array(min_valve_idx, nearest_valve_idxes_tmp)

        if ymax >= cfg.nozzle_center_locations[-1] and ymin <= cfg.nozzle_center_locations[-0]:
            nearest_valve_idxes_tmp = cfg.valid_valve_range.tolist()

        if len(nearest_valve_idxes_tmp) > 0:
            min_idx = max(cfg.valid_valve_range[0], min(nearest_valve_idxes_tmp))
            max_idx = min(cfg.valid_valve_range[-1], max(nearest_valve_idxes_tmp))
            if min_idx in cfg.valid_valve_range and max_idx in cfg.valid_valve_range:
                for idx in range(min_idx, max_idx+1):
                    self.add_valve_idx_to_array(idx, nearest_valve_idxes_tmp)
            # possible_nozzles = nearest_valve_idxes_tmp[-100:]
            possible_nozzles = nearest_valve_idxes_tmp
            possible_nozzles = np.unique(possible_nozzles)
            possible_nozzles = np.sort(possible_nozzles)
            nearest_valve_idxes_future = np.unique(possible_nozzles).tolist()
            cfg.log_xalg('nearest_valve_idxes_future:', nearest_valve_idxes_future, show=False)
        return nearest_valve_idxes_future

    def check_box_location_from_track(self, lastest_track_ids):
        for id_track, track in self.track_history.items():
            if id_track not in lastest_track_ids:
                continue

            # ADD THIS: Skip calculating triggers for tagged unsafe weeds
            if id_track in self.unsafe_track_ids:
                continue

            if len(track) < cfg.track_length_min:
                continue
            trigger_timestamp, time_needed, ave_pixel_speed, move_distance, future_position = self.calc_time_needed_to_reach_triger_area(track)
            if time_needed < 0 or time_needed == float('inf') or time_needed > cfg.time_needed_max_allowed:
                continue
            else:
                nearest_valve_idxes_future = self.find_closest_nozzle_to_weed(future_position)
            newest_box = track[-1]
            x,y,x1,y1, timestamp, conf, sku_idx, sku_label, is_fake = newest_box[:9]
            if id_track not in self.track_future_info:
                self.track_future_info[id_track] = {}
            self.track_future_info[id_track]['ave_pixel_speed'] = ave_pixel_speed
            self.track_future_info[id_track]['move_distance'] = round(move_distance, 3)
            self.track_future_info[id_track]['current_timestamp'] = timestamp
            self.track_future_info[id_track]['trigger_timestamp'] = trigger_timestamp
            self.track_future_info[id_track]['time_needed'] = time_needed
            self.track_future_info[id_track]['nozzles'] = nearest_valve_idxes_future
            self.track_future_info[id_track]['current_bbox'] = [x,y,x1,y1, conf, sku_idx, sku_label, is_fake]
            self.track_future_info[id_track]['future_position'] = future_position
            x_size = x1-x
            self.track_future_info[id_track]['x_size'] = x_size
            try:
                cfg.track_future_info_queue.get_nowait()  
            except Exception as e:
                pass
            cfg.track_future_info_queue.put_nowait(self.track_future_info)

    def filter_out_obj_overlap_with_crop(self, results, ref_results, ref_is_history):
        # boxes_latest_frame = results.boxes.cpu().numpy()
        if cfg.exp_type != 'field':
            return results
        bboxes = results.boxes.data.cpu().numpy().copy()
        bboxes_ref = ref_results.boxes.data.cpu().numpy().copy()
        keep_indices = []
        for i, box in enumerate(bboxes):
            #cls conf xyxy
            x = int(box[0])
            y = int(box[1])
            x1 = int(box[2])
            y1 = int(box[3])
            if results.boxes.id is not None:
                assert len(box) == 7
                data_id = int(box[4])
                conf = int(box[5])
                sku_idx = int(box[6])
            else:
                assert len(box) == 6
                conf = int(box[4])
                sku_idx = int(box[5])
            sku_label = cfg.global_detector.class_names[int(sku_idx)]
            is_crop = self.largely_overlap_with_crop([x, y, x1, y1, conf, sku_idx, sku_label], bboxes_ref, ref_is_history)
            if is_crop and sku_label not in cfg.vegetable_names:
                continue
            keep_indices.append(i)
        # Remove the boxes that don't meet criteria
        # Apply filtering
        if isinstance(results, list):
            results = [results[i] for i in keep_indices]
        else:
            results.boxes.data = results.boxes.data[keep_indices]
        return results

    def filter_out_obj_not_green(self, results, frame):
        if isinstance(results, list):
            bboxes = np.array(results)
        else:
            bboxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()

        keep_indices = []
        # Define range for green color in HSV
        lower_green = np.array([35, 30, 30])
        upper_green = np.array([85, 255, 255])
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask_frame = cv2.inRange(frame_hsv, lower_green, upper_green)
        frame = cv2.bitwise_and(frame, frame, mask=mask_frame)

        for i, box in enumerate(bboxes):
            xmin = int(box[0])
            ymin = int(box[1])
            xmax = int(box[2])
            ymax = int(box[3])
            conf = confs[i]
            
            bbox = frame[ymin:ymax, xmin:xmax]
            hsv_roi = cv2.cvtColor(bbox, cv2.COLOR_BGR2HSV)

            # Create a mask for green color
            mask = cv2.inRange(hsv_roi, lower_green, upper_green)

            # Check if there are any green pixels
            green_pixels = cv2.countNonZero(mask)

            if green_pixels <= 2000:
                continue
            # Keep box if it passed all filters
            keep_indices.append(i)

        # Apply filtering
        if isinstance(results, list):
            results = [results[i] for i in keep_indices]
        else:
            results.boxes.data = results.boxes.data[keep_indices]
        return results, frame

    def merge_bboxes_to_binary_classes(self, results):
        original_boxes = results.boxes.data
        boxes = original_boxes.clone()
        classes = results.boxes.cls.cpu().numpy()

        new_classes = []

        for i in range(boxes.shape[0]):
            cls_idx = int(classes[i])
            sku_label = cfg.global_detector.class_names[cls_idx]
            if sku_label in cfg.vegetable_names:
                new_classes.append(cfg.vegetable_id)  # Lettuce
            else:
                new_classes.append(cfg.weed_id)  # Weed
        if results.boxes.id is not None:
            sku_idx_pos = 6
        else:
            sku_idx_pos = 5
        # Replace class column (index 5)
        boxes[:, sku_idx_pos] = torch.tensor(new_classes, dtype=boxes.dtype, device=boxes.device)

        # Replace results.boxes with a new Boxes object
        results.boxes = Boxes(boxes, orig_shape=results.orig_shape)
        return results

    def filter_out_overlapped_plants(self, results, iou_threshold=0.25, iom_threshold=0.75):
        import torchvision
        if results.boxes is None or len(results.boxes) == 0:
            return results

        boxes = results.boxes.data.clone()
        xyxy = boxes[:, :4]
        classes = boxes[:, 5]

        # 1. Identify crops via tensor ops
        is_crop = torch.zeros(len(boxes), dtype=torch.bool, device=boxes.device)
        for i in range(len(boxes)):
            cls_idx = int(classes[i])
            sku_label = cfg.global_detector.class_names[cls_idx] if hasattr(cfg, 'global_detector') else str(cls_idx)
            if sku_label in getattr(cfg, 'vegetable_names', []) or cls_idx == getattr(cfg, 'vegetable_id', -1):
                is_crop[i] = True

        # 2. Vectorized Areas and Intersections
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        iou_matrix = torchvision.ops.box_iou(xyxy, xyxy)

        # Manually calculate Intersection Area matrix for IoM
        lt = torch.max(xyxy[:, None, :2], xyxy[:, :2])
        rb = torch.min(xyxy[:, None, 2:], xyxy[:, 2:])
        wh = (rb - lt).clamp(min=0)
        inter_matrix = wh[:, :, 0] * wh[:, :, 1]

        min_areas = torch.min(areas[:, None], areas[None, :])
        iom_matrix = inter_matrix / min_areas.clamp(min=1e-6)

        # 3. Find conflicts using masks
        overlap_mask = (iou_matrix > iou_threshold) | (iom_matrix > iom_threshold)
        overlap_mask.fill_diagonal_(False) # Ignore self-overlap

        keep_flags = torch.ones(len(boxes), dtype=torch.bool, device=boxes.device)
        conflicts = overlap_mask.nonzero(as_tuple=False)

        # 4. Resolve isolated conflicts cleanly
        for i, j in conflicts:
            if not keep_flags[i] or not keep_flags[j]:
                continue
            
            if is_crop[i] and not is_crop[j]:
                keep_flags[j] = False
            elif not is_crop[i] and is_crop[j]:
                keep_flags[i] = False
            else:
                if areas[i] <= areas[j]:
                    keep_flags[j] = False
                else:
                    keep_flags[i] = False

        # Only update the final list to save overhead
        results.boxes = Boxes(boxes[keep_flags], orig_shape=results.orig_shape)
        return results

    def filter_out_overlapped_plants_v0(self, results, iou_threshold=0.25, iom_threshold=0.75):
        """
        Filters bounding boxes inside the Ultralytics Results object based on overlap.
        Rules:
        1. Prefers Crop over Weed.
        2. If same category and highly overlapping or contained, keep the SMALLER area box.
        """
        if results.boxes is None or len(results.boxes) == 0:
            return results

        # 1. Extract data tensors
        original_boxes = results.boxes.data
        boxes = original_boxes.clone()
        
        xyxy = results.boxes.xyxy.cpu().numpy()
        conf = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()

        # 2. Identify which indices correspond to crops
        is_crop = []
        for i in range(boxes.shape[0]):
            cls_idx = int(classes[i])
            sku_label = cfg.global_detector.class_names[cls_idx] if hasattr(cfg, 'global_detector') else str(cls_idx)
            
            # Check against both naming and ID variables based on your cfg configurations
            crop_cond = (
                sku_label in getattr(cfg, 'vegetable_names', []) or 
                cls_idx in getattr(cfg, 'crop_label_idxes', []) or
                cls_idx == getattr(cfg, 'vegetable_id', -1)
            )
            is_crop.append(crop_cond)

        # 3. Pre-calculate areas for all bounding boxes
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        keep_flags = [True] * len(boxes)

        # 4. Compare all pairs to find overlaps and apply exclusion rules
        for i in range(len(boxes)):
            if not keep_flags[i]:
                continue
            for j in range(i + 1, len(boxes)):
                if not keep_flags[j]:
                    continue

                # Calculate Intersection bounds
                xA = max(xyxy[i, 0], xyxy[j, 0])
                yA = max(xyxy[i, 1], xyxy[j, 1])
                xB = min(xyxy[i, 2], xyxy[j, 2])
                yB = min(xyxy[i, 3], xyxy[j, 3])

                interArea = max(0, xB - xA) * max(0, yB - yA)
                if interArea == 0:
                    continue

                # Calculate metrics
                boxAArea = areas[i]
                boxBArea = areas[j]
                
                iou_val = interArea / float(boxAArea + boxBArea - interArea)
                iom_val = interArea / float(min(boxAArea, boxBArea))  # Intersection over Minimum

                is_overlapping = iou_val > iou_threshold
                is_contained = iom_val > iom_threshold

                # Apply Rules if an overlap threshold is crossed
                if is_overlapping or is_contained:
                    
                    # Rule 1: Preference to Crop over Weed
                    if is_crop[i] and not is_crop[j]:
                        keep_flags[j] = False
                    elif not is_crop[i] and is_crop[j]:
                        keep_flags[i] = False
                        break  # Break inner loop since box 'i' is discarded
                        
                    # Rule 2: Same class type logic
                    else:
                        # Overlap or Contained logic: Keep the SMALLER bounding box
                        if boxAArea <= boxBArea:
                            keep_flags[j] = False  # Box 'j' is larger, discard it
                        else:
                            keep_flags[i] = False  # Box 'i' is larger, discard it
                            break

        # 5. Extract retained rows and reconstruct Ultralytics Boxes object
        indices_to_keep = [idx for idx, keep in enumerate(keep_flags) if keep]
        
        # Only update if something was actually removed to save overhead
        if len(indices_to_keep) < len(boxes):
            filtered_boxes = boxes[indices_to_keep]
            results.boxes = Boxes(filtered_boxes, orig_shape=results.orig_shape)

        return results

    def filter_out_weeds_close_to_crops(self, results, safety_zone_size_pixel):
        if results.boxes is None or len(results.boxes) == 0:
            return results

        original_boxes = results.boxes.data
        boxes = original_boxes.clone()
        classes = results.boxes.cls.cpu().numpy()

        crop_indices = []
        weed_indices = []

        # Separate crop and weed indices based on configured labels/IDs
        for i in range(boxes.shape[0]):
            cls_idx = int(classes[i])
            sku_label = self.class_names[cls_idx]
            
            if sku_label in cfg.vegetable_names or cls_idx == getattr(cfg, 'vegetable_id', -1):
                crop_indices.append(i)
            # Exclude the dummy tracker ID if it somehow already exists in the raw prediction
            elif cls_idx != getattr(cfg, 'filtered_sku_id', -1):
                weed_indices.append(i)

        if not crop_indices and not weed_indices:
            return results
        if not weed_indices:
            return results

        weed_boxes = boxes[weed_indices, :4]  # [x1, y1, x2, y2]

        # --- 1. Gather All Crop Boxes (Detected + Tracked) ---
        all_crop_boxes_list = []
        if crop_indices:
            all_crop_boxes_list.append(boxes[crop_indices, :4])

        tracked_crop_boxes = []
        
        if cfg.lettuce_tracker_for_only_detection:
            # Extract tracking history boxes
            for tid, t_info in self.lettuce_tracker.tracks.items():
                tracked_crop_boxes.append(t_info['box'])
            
            if tracked_crop_boxes:
                # Convert tracked numpy boxes back to tensors on the correct device
                tracked_tensor = torch.tensor(np.array(tracked_crop_boxes), dtype=boxes.dtype, device=boxes.device)
                all_crop_boxes_list.append(tracked_tensor)

        # --- 2. Proximity / Safety Zone Distance Check ---
        is_too_close = torch.zeros(len(weed_indices), dtype=torch.bool, device=boxes.device)
        
        # --- 2. Proximity / Safety Zone Distance Check ---
        is_too_close = torch.zeros(len(weed_indices), dtype=torch.bool, device=boxes.device)
        
        if all_crop_boxes_list:
            all_crop_boxes = torch.cat(all_crop_boxes_list, dim=0)

            weed_centers = torch.stack([
                (weed_boxes[:, 0] + weed_boxes[:, 2]) / 2.0,
                (weed_boxes[:, 1] + weed_boxes[:, 3]) / 2.0
            ], dim=1)

            crop_centers = torch.stack([
                (all_crop_boxes[:, 0] + all_crop_boxes[:, 2]) / 2.0,
                (all_crop_boxes[:, 1] + all_crop_boxes[:, 3]) / 2.0
            ], dim=1)

            # distances shape: (num_weeds, num_crops)
            distances = torch.cdist(weed_centers, crop_centers, p=2.0)

            if cfg.safety_zone_size_adjusted_by_bbox_size:
                # Calculate the diagonal lengths of weeds and crops
                weed_diags = torch.hypot(weed_boxes[:, 2] - weed_boxes[:, 0], weed_boxes[:, 3] - weed_boxes[:, 1])
                crop_diags = torch.hypot(all_crop_boxes[:, 2] - all_crop_boxes[:, 0], all_crop_boxes[:, 3] - all_crop_boxes[:, 1])

                # Create a dynamic (num_weeds, num_crops) threshold matrix. 
                # (Divided by 2.0 to represent the radius from the center to the corner)
                # dynamic_threshold = safety_zone_size_pixel + (weed_diags.unsqueeze(1) / 2.0) + (crop_diags.unsqueeze(0) / 2.0)
                dynamic_threshold = safety_zone_size_pixel + (crop_diags.unsqueeze(0) / 2.0)
                
                # A weed is too close if its distance to ANY crop is less than that specific pair's dynamic threshold
                is_too_close = (distances <= dynamic_threshold).any(dim=1)
            else:
                min_distances_to_crop, _ = torch.min(distances, dim=1)
                is_too_close = min_distances_to_crop <= safety_zone_size_pixel

        # --- 3. IOU Tracker Overlap Check (Fake Weed Identifier) ---
        is_fake_weed = torch.zeros(len(weed_indices), dtype=torch.bool, device=boxes.device)
        
        if cfg.lettuce_tracker_for_only_detection and tracked_crop_boxes:
            iou_threshold = cfg.lettuce_tracker_for_only_detection_iou_t
            for w_idx, weed_box_tensor in enumerate(weed_boxes):
                if is_too_close[w_idx]:  # Skip overlap math if already flagged for proximity
                    continue
                
                # weed_box_np = weed_box_tensor.cpu().numpy()
                # for t_box in tracked_crop_boxes:
                #     if self.bb_intersection_over_union(weed_box_np, t_box) > iou_threshold:
                #         is_fake_weed[w_idx] = True
                #         break
                ious = torchvision.ops.box_iou(weed_boxes, tracked_tensor)
                # A weed is fake if its IoU with ANY tracked crop exceeds the threshold
                is_fake_weed = (ious > iou_threshold).any(dim=1)

        # --- 4. Re-assign Class IDs instead of Filtering ---
        # Identify which weed indices need to be suppressed
        suppress_mask = is_too_close | is_fake_weed
        suppress_weed_indices = [weed_indices[i] for i, suppress in enumerate(suppress_mask) if suppress]

        if suppress_weed_indices:
            dummy_id = cfg.filtered_sku_id
            # The class index in ultralytics Boxes.data is always the last column (-1)
            boxes[suppress_weed_indices, -1] = dummy_id

        if cfg.is_debug:
            removed_count = suppress_mask.sum().item()
            if removed_count > 0:
                proximity_count = is_too_close.sum().item()
                iou_count = is_fake_weed.sum().item()
                
                log_msg = (f"[Safety Filter] Flagged {removed_count} weeds as Fake "
                           f"({proximity_count} in safety zone, {iou_count} via tracker IOU overlap).")
                cfg.log_xalg(log_msg, show=True)

        # We keep all boxes (crops + valid weeds + flagged fake weeds) so they flow into the plotter
        results.boxes = Boxes(boxes, orig_shape=results.orig_shape)
        return results

    def filter_out_weeds_close_to_crops_v0(self, results, safety_zone_size_pixel):
        if results.boxes is None or len(results.boxes) == 0:
            return results

        original_boxes = results.boxes.data
        boxes = original_boxes.clone()
        classes = results.boxes.cls.cpu().numpy()

        crop_indices = []
        weed_indices = []

        # Separate crop and weed indices based on configured labels/IDs
        for i in range(boxes.shape[0]):
            cls_idx = int(classes[i])
            sku_label = cfg.global_detector.class_names[cls_idx]
            if sku_label in cfg.vegetable_names or cls_idx == cfg.vegetable_id:
                crop_indices.append(i)
            else:
                weed_indices.append(i)

        if not crop_indices or not weed_indices:
            return results

        crop_boxes = boxes[crop_indices, :4]  # [x1, y1, x2, y2]
        weed_boxes = boxes[weed_indices, :4]  # [x1, y1, x2, y2]

        # # --- 1. Overlap / Intersection Check ---
        # # Broadening dimensions for vector comparison:
        # # Weed shapes: (num_weeds, 1, 4), Crop shapes: (1, num_crops, 4)
        # w_b = weed_boxes.unsqueeze(1)
        # c_b = crop_boxes.unsqueeze(0)

        # # Determine coordinate boundaries of intersection boxes
        # inter_x1 = torch.max(w_b[:, :, 0], c_b[:, :, 0])
        # inter_y1 = torch.max(w_b[:, :, 1], c_b[:, :, 1])
        # inter_x2 = torch.min(w_b[:, :, 2], c_b[:, :, 2])
        # inter_y2 = torch.min(w_b[:, :, 3], c_b[:, :, 3])

        # # Overlap occurs when inter_x2 > inter_x1 AND inter_y2 > inter_y1
        # has_overlap = (inter_x2 > inter_x1) & (inter_y2 > inter_y1)
        # is_overlapping_crop = torch.any(has_overlap, dim=1)  # Shape: (num_weeds,)

        # --- 2. Proximity / Safety Zone Check ---
        weed_centers = torch.stack([
            (weed_boxes[:, 0] + weed_boxes[:, 2]) / 2.0,
            (weed_boxes[:, 1] + weed_boxes[:, 3]) / 2.0
        ], dim=1)

        crop_centers = torch.stack([
            (crop_boxes[:, 0] + crop_boxes[:, 2]) / 2.0,
            (crop_boxes[:, 1] + crop_boxes[:, 3]) / 2.0
        ], dim=1)

        distances = torch.cdist(weed_centers, crop_centers, p=2)
        min_distances_to_crop, _ = torch.min(distances, dim=1)
        is_too_close = min_distances_to_crop <= safety_zone_size_pixel

        # --- 3. Filter Combination ---
        # Weed is discarded if it overlaps OR is within the safety zone
        # weeds_to_keep_mask = ~(is_overlapping_crop | is_too_close)
        weeds_to_keep_mask = ~is_too_close
        kept_weed_indices = [weed_indices[i] for i in range(len(weed_indices)) if weeds_to_keep_mask[i]]

        if cfg.is_debug:
            # --- ADDED LOGGING START ---
            removed_mask = (is_too_close)
            removed_count = removed_mask.sum().item()

            if removed_count > 0:
                # overlap_count = is_overlapping_crop.sum().item()
                overlap_count = 0
                proximity_count = removed_count - overlap_count  # Weeds removed *only* because of proximity
                
                # Extract coordinates for the log as standard integers
                removed_boxes = weed_boxes[removed_mask].cpu().numpy().astype(int)
                
                log_msg = (f"[Safety Filter] Removed {removed_count} weeds ( "
                        f"{proximity_count} in safety zone pixel {safety_zone_size_pixel}). Distances: {min_distances_to_crop[is_too_close].tolist()}")
                
                # Log it to your console/file
                cfg.log_xalg(log_msg, show=True)
            # --- ADDED LOGGING END ---

        # Reconstruct box list keeping all crops and filtered weeds
        indices_to_keep = sorted(crop_indices + kept_weed_indices)
        filtered_boxes = boxes[indices_to_keep]

        results.boxes = Boxes(filtered_boxes, orig_shape=results.orig_shape)
        return results
        
    def filter_out_obj_not_in_roi(self, results):
        """
        Filter out objects not in ROI
        """
        if isinstance(results, list):
            bboxes = np.array(results)
        else:
            bboxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()

        keep_indices = []

        for i, box in enumerate(bboxes):
            xmin = int(box[0])
            ymin = int(box[1])
            xmax = int(box[2])
            ymax = int(box[3])
            conf = confs[i]
            # ROI logic
            # middle_point = (cfg.nozzle_trigger_x_start+cfg.nozzle_trigger_x_end) / 2

            if not cfg.is_pad_height:
                xmin_in_gui = cfg.detect_size_pad_half + xmin
                xmax_in_gui = cfg.detect_size_pad_half + xmax
                ymin_in_gui = ymin
                ymax_in_gui = ymax
            else:
                xmin_in_gui = xmin
                xmax_in_gui = xmax
                ymin_in_gui = ymin + cfg.detect_size_pad_half
                ymax_in_gui = ymax + cfg.detect_size_pad_half
            if cfg.move_direction_required < 0:
                if cfg.only_track:
                    # if xmin_in_gui < cfg.nozzle_trigger_x_start:
                    if xmax_in_gui < cfg.nozzle_trigger_x_end:
                        continue
                elif cfg.only_detection:
                    if xmax_in_gui < cfg.nozzle_trigger_x_end:
                        continue
                else:
                    assert False
                if conf < cfg.filter_out_obj_not_in_roi_conf_t:
                    if xmax_in_gui > cfg.detect_size_with_pad[0]-cfg.nozzle_trigger_x_start:
                        if xmin_in_gui > cfg.detect_size_with_pad[0]-2*cfg.nozzle_trigger_x_start:
                            continue
            else:
                # if xmax_in_gui > middle_point:
                #     continue
                if xmax_in_gui > cfg.nozzle_trigger_x_end:
                    continue

            if ymax_in_gui < min(cfg.nozzle_range_y_start, cfg.nozzle_range_y_end) - 2 * cfg.nozzle_gap:
                continue
            if ymin_in_gui > max(cfg.nozzle_range_y_start, cfg.nozzle_range_y_end) + 2 * cfg.nozzle_gap:
                continue

            # Keep box if it passed all filters
            keep_indices.append(i)

        # Apply filtering
        if isinstance(results, list):
            results = [results[i] for i in keep_indices]
        else:
            results.boxes.data = results.boxes.data[keep_indices]
        return results

    def main_loop(self):
        global g_frame_raw, g_resize_frame_for_display_raw

        self.cur_model_version = cfg.global_window.comboBox_model_name.currentText()
        self.cur_model = cfg.global_window.comboBox_pt.currentText()

        if not hasattr(self, '_fps_frame_count'):
            self._fps_frame_count = 0
            self._fps_last_time = time()
            self._fps_value = 0 

        msg = ''
        if self.model_loading:
            cfg.global_windowmodel_selection_finished = True
        elif self.model_added:
            cfg.global_windowmodel_selection_finished = False

        try:
            results = None
            with g_frame_raw_lock:
                if g_frame_raw is None:
                    return
                frame = g_frame_raw.copy()

                if cfg.g_green_mask_frame is not None:
                    final_frame = cfg.g_green_mask_frame.copy()
                else:
                    final_frame = g_resize_frame_for_display_raw.copy()
            if cfg.has_infrared_channel:
                final_frame = final_frame[:, :, :3]
                final_frame = np.ascontiguousarray(final_frame)
            if cfg.global_window.detect_check.isChecked():
                if not cfg.async_detection:
                    results_grade, results_track, results_frame_timestamp = self.sync_detection()
                else:
                    try:
                        frame_result_info = cfg.frame_result_for_display_queue.get(timeout=0.01)  # wait briefly for new frame
                    except Exception as e:
                        return 
                    results_grade = frame_result_info['results_grade']
                    results_track = frame_result_info['results_track']
                try:
                    if final_frame is None:
                        return
                    resize_frame_for_display = self.plotter.plot_valve_location(final_frame)

                    if not cfg.use_track:
                        final_frame = self.plotter.plot_boxes_detection(results_grade, resize_frame_for_display)
                    else:
                        if results_track is None:
                            if self.results_track_last is not None:
                                results = self.results_track_last
                            else:
                                results = None
                        else:
                            results = results_track
                            self.results_track_last = results_track
                        if results is not None:
                            final_frame = self.plotter.plot_boxes_track(results, self.track_future_info, self.track_history, resize_frame_for_display)
                        else:
                            # print(f'display loop_idx: {self.loop_idx} ->  results is None')
                            pass
                        final_frame = self.plotter.plot_boxes_track_triggering(final_frame)
                except Exception as e:
                    print(traceback.format_exc())
                    print(e)

            if final_frame is None:
                return

            self.show_frame_rates(final_frame)

            label_width = cfg.global_window.label_pic.width()
            label_height = cfg.global_window.label_pic.height()
            temp_imgSrc = QImage(final_frame, final_frame.shape[1], final_frame.shape[0], final_frame.shape[1] * 3, QImage.Format_RGB888)
            pixmap_imgSrc = QPixmap.fromImage(temp_imgSrc).scaled(label_width, label_height)
            cfg.global_window.label_pic.setPixmap(pixmap_imgSrc)
            cfg.global_window.label_pic.repaint()

            if cfg.global_window.save_check.isChecked():
                if cfg.save_newest_frame:
                    save_frame = final_frame
                else:
                    global g_frame_info_for_save_queue
                    try:
                        frame_save_info = g_frame_info_for_save_queue.get(timeout=0.01)  # wait briefly for new frame
                    except Exception as e:
                        return 
                    save_frame = self.plotter.plot_valve_location(frame_save_info['frame'])
                    if results_track is None:
                        results = frame_save_info['results_grade']
                        save_frame = self.plotter.plot_boxes_detection(results, save_frame)
                    else:
                        results = frame_save_info['results_track']
                        save_frame = self.plotter.plot_boxes_track(results, self.track_future_info, self.track_history, save_frame)
                        save_frame = self.plotter.plot_boxes_track_triggering(save_frame)
                    self.show_frame_rates(save_frame)
                cfg.frame_cnt_save += 1
                frame_info_for_save_dict = {}
                if cfg.save_origin_video:
                    frame_info_for_save_dict['frame_origin_for_save'] = frame
                if cfg.save_detect_video:
                    frame_info_for_save_dict['frame_detected_for_save'] = save_frame
                frame_info_for_save_dict['results_save'] = None
                frame_info_for_save_dict['frame_cnt_save'] = cfg.frame_cnt_save
                try:
                    cfg.frame_info_for_save_queue.get_nowait()  
                except Exception as e:
                    pass
                cfg.frame_info_for_save_queue.put_nowait(frame_info_for_save_dict)

            if cfg.show_step_by_step:
                global g_pause_event
                g_pause_event.set()
                cv2.namedWindow('detect', 0)
                cv2.imshow('detect', save_frame[:, :, ::-1])
                cv2.waitKey(0)

            if cfg.sync_valve_control:
                cfg.global_relay_monitor.worker_send_signal_valve.run_sync()

            self._fps_frame_count += 1
            now = time()
            if now - self._fps_last_time >= 1.0:
                self._fps_value = np.round(self._fps_frame_count / (now - self._fps_last_time), 1)
                self._fps_last_time = now
                self._fps_frame_count = 0
                self.main_loop_fps = self._fps_value
                msg = f'Display FPS: {self.main_loop_fps}, \
                    camera retrieve FPS: {self.cur_camera_retrieve_fps}, \
                        track FPS: {self.cur_model_track_fps}, \
                        pure detection FPS: {self.cur_model_fps_purely}'
                        
                cfg.global_window.statusBar().showMessage(msg)
            self.loop_idx+=1
        except Exception as e:
            print(traceback.format_exc())
            print(e)


def main():
    detection_mode_path = r'.\model\best_v11s_36epochs_800.pt'
    model_grade = YOLO(detection_mode_path)
    # Explicitly move the model to CUDA if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_grade.to(device)
    xxx = cv2.imread(r'demo\00000064_00000000003A89A8.png')
    xxx = cv2.resize(xxx, (800,800))
    results_grade = model_grade.predict(xxx, show=False, save=False, show_labels=False, show_conf=False, conf=0.5, iou=0.4, save_txt=False, verbose=False)
    pass


if __name__ =='__main__':
    main()
