import sys
import cv2
import torch
import json
import os
import threading
from typing import Optional
from time import time, sleep
from datetime import datetime
import numpy as np
from queue import SimpleQueue, Queue
from itertools import count
from loguru import logger
from collections import defaultdict

# from yolox.data.data_augment import ValTransform
# from yolox.data.datasets import COCO_CLASSES
# from yolox.exp import get_exp
# from yolox.utils import fuse_model, get_model_info, postprocess, vis

import ctypes
import random
import config as cfg
from detection.util import match_track_with_detect, combine_measurements, save_data

def GetFinalGrade(Final_list):

    id, m_len, mid_len, m_wid, mid_wid, f_grad=Final_list
    if mid_len<9 and mid_len>=6 and mid_wid <3.5 and (f_grad=="Normal" or f_grad=="Minor defects"):
        F_SampleGrade="Premium"
    elif mid_len<6 and mid_len>=3 and mid_wid <3.5 and (f_grad=="Normal" or f_grad=="Minor defects"):
        F_SampleGrade = "Good"
    elif mid_wid <3.5 and f_grad=="Severe defects":
        F_SampleGrade = "Fair"
    else:
        F_SampleGrade = "Cull"
    return F_SampleGrade

class Plotter():
    def __init__(self) -> None:
        self.object_records = {}
        self.frame_counter = {}
        self.skip_frames=5
        self.cnt_frame = 0
        self.graded_sps = {}
        self.tot_weed_num = 0
        self.tot_valid_track_info = {}

    def plot_boxes(self, model, results, frame):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        self = model
        if self.cur_model_version == "yolov8":
            labels = results[0].boxes.cls
            conf = results[0].boxes.conf
            cord = results[0].boxes.xyxyn
            conf = torch.unsqueeze(conf, dim=1)
            cord = torch.cat((cord, conf), axis=1)
        elif self.cur_model_version == "yolox":
            if self.cur_model == "best_ckpt1.pth":
                size_ratio = 800.0/640
            else:
                size_ratio = 1.0
            results = results[0]
            if results is None:
                return frame
            labels = results[:, 6]
            conf = results[:, 4]*results[:, 5]
            conf = torch.unsqueeze(conf, dim=1)
            cord = results[:, 0:4]*size_ratio
            cord = torch.cat((cord, conf), axis=1)
        elif self.cur_model_version == "yolor" or self.cur_model_version == "yolov4":
            results = results[0]
            labels = results[:, 5]
            conf = results[:, 4]
            conf = torch.unsqueeze(conf, dim=1)
            cord = results[:, 0:4]
            cord = torch.cat((cord, conf), axis=1)
        elif self.cur_model == "bestS.engine":
            cord = np.concatenate((results[0], results[1].reshape(-1, 1)), axis=1)
            labels = results[2]
        else:
            labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= self.confSpinBox.value():
                x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(
                    row[3] * y_shape)
                if self.cur_model_version == "yolox" or self.cur_model == "bestS.engine":
                    x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
                elif self.cur_model_version == "yolor":
                    x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
                bgr = (255, 150, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 3)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 3)
        self.show_statistic(results)
        return frame

    def plot_oriboxes(self, results, c_frame):
        if self.cur_model_version == "yolov8":
            labels = results[0].boxes.cls
            conf = results[0].boxes.conf
            cord = results[0].boxes.xyxyn
            conf = torch.unsqueeze(conf, dim=1)
            cord = torch.cat((cord, conf), axis=1)
        elif self.cur_model_version == "yolox":
            if self.cur_model == "best_ckpt1.pth":
                size_ratio = 800.0 / 640
            else:
                size_ratio = 1.0
            results = results[0]
            if results is None:
                return c_frame
            labels = results[:, 6]
            conf = results[:, 4]*results[:, 5]
            conf = torch.unsqueeze(conf, dim=1)
            cord = results[:, 0:4]*size_ratio
            cord = torch.cat((cord, conf), axis=1)
        elif self.cur_model_version == "yolor" or self.cur_model_version == "yolov4":
            results = results[0]
            labels = results[:, 5]
            conf = results[:, 4]
            conf = torch.unsqueeze(conf, dim=1)
            cord = results[:, 0:4]
            cord = torch.cat((cord, conf), axis=1)
        elif self.cur_model == "bestS.engine":

            cord = np.concatenate((results[0], results[1].reshape(-1, 1)), axis=1)
            labels = results[2]
        else:
            labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

        n = len(labels)
        x_shape, y_shape = c_frame.shape[1], c_frame.shape[0]
        scale_x = x_shape / 717
        scale_y = y_shape / 669
        for i in range(n):
            row = cord[i]
            if row[4] >= self.confSpinBox.value():

                x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(
                    row[3] * y_shape)
                if self.cur_model_version == "yolox" or self.cur_model == "bestS,engine":
                    x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
                    x1, y1, x2, y2 = x1*scale_x, y1*scale_y, x2*scale_x, y2*scale_y
                elif self.cur_model_version == "yolor":
                    x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
                    x1, y1, x2, y2 = x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y
                bgr = (255, 0, 0)
                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)
                cv2.rectangle(c_frame, (x1, y1), (x2, y2), bgr, 3)
                cv2.putText(c_frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 3)
        return c_frame
    
    def plot_one_box(self, x, img, color=None, label=None, line_thickness=2, show=False):
        # Plots one bounding box on image img
        tl = line_thickness or round(0.001 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
        random.seed(0)
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label and cfg.plot_labels:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3

            if cfg.plot_improve_label_visibility:
                cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
                cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
            else:
                cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, color, thickness=tf, lineType=cv2.LINE_AA)
        if show:
            cv2.namedWindow('x', 0)
            cv2.imshow('x', img)
            cv2.waitKey(0)

    def overlay(self, image, mask, color, alpha, resize=None):
        """Combines image and its segmentation mask into a single image.

        Params:
            image: Training image. np.ndarray,
            mask: Segmentation mask. np.ndarray,
            color: Color for segmentation mask rendering.  tuple[int, int, int] = (255, 0, 0)
            alpha: Segmentation mask's transparency. float = 0.5,
            resize: If provided, both image and its mask are resized before blending them together.
            tuple[int, int] = (1024, 1024))

        Returns:
            image_combined: The combined image. np.ndarray

        """
        # color = color[::-1]
        colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
        colored_mask = np.moveaxis(colored_mask, 0, -1)
        masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
        image_overlay = masked.filled()

        if resize is not None:
            image = cv2.resize(image.transpose(1, 2, 0), resize)
            image_overlay = cv2.resize(image_overlay.transpose(1, 2, 0), resize)

        image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)
        return image_combined

    def plot_boxes_track_triggering(self, frame):
        for id_track in cfg.track_schedule_table.copy():
            if id_track not in cfg.track_schedule_table:
                continue
            if cfg.track_schedule_table[id_track]['triggered'] is False:
                continue
            if cfg.track_schedule_table[id_track]['triggering_plot'] is True:
                continue
            nozzles = cfg.track_schedule_table[id_track]['nozzles'][-1].copy()
            nozzle_time_offsets = cfg.track_schedule_table[id_track]['nozzle_time_offsets'].copy()
            for nozzle_idx in nozzles:
                if nozzle_idx == float('inf'):
                    continue
                nozzle_idx = np.clip(nozzle_idx, cfg.valid_valve_range[0], cfg.valid_valve_range[-1])
                if cfg.nozzle_center_locations[nozzle_idx-1] == float('inf'):
                    continue
                # show = f'{nozzle_idx}-{id_track}'
                show = f'{id_track}'
                if cfg.use_time_offset:
                    if nozzle_idx in nozzle_time_offsets:
                        time_offset = nozzle_time_offsets[nozzle_idx]
                        time_offset = round(time_offset, 2)
                        show += f'->{time_offset}'
                    else:
                        pass
                if nozzle_idx > cfg.valve_num:
                    print('nozzle_idx:', nozzle_idx)
                    
                nozzle_y = round(cfg.nozzle_center_locations[nozzle_idx-1])
                if cfg.is_pad_height:
                    x_start_gui = cfg.nozzle_trigger_x_start
                else:
                    x_start_gui = cfg.detect_size_pad_half+cfg.nozzle_trigger_x_start

                cv2.putText(frame, show, (x_start_gui, nozzle_y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 5, 0), 2)
            cfg.track_schedule_table[id_track]['triggering_plot_frame_num'] -= 1
            if cfg.track_schedule_table[id_track]['triggering_plot_frame_num'] < 0:
                cfg.track_schedule_table[id_track]['triggering_plot'] = True
        return frame

    def plot_boxes_track(self, results, track_future_info, track_history_, resize_frame_for_display):
        frame = self.plot_boxes_detection(results, resize_frame_for_display)

        if results.boxes and results.boxes.is_track:
            # Visualize the result on the frame
            # frame = results.plot()
            # Plot the tracks
            track_ids_latest_frame = results.boxes.id.int().cpu().numpy().tolist()
            track_history = track_history_.copy()
            for track_id, track in track_history.items():
                is_new_track = False 
                if track_id not in track_ids_latest_frame:
                    continue
                if len(track_future_info) == 0:
                    continue
                if track_id not in track_future_info:
                    continue
                trigger_timestamp = track_future_info[track_id]['trigger_timestamp']
                if trigger_timestamp is None:
                    continue
                current_timestamp = datetime.now()
                time_gap = (trigger_timestamp-current_timestamp).total_seconds()
                if time_gap < 0 or time_gap > cfg.show_track_time_gap_max:
                    continue

                # if len(track) >= cfg.track_length_min:
                #     if track_id not in self.tot_valid_track_info:
                #         self.tot_valid_track_info[track_id] = track
                #         self.tot_weed_num += 1
                #         is_new_track = True
                #     else:
                #         self.tot_valid_track_info[track_id] = track

                time_needed = track_future_info[track_id]['time_needed']

                box_latest = track[-1]
                track_center = []
                track_center_fake = []
                for box in track:
                    x, y, x1, y1, timestamp, _, _, _, is_fake = box[:9]
                    if is_fake:
                        track_center_fake.append([(x + x1) / 2, (y + y1) / 2])
                    else:
                        track_center.append([(x + x1) / 2, (y + y1) / 2])
                timestamp_str = timestamp.strftime("%M:%S.%f")[:-3]
                if is_new_track:
                    color = (255, 155, 0)
                    trace_color = color
                else:
                    color = (155, 0, 155)
                    trace_color = (100, 230, 150)
                # Draw the tracking lines
                if len(track_center) > 0:
                    points = np.hstack(track_center).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [points], isClosed=False, color=trace_color, thickness=2)
                if len(track_center_fake) > 0:
                    points_fake = np.hstack(track_center_fake).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [points_fake], isClosed=False, color=(50,50,255), thickness=3)

                x, y, x1, y1, timestamp = box_latest[:5]
                x, y, x1, y1 = round(x), round(y), round(x1), round(y1)

                time_needed = str(round(time_needed, 2))
                show = time_needed + '<-' + timestamp_str
                cv2.putText(frame, f"{show}", (x, y-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (25, 25, 255), 1, )
                cv2.putText(frame, f"Id {track_id}", (x1+5, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, )
            # cv2.putText(frame, f"Weed Count {self.tot_fruit_num}", (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 3, cv2.LINE_AA)
        return frame
    
    def plot_boxes_track_sweetpotato(self, results_track, results_grade, resize_frame_for_detect, resize_frame_for_display):
        self.cnt_frame += 1
        offset = 0.0332 # in/px
        line_left_x = 50
        line_right_x = cfg.detect_size_with_pad[1]-50
        
        random.seed(0)
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in cfg.global_detector.class_names]
        
        h_detect, w_detect = resize_frame_for_detect.shape[:2]
        h_display, w_display = resize_frame_for_display.shape[:2]
        h_ratio = h_display / h_detect
        w_ratio = w_display / w_detect
        img = resize_frame_for_display
        if results_track.masks is not None and results_track.boxes.id is not None:
            masks = results_track.masks.data.cpu()
            ids = results_track.boxes.id.cpu().numpy().astype(int)
            for seg, box, id in zip(masks.data.cpu().numpy(), results_track.boxes, ids):
                seg = cv2.resize(seg, (w_detect, h_detect))
                seg_int = seg.astype(np.uint8)
                res = cv2.findContours(seg_int, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # _, contours, _ = res
                contours, _ = res
                contour = contours[0]
                bboxes = results_grade.boxes.data.cpu().numpy()
                if len(contour) >= 20 and len(bboxes) > 0:
                    ellipse = cv2.fitEllipse(contour)
                    center, axes, angle = ellipse
                    long_axis = max(axes)
                    short_axis = min(axes)
                    # Compute endpoints of long axis
                    long_axis_angle_rad = np.deg2rad(angle)
                    long_axis_length = long_axis / 2
                    sin_angle = np.sin(long_axis_angle_rad)
                    cos_angle = np.cos(long_axis_angle_rad)
                    pt1 = (int(center[0] - long_axis_length * sin_angle), int(center[1] + long_axis_length * cos_angle))
                    pt2 = (int(center[0] + long_axis_length * sin_angle), int(center[1] - long_axis_length * cos_angle))

                    # Compute endpoints of short axis
                    short_axis_angle_rad = np.deg2rad(angle + 90)
                    short_axis_length = short_axis / 2
                    sin_angle = np.sin(short_axis_angle_rad)
                    cos_angle = np.cos(short_axis_angle_rad)
                    pt3 = (int(center[0] - short_axis_length * sin_angle), int(center[1] + short_axis_length * cos_angle))
                    pt4 = (int(center[0] + short_axis_length * sin_angle), int(center[1] - short_axis_length * cos_angle))

                    # Draw the long and short axes
                    cv2.line(img, pt1, pt2, (0, 255, 0), 2)
                    cv2.line(img, pt3, pt4, (0, 255, 0), 2)

                    # Display the lengths of the long and short axes
                    length = long_axis_length*2*offset
                    width = short_axis_length*2*offset

                    text = f"L: {length:.2f}, W: {width:.2f}"
                    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    text_position = (int(center[0] - text_size[0] / 2), int(center[1] + long_axis_length + 20))

                    xmin = int(box.data[0][0])
                    ymin = int(box.data[0][1])
                    xmax = int(box.data[0][2])
                    ymax = int(box.data[0][3])

                    box_track = [xmin, ymin, xmax, ymax]
                    # Bs=results_grade.boxes.data.cpu().numpy()
                    boxes_grade = bboxes

                    final_idx = match_track_with_detect(box_track, boxes_grade)
                    sku = boxes_grade[final_idx][5]

                    img = self.overlay(img, seg, colors[int(sku)], 0.4)

                    if xmin >= line_left_x:
                        cv2.putText(img, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    cv2.putText(img, f"Id {id}", (xmax-10, ymax-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, )

                    self.plot_one_box([xmin, ymin, xmax, ymax], img, colors[int(sku)], f'{cfg.global_detector.class_names[int(sku)]} {float(boxes_grade[final_idx][4]):.3}')

                    cv2.line(img, (line_right_x, 0), (line_right_x, 384), (0, 100, 255), 2)
                    cv2.line(img, (line_left_x, 0), (line_left_x, 384), (0, 100, 255), 2)

                    cv2.putText(img, "Record", (line_right_x, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(img, "Grade", (line_left_x, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    if xmin <= line_right_x:
                        if xmax >= line_right_x:
                            cv2.putText(img, f"Start record {id}", (xmin-10, int((ymax+ymin)/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        (100, 255, 100), 2)
                        # Record object properties
                        if id not in self.object_records:
                            self.object_records[id] = []
                            self.frame_counter[id] = 0
                        if self.frame_counter[id] % self.skip_frames == 0:
                            self.object_records[id].append({
                                # 'frame': cap.get(cv2.CAP_PROP_POS_FRAMES),
                                'frame': self.cnt_frame,
                                'class': cfg.global_detector.class_names[int(sku)],
                                "confidence": boxes_grade[final_idx][4],
                                'length': length,
                                'width': width
                            })

                    if xmax-20 >= line_left_x and xmin <= line_left_x:
                        cv2.putText(img, f"Start grade {id}", (xmin - 10, int((ymax + ymin) / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 2)
                        lengths = [record.get('length') for record in self.object_records[id]]
                        if lengths is None:
                            print("none")
                        widths = [record['width'] for record in self.object_records[id]]
                        grades = [record['class'] for record in self.object_records[id]]
                        t_confidences = [record['confidence'] for record in self.object_records[id]]
                        m_len = round(sum(lengths)/len(lengths),3)
                        m_wid = round(sum(widths)/len(widths), 3)
                        mid_len = round(np.median(np.array(lengths)),3)
                        mid_wid = round(np.median(np.array(widths)),3)
                        confidences = [tensor.item() for tensor in t_confidences]
                        f_grade_surface = combine_measurements(grades, confidences)

                        f_list = (id, m_len, mid_len, m_wid, mid_wid, f_grade_surface)
                        f_grade_sample= GetFinalGrade(f_list)

                        text = f"F_L: {m_len:.2f}, F_W: {m_wid:.2f}, {f_grade_sample}"

                        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        text_position = (int(center[0] - text_size[0] / 2), int(center[1] + long_axis_length + 20))
                        cv2.putText(img, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        key = cv2.waitKey(10)
                        if key & 0xFF == ord('s'):
                            save_data(self.object_records)
                        if f_grade_sample != cfg.sku_names[0]:
                            if id not in self.graded_sps:
                                #memory increasing
                                self.graded_sps[id] = {'id': id, 'm_len':m_len, 'mid_len':mid_len, 'm_wid':m_wid, 'mid_wid':mid_wid, 'f_grade_surface': f_grade_surface, 'f_grade_sample': f_grade_sample}
                            # del self.object_records[id]
        return img

    def draw_dashed_rectangle(self, img, pt1, pt2, color, thickness=3, dash_length=10):
        """Helper function to draw a dashed rectangle."""
        x1, y1 = min(pt1[0], pt2[0]), min(pt1[1], pt2[1])
        x2, y2 = max(pt1[0], pt2[0]), max(pt1[1], pt2[1])

        # Top edge
        for x in range(x1, x2, dash_length * 2):
            cv2.line(img, (x, y1), (min(x + dash_length, x2), y1), color, thickness)
        # Bottom edge
        for x in range(x1, x2, dash_length * 2):
            cv2.line(img, (x, y2), (min(x + dash_length, x2), y2), color, thickness)
        # Left edge
        for y in range(y1, y2, dash_length * 2):
            cv2.line(img, (x1, y), (x1, min(y + dash_length, y2)), color, thickness)
        # Right edge
        for y in range(y1, y2, dash_length * 2):
            cv2.line(img, (x2, y), (x2, min(y + dash_length, y2)), color, thickness)
            
    def plot_boxes_detection(self, results, resize_frame_for_display):
        if getattr(cfg, 'plot_boxes', True) is False:
            return resize_frame_for_display
        
        self.cnt_frame += 1
    
        w_detect, h_detect = cfg.detect_size_with_pad
        h_display, w_display = resize_frame_for_display.shape[:2]
        h_ratio = h_display / h_detect
        w_ratio = w_display / w_detect

        img = resize_frame_for_display
        if results is None:
            return img
        if type(results) is list:
            bboxes = results
        else:
            bboxes = results.boxes.data.cpu().numpy()

        # --- CALCULATE BASE SAFETY ZONE ---
        base_safety_radius = 0.0
        if hasattr(cfg, 'safety_zone_size') and hasattr(cfg, 'pixel_to_physical_space_resized'):
            if cfg.pixel_to_physical_space_resized > 0:
                base_safety_radius = cfg.safety_zone_size / cfg.pixel_to_physical_space_resized

        # --- 1. VISUALIZE THE LETTUCE TRACKER ZONES ---
        if getattr(cfg, 'lettuce_tracker_for_only_detection', False) and hasattr(cfg.global_detector, 'lettuce_tracker'):
            for tid, t_info in cfg.global_detector.lettuce_tracker.tracks.copy().items():
                l_box = t_info['box']
                l_xmin, l_ymin, l_xmax, l_ymax = map(int, l_box[:4])
                
                # Draw the tracked lettuce zone with thick dashed lines
                tracker_color = (255, 150, 0)
                
                self.draw_dashed_rectangle(
                    img, (l_xmin, l_ymin), (l_xmax, l_ymax), 
                    tracker_color, thickness=4, dash_length=15
                )
                
                # cv2.putText(img, f"Tracker Zone {tid}", (l_xmin, l_ymin - 8), 
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, tracker_color, 2, cv2.LINE_AA)
                
                # Plot the dynamic safety zone circle for the tracked crop
                if base_safety_radius > 0:
                    l_cx, l_cy = (l_xmin + l_xmax) / 2.0, (l_ymin + l_ymax) / 2.0
                    
                    # Calculate bounding box radius (half the diagonal)
                    crop_diag_half = np.hypot(l_xmax - l_xmin, l_ymax - l_ymin) / 2.0
                    dynamic_radius = int(round(base_safety_radius + crop_diag_half))
                    
                    cv2.circle(img, (int(l_cx), int(l_cy)), dynamic_radius, tracker_color, 2, cv2.LINE_AA)
        # ----------------------------------------------

        for box in bboxes:
            xmin, ymin, xmax, ymax = map(int, box[:4])
            
            # Universal extraction regardless of tracking status
            conf = int(round(box[-2], 2) * 100)
            sku = int(box[-1])

            # --- 2. FORMAT AND PLOT FLAGS ---
            is_crop = False
            
            if sku == cfg.filtered_sku_id:
                color = (255, 0, 0) # Turn suppressed weeds solid Red
                label = f"Fake Weed {conf}%"
            elif sku >= 0 and sku < len(getattr(cfg.global_detector, 'class_names', [])):
                color = cfg.sku_colors[sku]
                label = f'{conf}% {cfg.global_detector.class_names[sku]}'
                
                if cfg.global_detector.class_names[sku] in getattr(cfg, 'vegetable_names', []) or sku == getattr(cfg, 'vegetable_id', -1):
                    is_crop = True
            else:
                color = (120, 194, 126)
                label = f'{conf}% Target'

            self.plot_one_box([xmin, ymin, xmax, ymax], img, color, label)

            # Plot the dynamic safety zone circle for newly detected crops
            if is_crop and base_safety_radius > 0:
                cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
                
                # Calculate bounding box radius (half the diagonal)
                crop_diag_half = np.hypot(xmax - xmin, ymax - ymin) / 2.0
                dynamic_radius = int(round(base_safety_radius + crop_diag_half))
                
                cv2.circle(img, (int(cx), int(cy)), dynamic_radius, color, 2, cv2.LINE_AA)

        return img

    def plot_valve_location(self, img):
        # nozzle_center_locations = np.arange(start, end - range_per_valve//2, range_per_valve)
        # nozzle_center_locations += range_per_valve//2

        if cfg.is_pad_height:
            x_start_gui = cfg.nozzle_trigger_x_start
            x_end_gui = cfg.nozzle_trigger_x_end
        else:
            x_start_gui = cfg.detect_size_pad_half+cfg.nozzle_trigger_x_start
            x_end_gui = cfg.detect_size_pad_half+cfg.nozzle_trigger_x_end

        for i_valve, valve_location in enumerate(cfg.nozzle_center_locations):
            nozzle_idx = i_valve + 1
            if nozzle_idx < cfg.valid_valve_range[0]:
                continue
            if nozzle_idx > cfg.valid_valve_range[-1]:
                continue
            valve_location = round(valve_location)
            cv2.line(img, (0, valve_location), (50, valve_location), (255, 128, 0), 3)
            cv2.line(img, (x_start_gui, valve_location), (x_end_gui, valve_location), (0, 255, 0), 3)
            cv2.putText(img, str(nozzle_idx), (5, valve_location-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 128, 0), 2)
        return img
