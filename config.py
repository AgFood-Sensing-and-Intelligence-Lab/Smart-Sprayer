import json
import random
import numpy as np
from queue import SimpleQueue, Queue
import platform
import cv2
import sys
import os

sys_name = platform.system()

is_debug = False

# indoor or field
exp_type = 'field'      
if exp_type == 'indoor':
    detection_model_name = 'indoor_testing_best_tractor'
    with open('./model/catname2id_yolo_indoor.json') as f:
        catname2id_yolo = json.load(f)
else:
    # v26l_68epochs_800_data_2021_to_2025 v26l_68epochs_800_DINOv3_data_2021_to_2025
    # v26l_68epochs_800_08022026 v26m_68epochs_800_08022026 v26s_68epochs_800_08022026
    # v26l_68epochs_800_08042026 v26m_68epochs_800_08022026 v26s_68epochs_800_08042026
    detection_model_name = 'v26l_68epochs_800_08202026'
    with open('./model/catname2id_yolo.json') as f:
        catname2id_yolo = json.load(f)
        
YOLO = None

from detection.detection import Detector
from sensing.speed_monitor import SpeedMonitor
from sensing.relay_monitor import RelayMonitor
from im_operation.im_operation import ImageHandler
from camera.camera_handler import CameraHandler
from video.video_handler import VideoHandler
# from gui.MyWeedGUI import MainWindow
from custom_timer.timer import Timer

is_for_Mark_sprayer = False
is_for_Tractor_sprayer = False
is_for_FarmNG_sprayer_v3 = True

has_infrared_channel = False

vegetable_names = ['Crop', 'Radish', 'Turnip', 'Lettuce', 'Cabbage', 'Carrot', 'Broccoli', 'Onion', 'Potato']
# vegetable_names = []
vegetable_id = 44
weed_id = 29

# vimba opencv realsense JAI
camera_vendor = 'vimba'

"""
ref point (like valve tip screw) to nozzle orifice in physic space
"""
# valve_ref_pos_to_front_valve_distance = 17.0 + 3.0  # cm
valve_ref_pos_to_front_valve_distance = 0.0  # cm
if is_for_FarmNG_sprayer_v3:
    """
    actual space is slightly larger than 1 inch
    """
    # valve_ref_pos_to_front_valve_distance = 8.0  # cm
    # valve_ref_pos_to_front_valve_distance = 4.0  # cm
    # valve_ref_pos_to_front_valve_distance = 2.0  # cm
    # valve_ref_pos_to_front_valve_distance = 1.5  # cm
    # valve_ref_pos_to_front_valve_distance = 0.1  # cm
    valve_ref_pos_to_front_valve_distance = 2.0  # 3.6cm

detection_conf_t = 0.25 
detection_iou_t = 0.5 
spray_conf_t = 0.25
filter_out_obj_not_in_roi_conf_t = 0.25

if is_for_FarmNG_sprayer_v3:
    # detection_conf_t = 0.05 
    detection_conf_t = 0.25 
if is_for_Mark_sprayer:
    detection_conf_t = 0.25 
    detection_iou_t = 0.25 
if is_for_FarmNG_sprayer_v3:
    detection_conf_t = 0.05 
    detection_iou_t = 0.75 
    spray_conf_t = detection_conf_t

use_opening_duration_to_close_valve = True
valve_opening_duration_max = 10.0
valve_opening_duration_min = 3.0

"""
physical_speed = cfg.current_physical_speed
"""
if is_for_Mark_sprayer:
    valve_response_time = 0.02
    spray_touch_ground_time = 0.00
    current_physical_speed = 50.8   # cm/s, 15ft/min
    valve_opening_duration = 0.025  # s
    valve_opening_duration_max = 0.1
    valve_opening_duration_min = 0.02

if is_for_Tractor_sprayer:
    valve_response_time = 0.02
    spray_touch_ground_time = 0.00
    current_physical_speed = 50.8   # cm/s, 15ft/min
    valve_opening_duration = 0.025  # s

if is_for_FarmNG_sprayer_v3:
    valve_response_time = 0.02
    spray_touch_ground_time = 0.00
    # current_physical_speed = 25.4   # cm/s, 15ft/min
    current_physical_speed = 12.7   # cm/s, 15ft/min
    # valve_opening_duration = 0.35  # s
    valve_opening_duration = 0.5  # s
    
    valve_opening_duration_max = 0.9
    valve_opening_duration_min = 0.4

    # valve_opening_duration = 0.5  # s
    # valve_opening_duration_max = 0.8
    # valve_opening_duration_min = 0.3

only_binary_classes_crop_and_weed = False
if exp_type == 'indoor':
    only_binary_classes_crop_and_weed = False

filter_out_obj_for_reasons = True
filter_out_weeds_close_to_crops = True
filter_out_overlapped_plants= False
filter_out_overlapped_plants_old = False
filter_out_obj_not_in_roi = True
filter_out_by_object_not_green = False
filter_out_by_nearest_valve_idxes_in_process = False


"""
only show green masked image
"""
# global g_mask_frame
g_green_mask_frame = None
use_g_green_mask_frame = False and filter_out_by_object_not_green

use_physical_speed_from_wheel = False
# this still has bugs, since some weeds' bboxes enter the view may have diff speed  from the weeds are fully in the view
use_average_speed_from_all_weeds = False
use_speed_from_each_weed = True

nozzle_cover_range_vertical_direction = (300, 1770)
nozzle_cover_range_move_direction = (450, 550)  
if is_for_Mark_sprayer:
    nozzle_cover_range_vertical_direction = (310, 980) 
    # nozzle_cover_range_vertical_direction = (370, 1030) 

    nozzle_cover_range_move_direction = (0, 100)  #

if is_for_Tractor_sprayer:
    nozzle_cover_range_vertical_direction = (310, 980) 
    if exp_type == 'indoor':
        nozzle_cover_range_move_direction = (1250*650/800, 1250*700/800)  #
    else:
        nozzle_cover_range_move_direction = (1250*100/800, 1250*150/800)  #

if is_for_FarmNG_sprayer_v3:
    if camera_vendor == 'JAI':
        nozzle_cover_range_vertical_direction = (0, 2048) 
        nozzle_cover_range_move_direction = (1536*10/984, 1536*60/984)  #valve tip screw 
    elif camera_vendor == 'vimba':
        """
        spray on x=? on the ground
        #valve tip screw 
        """
        # nozzle_cover_range_vertical_direction = (0+2065*20/936, 2056-2065*20/936) 
        nozzle_cover_range_vertical_direction = (0, 2048) 
        # nozzle_cover_range_move_direction = (2464*170/1120, 2464*210/1120)  
        # nozzle_cover_range_move_direction = (2464*125/1120, 2464*175/1120)  

        # nozzle_cover_range_move_direction = (2464*150/1120, 2464*200/1120)  
        # nozzle_cover_range_move_direction = (2464*170/1120, 2464*200/1120)     
        nozzle_cover_range_move_direction = (2464*220/1120, 2464*250/1120)     
    elif camera_vendor == 'opencv':
        nozzle_cover_range_vertical_direction = (0, 2048) 
        # nozzle_cover_range_move_direction = (2464*170/1120, 2464*200/1120)     
        nozzle_cover_range_move_direction = (2464*220/1120, 2464*250/1120)     


camera_height = 33.02  # cm 13 in * 2.54
camera_to_sprayed_ground_distance = 17.78  # cm 7 in * 2.54 

open_duration_based_on_bbox_size = True
height_compensation_coeff = 1/(800.0*2)
valve_opening_duration_factor = 0.8
large_bbox_add_distance_before_nozzle_for_parallax = True
# additional_distance_factor = 0.5
# additional_distance_factor = 0.25
additional_distance_factor = 0.1

large_bbox_valve_long_opening_duration = True
if is_for_Mark_sprayer:
    open_duration_based_on_bbox_size = False
    if not open_duration_based_on_bbox_size:
        large_bbox_add_distance_before_nozzle_for_parallax = True
        large_bbox_max_x_size_t_1 = 125.7  # 125.7 /800 ~ 4 cm
        large_bbox_max_x_size_t_2 = 256.4
        # 2 4
        large_bbox_add_physical_distance_1 = 2  # cm
        large_bbox_add_physical_distance_2 = 4  # cm
        
        large_bbox_valve_long_opening_duration = False

        large_bbox_add_physical_distance_for_duration_1 = 1.5  # cm
        large_bbox_add_physical_distance_for_duration_2 = 3  # cm

if is_for_Mark_sprayer:
    # cm correct distances
    nozzle_distance_inbetween = {
        1: 13.5, 2: 9.5, 3: 5, 4: 0, 
        5: 13.5, 6: 9.5, 7: 5, 8: 0, 
        9: 13.5, 10: 9.5, 11: 5, 12: 0,
    }
    # nozzle_distance_inbetween = {
    #     1: 13.5, 2: 9, 3: 4.5, 4: 0, 
    #     5: 13.5, 6: 9, 7: 4.5, 8: 0, 
    #     9: 13.5, 10: 9, 11: 4.5, 12: 0,
    # }

if is_for_Tractor_sprayer:
    nozzle_distance_inbetween = {
        1: 2, 2: 12, 
        3: 2, 4: 12, 
        5: 2, 6: 12, 
        7: 2, 8: 12, 
        9: 2, 10: 12, 
        11: 2, 12: 12,
    }

    nozzle_distance_to_view = {}
    for valve_idx in nozzle_distance_inbetween:
        nozzle_distance_to_view[valve_idx] = nozzle_distance_inbetween[valve_idx] + valve_ref_pos_to_front_valve_distance  # add some extra

front_valv_to_back_valve_distance = -1  # cm

even_idx_is_close_to_camera = True
if is_for_FarmNG_sprayer_v3:
    front_valv_to_back_valve_distance = 10  # cm
    even_idx_is_close_to_camera = True
    nozzle_distance_inbetween = {
        2: 0, 1: front_valv_to_back_valve_distance, 
        4: 0, 3: front_valv_to_back_valve_distance, 
        6: 0, 5: front_valv_to_back_valve_distance, 
        8: 0, 7: front_valv_to_back_valve_distance, 
        10: 0, 9: front_valv_to_back_valve_distance, 
        12: 0, 11: front_valv_to_back_valve_distance,
        14: 0, 13: front_valv_to_back_valve_distance,
        16: 0, 15: front_valv_to_back_valve_distance,
        18: 0, 17: front_valv_to_back_valve_distance,
        20: 0, 19: front_valv_to_back_valve_distance,
        22: 0, 21: front_valv_to_back_valve_distance,
        24: 0, 23: front_valv_to_back_valve_distance,
    }
    nozzle_distance_to_view = {}
    for valve_idx in nozzle_distance_inbetween:
        nozzle_distance_to_view[valve_idx] = nozzle_distance_inbetween[valve_idx] + valve_ref_pos_to_front_valve_distance  # add some extra

if not is_for_Mark_sprayer and not is_for_Tractor_sprayer and not is_for_FarmNG_sprayer_v3:
    assert False, "only support registered sprayers"
valve_ref_pos_to_back_valve_distance = valve_ref_pos_to_front_valve_distance + front_valv_to_back_valve_distance

assert front_valv_to_back_valve_distance > 0

show_step_by_step = False

valve_idx_to_text_dict = {
    1: '1',
    2: '2',
    3: '3',
    4: '4',
    5: '5',
    6: '6',
    7: '7',
    8: '8',
    9: '9',
    10: '10',
    11: '11',
    12: '12',
    13: '1',
    14: '2',
    15: '3',
    16: '4',
    17: '5',
    18: '6',
    19: '7',
    20: '8',
    21: '9',
    22: '10',
    23: '11',
    24: '12',
}
if is_for_Mark_sprayer or is_for_Tractor_sprayer:
    valve_idx_to_text_dict = {
        1: '1',
        2: '2',
        3: '3',
        4: '4',
        5: '5',
        6: '6',
        7: '7',
        8: '8',
        9: '9',
        10: '10',
        11: '11',
        12: '12',
    }
if is_for_FarmNG_sprayer_v3:
    valve_idx_to_text_dict = {
        1: '1',
        2: '1',
        3: '2',
        4: '2',
        5: '3',
        6: '3',
        7: '4',
        8: '4',
        9: '5',
        10: '5',
        11: '6',
        12: '6',
        13: '7',
        14: '7',
        15: '8',
        16: '8',
        17: '9',
        18: '9',
        19: '10',
        20: '10',
        21: '11',
        22: '11',
        23: '12',
        24: '12',
    }

valve_num = len(valve_idx_to_text_dict)


"""
Camera/Display/Detection resolutions
"""
if camera_vendor == 'opencv':
    original_size = {'x':2464, 'y':2056}
    original_size_ROI = None
    if is_for_Mark_sprayer:
        original_size = {'x':1250, 'y':1250}
        # original_size = {'x':1400, 'y':1400}
    if is_for_Tractor_sprayer:
        original_size = {'x':1250, 'y':1250}
        # original_size = {'x':1500, 'y':904}
    if is_for_FarmNG_sprayer_v3:
        # original_size = {'x':1536, 'y':2048}
        original_size = {'x':2464, 'y':2056}

    # original_size_ROI = {'x':0, 'x1':2056, 'y':0, 'y1':2056}
elif camera_vendor == 'JAI':
    original_size = {'x':2048, 'y':1536}
    original_size_ROI = None
    assert is_for_FarmNG_sprayer_v3

elif camera_vendor == 'vimba':
    original_size = {'x':2464, 'y':2056}
    # original_size_ROI = {'x':0, 'x1':2056, 'y':0, 'y1':2056}  # detection degraded
    original_size_ROI = {'x':0, 'x1':2464, 'y':0, 'y1':2056}
    if is_for_Mark_sprayer:
        # original_size_ROI = {'x':1500, 'x1':1500+904, 'y':700, 'y1':700+904}
        # original_size_ROI = {'x':1150, 'x1':2400, 'y':750, 'y1':1580}  # 1250 840
        original_size_ROI = {'x':1150, 'x1':2400, 'y':545, 'y1':1785}  # 1250 1250 (750-205, 1580+205)

        # original_size_ROI = {'x':550, 'x1':1950, 'y':200, 'y1':1600}
    if is_for_Tractor_sprayer:
        assert False, "need to set original_size_ROI for Tractor sprayer if needed"

if original_size_ROI is not None:
    original_size['x'] = original_size_ROI['x1']-original_size_ROI['x']
    original_size['y'] = original_size_ROI['y1']-original_size_ROI['y']

"""
default x 0-2464, y 0-2056
None
ROTATE_90_CLOCKWISE  
ROTATE_90_COUNTERCLOCKWISE
ROTATE_180

add some extra
y 330
x 525 -> 560
(330, 1750)
nozzle_cover_range_vertical_direction = (300, 1770)
nozzle_cover_range_move_direction = (450, 550)  

Mark's sprayer with 2 valves. 3 inch highet
x 1500, w 904
y 700, h 904
    nozzle_cover_range_vertical_direction = (200, 660) 
    nozzle_cover_range_move_direction = (100, 150)
"""

if camera_vendor == 'opencv':
    frame_need_rotate_angle = None
    if is_for_Tractor_sprayer:
        if exp_type == 'indoor':
            frame_need_rotate_angle = cv2.ROTATE_180
elif camera_vendor == 'JAI':
    frame_need_rotate_angle = None
    if is_for_Tractor_sprayer:
        if exp_type == 'indoor':
            frame_need_rotate_angle = cv2.ROTATE_180
    if is_for_FarmNG_sprayer_v3:
        frame_need_rotate_angle = cv2.ROTATE_90_COUNTERCLOCKWISE
else:
    frame_need_rotate_angle = None
    if is_for_Mark_sprayer:
        frame_need_rotate_angle = None

        frame_need_rotate_angle = cv2.ROTATE_90_COUNTERCLOCKWISE
    elif is_for_FarmNG_sprayer_v3:
        frame_need_rotate_angle = cv2.ROTATE_180

if frame_need_rotate_angle is not None:
    if frame_need_rotate_angle == cv2.ROTATE_90_CLOCKWISE or frame_need_rotate_angle == cv2.ROTATE_90_COUNTERCLOCKWISE:
        original_size['x'], original_size['y'] = original_size['y'], original_size['x']
        # nozzle_cover_range_vertical_direction = (original_size['y']-nozzle_cover_range_vertical_direction[1], original_size['y']-nozzle_cover_range_vertical_direction[0])
    elif frame_need_rotate_angle == cv2.ROTATE_180:
        nozzle_cover_range_vertical_direction = (original_size['y']-nozzle_cover_range_vertical_direction[1], original_size['y']-nozzle_cover_range_vertical_direction[0])
        # nozzle_cover_range_move_direction = (original_size['x']-nozzle_cover_range_move_direction[1], original_size['x']-nozzle_cover_range_move_direction[0])
    else:
        assert False, "not support other rotation"

target_detection_size = 800
if is_for_Mark_sprayer:
    target_detection_size = 800
if is_for_Tractor_sprayer:
    target_detection_size = 800
if is_for_FarmNG_sprayer_v3:
    target_detection_size = 800

is_pad_height = False
if original_size['x'] > original_size['y']:
    detect_size_no_pad = (target_detection_size, round(original_size['y'] * target_detection_size // original_size['x']))
    is_pad_height = True
else:
    detect_size_no_pad = (round(original_size['x'] * target_detection_size // original_size['y']), target_detection_size)
    is_pad_height = False
detect_size_with_pad = (target_detection_size, target_detection_size)

if camera_vendor == 'vimba':
    detect_size_with_pad = detect_size_no_pad
    is_pad_height = False

if detect_size_with_pad!= detect_size_no_pad:
    detect_size_pad_gap = max(detect_size_no_pad) - min(detect_size_no_pad)
    detect_size_pad_half = detect_size_pad_gap // 2
else:
    detect_size_pad_half = 0

display_size = detect_size_with_pad
# assert detect_size_with_pad == detect_size_no_pad, "detect_size_with_pad should be the same as detect_size_no_pad, no padding needed" 

# if frame_need_rotate_angle is not None:
#     if frame_need_rotate_angle == cv2.ROTATE_90_CLOCKWISE or frame_need_rotate_angle == cv2.ROTATE_90_COUNTERCLOCKWISE:
#         pixel_to_physical_space_scale = (original_size['y'] / detect_size_no_pad[1])
#     elif frame_need_rotate_angle == cv2.ROTATE_180:
#         pixel_to_physical_space_scale = (original_size['x'] / detect_size_no_pad[0])
# else:
#     # the calculation of pixel_to_physical_space_scale is wrong?
#     pixel_to_physical_space_scale = (original_size['x'] / detect_size_no_pad[0])
"""
Pixel to physical space mapping

(10*2.54)/(1090-380)
valve_top_screw_position_x = 380
The image's distortion may be bad because the lens distortion on the edges ? No, only use central ROI of the camera

The plant has a height, and its  detected bounding box move in the pixel space is "faster" 
than the object with height=0 (ground) due to the projection of the camera.
"""
# nozzle_cover_range_move_direction
# the computed delay from back nozzles are too long (~2, most of them over 2s) compared to delay_time_run_13_to_24 (1.3)?
# pixel to physical mapping


# pixel_space_length = detect_size_no_pad[0]  # pixel
pixel_space_length = 445/1550*detect_size_no_pad[0]  # pixel
physical_space_length = 10*2.54  # cm
pixel_to_physical_space = 0.0358  # cm/pixel, not use for now
if is_for_Mark_sprayer:
    # 0.03238448844884489 cm/pixel
    pixel_space_length = 1212/1256*detect_size_no_pad[0]  # pixel
    physical_space_length = 20  # cm

    # 0.03181818181818182 cm/pixel
    # pixel_space_length = 1320/1400*detect_size_no_pad[0]  # pixel
    # physical_space_length = 24  # cm
if is_for_Tractor_sprayer:
    pixel_space_length = 1212/1256*detect_size_no_pad[0]  # pixel
    physical_space_length = 20  # cm
if is_for_FarmNG_sprayer_v3:
    if camera_vendor == 'JAI':
        pixel_space_length = 578/984*detect_size_no_pad[0]  # pixel
        physical_space_length = 20  # cm
    else: # vimba
        """
        6.9 cm/100 pixel
        ->
        7.25 cm/100 px
        """
        # pixel_space_length = 515/1120*detect_size_no_pad[0]  # pixel
        # physical_space_length = 25.4  # cm  
        pixel_space_length = 735/1120*detect_size_no_pad[0]  # pixel
        physical_space_length = 38.1  # 15 inches cm
"""
delay longer on a thread is not accurate
0.1 -> 0.01 longer = 0.11
1.2 -> ~0.04 longer = 1.6 
18 cm / 12cm/s = 1.5 s
It will open many valves at the moment when I close the software, why?
delay_time_run_1_to_12 = 0.1
delay_time_run_13_to_24 = 1.1
"""
delay_time_run_1_to_12=0.0
delay_time_run_13_to_24=0.0
if is_for_FarmNG_sprayer_v3:
    delay_time_run_1_to_12 = valve_ref_pos_to_front_valve_distance/current_physical_speed
    delay_time_run_13_to_24 = (valve_ref_pos_to_front_valve_distance+nozzle_distance_inbetween[2])/current_physical_speed

# pixel_to_physical_space_resized = pixel_to_physical_space * pixel_to_physical_space_scale  # for tractor,  frame_need_rotate_angle = cv2.ROTATE_90_COUNTERCLOCKWISE
"""
For higher plants, 1 pixel represent less physical distance than the ground, because the projection of the camera. 

"""
pixel_to_physical_space_resized = physical_space_length / pixel_space_length  # cm/pixel for tractor  


safety_zone_size = 5   # cm
safety_zone_size_pixel = round(safety_zone_size / pixel_to_physical_space_resized, 2)
safety_zone_size_adjusted_by_bbox_size = True
"""
Pixel to physical space nozzle idx mapping
"""
# Valve far
nozzle_mapping_odd = {k: v for k, v in zip(range(1, 25, 2), range(1, 13))}
# Valve close
nozzle_mapping_even = {k: v for k, v in zip(range(2, 26, 2), range(1, 13))}
nozzle_mapping_all = nozzle_mapping_odd.copy()  # make a copy to avoid modifying the original
nozzle_mapping_all.update(nozzle_mapping_even)
if is_for_Mark_sprayer:
    nozzle_mapping_all = {k: v for k, v in zip(range(1, 13, 1), range(12, 0, -1))}
if is_for_Tractor_sprayer:
    nozzle_mapping_all = {k: v for k, v in zip(range(1, 13, 1), range(12, 0, -1))}
if is_for_FarmNG_sprayer_v3:
    nozzle_mapping_all = nozzle_mapping_all
"""
Camera to Display/Detection space triger position mapping
"""
nozzle_trigger_x_start_tmp = int((nozzle_cover_range_move_direction[0]/original_size['x'])*detect_size_with_pad[0])
nozzle_trigger_x_end_tmp = int((nozzle_cover_range_move_direction[1]/original_size['x'])*detect_size_with_pad[0])
nozzle_trigger_x_start = min(nozzle_trigger_x_start_tmp, nozzle_trigger_x_end_tmp)
nozzle_trigger_x_end = max(nozzle_trigger_x_start_tmp, nozzle_trigger_x_end_tmp)

nozzle_range_y_start = int(nozzle_cover_range_vertical_direction[0]*detect_size_with_pad[1]/original_size['y'])
nozzle_range_y_end = int(nozzle_cover_range_vertical_direction[1]*detect_size_with_pad[1]/original_size['y'])
range_vertical = nozzle_range_y_end - nozzle_range_y_start
range_per_valve = range_vertical/valve_num
# nozzle_center_locations = np.linspace(nozzle_range_y_start+range_per_valve//2, nozzle_range_y_end-range_per_valve//2, valve_num)
nozzle_center_locations = np.linspace(nozzle_range_y_start, nozzle_range_y_end, valve_num)

valid_valve_num = valve_num
valid_valve_range = np.arange(1, valid_valve_num+1, 1)
do_center_row_only = False
if do_center_row_only:
    valve_num_center_row = 12
    valid_valve_num = valve_num_center_row
    valid_valve_range = np.arange(7, 18+1, 1)
    range_per_valve_center_row = range_vertical/valve_num_center_row
    nozzle_center_locations = np.array([float('inf')]*valve_num)
    nozzle_center_locations[(valid_valve_range-1).tolist()] = np.linspace(nozzle_range_y_start+range_per_valve_center_row//2, nozzle_range_y_end-range_per_valve_center_row//2, valve_num_center_row)
    nozzle_center_locations[:valid_valve_range[0]-1] = float('-inf')
nozzle_gap = nozzle_center_locations[valid_valve_range[1]] - nozzle_center_locations[valid_valve_range[0]]

frame_buffer_count = 2  # number of frames to buffer for camera stream
frame_buffer_count_origin = 2 

nearest_valve_idxes_manual_control = []
nearest_valve_idxes_in_process = []
nearest_valve_idxes_open_now = []
# Closed / Open  resultWidget
valve_states_dict = {}
for i_valve in range(1, valve_num+1):
    valve_states_dict[i_valve] = 'Closed'

"""
Detection/Tracking hyperparameters
"""
use_fake_bbox_in_track = False
if is_for_Mark_sprayer:
    use_fake_bbox_in_track = False
if is_for_Tractor_sprayer:
    use_fake_bbox_in_track = False
if is_for_FarmNG_sprayer_v3:
    use_fake_bbox_in_track = False

track_move_distance_min = 1

time_needed_max_allowed = 10
show_track_time_gap_max = time_needed_max_allowed
track_remove_time_gap = 10  # second
if is_for_Mark_sprayer:
    time_needed_max_allowed = 3
    # time_needed_max_allowed = 10
    show_track_time_gap_max = time_needed_max_allowed
    track_remove_time_gap = 5
if is_for_Tractor_sprayer:
    time_needed_max_allowed = 3
    show_track_time_gap_max = time_needed_max_allowed
    track_remove_time_gap = 5
if is_for_FarmNG_sprayer_v3:
    time_needed_max_allowed = 3
    show_track_time_gap_max = time_needed_max_allowed
    track_remove_time_gap = 5

move_direction_required = -1

# assert original_size['x'] >= original_size['y'], "only support landscape mode now"
target_closest_trigger_x = 0
if not is_pad_height:
    if move_direction_required < 0:
        target_closest_trigger_x = detect_size_pad_half+nozzle_trigger_x_end 
    else:
        target_closest_trigger_x = detect_size_pad_half+nozzle_trigger_x_start 

"""
Sync/Async -> camera/detection/valve control
"""
sync_valve_control = False
use_WorkerDetectionTrack = True
if exp_type == 'indoor':
    async_camera_stream = False 
    async_detection = True
    async_valve_control = True
    pattern_detection_by_morphology = False
    if pattern_detection_by_morphology:
        async_detection = False
else:
    async_camera_stream = False
    async_detection = True
    async_valve_control = True
    if camera_vendor == 'JAI':
        async_camera_stream = True 
        async_detection = True
        async_valve_control = True
    elif camera_vendor == 'vimba':
        async_camera_stream = True 
        async_detection = True
        async_valve_control = True

    pattern_detection_by_morphology = False

# for outdoor testing True False
only_detection = False and not pattern_detection_by_morphology
only_track = not only_detection

fast_tracker_for_only_detection = False and only_detection
lettuce_tracker_for_only_detection = False and only_detection
lettuce_tracker_for_only_detection_iou_t = 0.01
filtered_sku_id = 9999

use_track = only_track or fast_tracker_for_only_detection
allowed_instant_id_track_num = 200

largely_overlap_with_crop_iou = 0.75
if is_for_Mark_sprayer:
    largely_overlap_with_crop_iou = 0.25
if is_for_Tractor_sprayer:
    largely_overlap_with_crop_iou = 0.25
if is_for_FarmNG_sprayer_v3:
    largely_overlap_with_crop_iou = 0.25
"""
Valve control timing hyperparameters
"""
end_effector_no_delay = False
# pixel_to_physical_space_resized > 0 -> use_time_offset = True
use_time_offset = False
set_time_offset_zero = False
if not only_detection:
    use_time_offset = True
    set_time_offset_zero = False
max_time_offset = 5.0
max_time_to_close_valve_basic_t = max_time_offset


"""
Speed relevant

track_length_min
valve_opening_duration
largely_overlap_with_crop_iou
detection_results_history_len
"""
# int(global_detector.main_loop_fps//triggering_plot_frame_num_t_divider)
triggering_plot_frame_num_t = 10
triggering_plot_frame_num_t_divider = 5

track_trace_max_num = 100
track_schedule_table_each_trace_max_len = 1000
track_length_max = 20
track_length_min = 5
track_ref_frame_gap = 10
if is_for_Mark_sprayer:
    track_trace_max_num = 30
    # track_trace_max_num = 50
    track_schedule_table_each_trace_max_len = 200
    track_length_max = 20
    # track_length_min = 4
    # track_ref_frame_gap = 6
    track_length_min = 2
    track_ref_frame_gap = 3

    # track_length_min = 3
    # track_ref_frame_gap = 5
if is_for_Tractor_sprayer:
    track_trace_max_num = 30
    track_schedule_table_each_trace_max_len = 200
    track_length_max = 20
    track_length_min = 2
    track_ref_frame_gap = 3
if is_for_FarmNG_sprayer_v3:
    track_trace_max_num = 300
    track_schedule_table_each_trace_max_len = 200
    track_length_max = 20
    track_length_min = 2
    track_ref_frame_gap = 3
    
detection_results_history_len = 15

# <0.05, verified by Arduino send back signal
valve_response_time = 0.04
software_response_time = 0.01
spray_touch_ground_time = 0.01

# assert False

nearest_valve_idxes_in_process_time_gap = valve_opening_duration / 2

"""
Camera Exposure

>25000 indoor
<10000 outdoor no enclosure
~15000 outdoor with enclosure
"""
camera_time_expo = 1000
if is_for_Mark_sprayer:
    camera_time_expo = 1100
if is_for_Tractor_sprayer:
    camera_time_expo = 1100
if is_for_FarmNG_sprayer_v3:
    # camera_time_expo = 10000 # how to set exposure for JAI?
    # camera_time_expo = 15000 # how to set exposure for JAI?
    # camera_time_expo = 7500 # how to set exposure for JAI?
    camera_time_expo = 5000  # also add gain=7 dB? 

camera_cur_fps = 30

frame_result_for_signal_queue = Queue(maxsize=1)
frame_result_for_track_queue = Queue(maxsize=1)
frame_result_for_display_queue = Queue(maxsize=1)
track_future_info_queue = Queue(maxsize=1)

plot_labels = False
plot_improve_label_visibility = True
plot_boxes = True

camera_latest_frame = None
camera_latest_frame_timestamp = None
camera_latest_frame_idx = None

use_camera_timestamp = False
cam_start_streaming_time = None

"""
Save Image/Video
"""
save_video = True
save_origin_video = True
save_detect_video = False

save_frame = not save_video

frame_cnt_save = 0
frame_info_for_save_queue = Queue(maxsize=1)

save_newest_frame = True
if show_step_by_step:
    save_newest_frame = True

save_video_target_fps = 30  # the save for high resolution images of both origin and detected images can only run around 15 FPS
if is_for_Mark_sprayer:
    # save_video_target_fps = 60
    save_video_target_fps = 50
    # save_video_target_fps = 45
if is_for_Tractor_sprayer:
    save_video_target_fps = 30
if is_for_FarmNG_sprayer_v3:
    save_video_target_fps = 30

"""
QT old functions
"""
use_old_GUI_functions = False

"""
Global classes
"""
global_speed_monitor = SpeedMonitor()
global_relay_monitor = RelayMonitor()
global_image_handler = ImageHandler()
global_camera_handler = CameraHandler()
global_video_handler = VideoHandler()
global_detector = Detector()

global_timer = Timer()
global_window = None

# log_path = './log/log.txt'
log_path = None
def log_xalg(*info, log_path=log_path, show=True, end=None):
    from datetime import datetime
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S.%f")[:-3]
    if show:
        if end is not None:
            print(current_time, *info, end=end)
        else:
            print(current_time, *info)
    if log_path is not None:
        f_log = open(log_path, 'a')
        print(current_time, *info, file=f_log)
        f_log.close()


def list_dir(path, list_name, extension, return_names=False):
    import os
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            list_dir(file_path, list_name, extension)
        else:
            if file_path.endswith(extension):
                if return_names:
                    list_name.append(file)
                else:
                    list_name.append(file_path)
    try:
        list_name = sorted(list_name, key=lambda k: int(os.path.split(k)[1].split(extension)[0].split('_')[-1]))
    except Exception as e:
        print(e)
    return list_name


"""
Detection/Tracking Model selection

indoor_testing_best_tractor
"""

if sys_name == "Windows":
    detection_mode_path = rf'.\model\{detection_model_name}.pt'
    model_paths = {
        'track': detection_mode_path,
        'detection': detection_mode_path
    }
else:
    detection_mode_path = f'./model/{detection_model_name}.pt'
    model_paths = {
        'track': detection_mode_path,
        'detection': detection_mode_path
    }
log_xalg('------------------------------Start-----------------------------------')
log_xalg('detection_model_name:', detection_model_name)

"""
botsort: By default, ReID is turned off to minimize performance overhead. Turn it on to address the unstablity in plant detection?
bytetrack: No ReID
"""
track_alg = './trackers/bytetrack.yaml '

track_schedule_table = {}


sku_names = []
readable_name_to_sku_name_dict = {}
catname2id_yolo = sorted(catname2id_yolo.items(), key=lambda x: x[1])
for item in catname2id_yolo:
    sku_name = item[0]
    sku_names += [sku_name]
    readable_name_to_sku_name_dict[sku_name] = sku_name

readable_names = list(readable_name_to_sku_name_dict.keys())
sku_name_to_readable_name_dict = {}
for grade_name in sku_names:
    sku_name_to_readable_name_dict[readable_name_to_sku_name_dict[grade_name]] = grade_name

random.seed(0)
color_num = max(len(sku_names), 100)
sku_colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(color_num)]