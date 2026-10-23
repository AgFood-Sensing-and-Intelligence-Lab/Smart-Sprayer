# from vimba import *
# from vimba import Vimba, intersect_pixel_formats, OPENCV_PIXEL_FORMATS, COLOR_PIXEL_FORMATS, MONO_PIXEL_FORMATS
import vmbpy 
from queue import Queue
import numpy as np
from time import time, sleep
import math
from datetime import datetime, timedelta

from vmbpy import *
from vmbpy import intersect_pixel_formats, OPENCV_PIXEL_FORMATS, COLOR_PIXEL_FORMATS, MONO_PIXEL_FORMATS
Vimba=vmbpy.VmbSystem

opencv_display_format = PixelFormat.Bgr8

# First import the library
from camera_abstract import CameraBase
import cv2
from functools import partial
import os
import traceback
import sys
if __name__ != '__main__':
    try:
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        import config as cfg
    except Exception as e:
        cfg = None
        print(e)
        print(traceback.format_exc())


class Handler:
    def __init__(self):
        # self.display_queue = Queue(10)
        self.display_queue = Queue(cfg.frame_buffer_count_origin)

    def get_image(self):
        return self.display_queue.get(True)

    def __call__(self, cam: Camera, stream: Stream, frame: Frame):
        if frame.get_status() == FrameStatus.Complete:
            # print('{} acquired {}'.format(cam, frame), flush=True)
            # timestamp_sec = frame.timestamp / 1e9
            timestamp_sec = 1
            timestamp = cfg.cam_start_streaming_time + timedelta(seconds=timestamp_sec)
            # Convert frame if it is not already the correct format
            if frame.get_pixel_format() == opencv_display_format:
                display = frame
            else:
                # This creates a copy of the frame. The original `frame` object can be requeued
                # safely while `display` is used
                display = frame.convert_pixel_format(opencv_display_format)

            self.display_queue.put((display.as_opencv_image(), timestamp), True)

        cam.queue_frame(frame)

handler = Handler()

def frame_handler(queue, cam, frame):
    img = frame.as_numpy_ndarray()
    frame_np = cv2.rotate(frame_np, cv2.ROTATE_90_CLOCKWISE)
    frame_np = frame_np[:, :, ::-1]
    # img_rgb = cv2.cvtColor(img, cv2.COLOR_BAYER_RG2RGB)
    try:
        # Try to put the frame in the queue...
        queue.put_nowait(frame_np)
    # except queue.Full:
    except Exception as e:
        # If that fials (queue is full), just drop the frame
        # NB: You may want to handle this better...
        print('Dropped Frame')
    cam.queue_frame(frame)

class CameraVimba(CameraBase):
    def __init__(self) -> None:
        super().__init__()
        self.cam = None
        self.window = None

    def capture(self):
        success=True
        # with Vimba.get_instance() as vimba:
        #     with vimba.get_all_cameras()[0] as camera:
        # self.cam = camera
        frame_np, timestamp = handler.get_image()
        # frame = camera.get_frame()
        # frame_np = frame.as_numpy_ndarray()
        # frame_np = frame_np[:, :, ::-1]

        # no need, directly set in initilization
        # frame_np = frame_np[:, 300:, :]
        # print(frame_np.shape)
        
        # depth = frames.get_depth_frame()
        # if not depth:
        #     continue

        if __name__ != '__main__':
            if cfg.frame_need_rotate_angle is not None:
                frame_np = cv2.rotate(frame_np, cfg.frame_need_rotate_angle)
                """
                the Vimba coordinate origin is bottom-left, while the OpenCV coordinate origan is top-left
                rotate 90 will change the x, y to y,x
                """
        return success, frame_np, timestamp

    def end(self):
        self.is_streaming = False
        if hasattr(self, 'cap') and self.cap is not None:
            try:
                self.cap.release()
            except Exception as e:
                print(f"Error releasing cap: {e}")

    def config(self):
        with Vimba.get_instance() as vimba:
            with vimba.get_all_cameras()[0] as camera:
                camera.ExposureAuto.set(self.ExpState)
                # width_feature = camera.getFeatureByName('Width')
                # print(width_feature)

    def camera_streaming(self, frame_queue):
        self.is_streaming = True
        print("streaming started")
        if __name__ != '__main__':
            cfg.global_window.statusBar().showMessage("streaming started")
            buffer_count=cfg.frame_buffer_count
        else:
            buffer_count=10
        frame_idx = 0

        with Vimba.get_instance() as vimba:
            with vimba.get_all_cameras()[0] as cam:
                cfg.cam_start_streaming_time = datetime.now()
                cam.start_streaming(handler=handler, buffer_count=buffer_count)
                while self.is_streaming:
                    if not hasattr(self, '_fps_frame_count'):
                        self._fps_frame_count = 0
                        self._fps_last_time = time()
                        self._fps_value = 0

                    try:
                        res, frame_np, timestamp = self.capture()
                    except Exception as e:
                        print(e)

                    # if frame_idx % 300 == 0:
                    #     msg = 'camera frame rate:' + str(np.round(self._fps_value, 2)) + ' '+ str(frame_np.shape)
                    #     cfg.log_xalg(msg)
                        # f=camera.AcquireSingleImage()

                    # cfg.global_window.statusBar().showMessage(msg)

                    # frame = frame.as_opencv_image()
                    # im = Image.fromarray(frame)
                    # img = ImageTk.PhotoImage(im)

                    if frame_np is not None:
                        if cfg.use_camera_timestamp:
                            frame_timestamp = timestamp
                        else:
                            frame_timestamp = datetime.now()
                        cfg.camera_latest_frame = frame_np
                        cfg.camera_latest_frame_timestamp =frame_timestamp
                        cfg.camera_latest_frame_idx = frame_idx
                        # frame_queue.put((frame_np, frame_timestamp, frame_idx))  # put the capture image into queue
                        frame_idx += 1
                        frame_info = (frame_np, frame_timestamp, frame_idx)
                        try:
                            frame_queue.put_nowait(frame_info)  # put the capture image into queue
                        # except frame_queue.Full:
                        except Exception as e:
                            try:
                                _ = frame_queue.get_nowait()  # drop oldest
                                with frame_queue.mutex:
                                    frame_queue.queue.clear()
                                    # half = math.floor(len(queue.queue) / 2)
                                    # for _ in range(half):
                                    #     queue.queue.popleft()  # Remove from the front (FIFO)
                            except frame_queue.Empty:
                                pass
                            frame_queue.put_nowait(frame_info)

                    self._fps_frame_count += 1
                    now = time()
                    if now - self._fps_last_time >= 1.0:
                        self._fps_value = np.round(self._fps_frame_count / (now - self._fps_last_time), 2)
                        self._fps_last_time = now
                        self._fps_frame_count = 0
                    cfg.global_camera_handler.camera_fps = self._fps_value
        print("streaming stopped")
        if __name__ != '__main__':
            cfg.global_window.statusBar().showMessage("streaming stopped")

    def show_camera(self):
        pass

    def set_nearest_value(self, cam: Camera, feat_name: str, feat_value: int):
        # Helper function that tries to set a given value. If setting of the initial value failed
        # it calculates the nearest valid value and sets the result. This function is intended to
        # be used with Height and Width Features because not all Cameras allow the same values
        # for height and width.
        feat = cam.get_feature_by_name(feat_name)

        try:
            feat.set(feat_value)

        except VmbFeatureError:
            min_, max_ = feat.get_range()
            inc = feat.get_increment()

            if feat_value <= min_:
                val = min_

            elif feat_value >= max_:
                val = max_

            else:
                val = (((feat_value - min_) // inc) * inc) + min_

            feat.set(val)

            msg = ('Camera {}: Failed to set value of Feature \'{}\' to \'{}\': '
                'Using nearest valid value \'{}\'. Note that, this causes resizing '
                'during processing, reducing the frame rate.')
            Log.get_instance().info(msg.format(cam.get_id(), feat_name, feat_value, val))
            
    def setup_camera(self, cam=None):
        window = self.window
        if cam is not None:
            # cam.set_pixel_format(PixelFormat.BayerRG8)
            # cam.ExposureTimeAbs.set(10000)
            # cam.BalanceWhiteAuto.set('Off')
            # cam.Gain.set(0)
            # cam.AcquisitionMode.set('Continuous')
            # cam.GainAuto.set('Off')
            # # NB: Following adjusted for my Manta G-033C
            # cam.Height.set(492)
            # cam.Width.set(656)
            # Enable auto exposure time setting if camera supports it
            try:
                """
                    These settings cannot be supported by yhis vimba camera:
                    1800 U-507c-01J5H
                    ExposureAutoOutliers
                    Range: [0 to 1000]
                    Default: 0
                    Unit: 0.01% i.e. 1000 = 10%
                    ExposureAutoOutliers is the percentage of pixels on the upper bound of the
                    image brightness distribution graph that are ignored by the ExposureAuto
                    algorithm.
                    
                    ExposureAutoAdjustTol
                    Range: [0 to 50]
                    Default: 5

                    ExposureAutoAlg
                    Mean / FitRange

                    ExposureAutoMax
                    Default: 500000

                    ExposureAutoMin
                    Default: Camera dependent

                    ExposureAutoRate
                    Range: [1 to 100]
                    Default: 100

                    ExposureAutoTarget
                    Range: [0 to 100]
                    Default: 50

                    ExposureMode
                    Auto / Manual / PieceWiseLinearHDR
                    ExposureValue / ExposureValuePWL1 / ExposureValuePWL2
                    ThresholdPWL1 / ThresholdPWL2
                    ThresholdPWL2 < ThresholdPWL1
                    Range: [0 to 63]
                    Default: 63
                    0 = no light capacity, 63 = full pixel light capacity.
                """

                # cam.ExposureAuto.set('Off')
                # cam.ExposureAuto.set('On')
                exposure_time = cam.ExposureTime

                time_expo = exposure_time.get()
                print('exposure_time:', time_expo)
                inc = exposure_time.get_increment()
                # exposure_time.set(time + inc)
                # cam.ExposureAutoTarget.set(30)
                # cam.ExposureAuto.set('Continuous')
                if __name__ != '__main__':
                    exposure_time.set(cfg.camera_time_expo)
                else:
                    exposure_time.set(10000)
                # self.set_nearest_value(cam, 'Height', cfg.original_size['y'])
                # self.set_nearest_value(cam, 'Width', cfg.original_size['x']])
                """
                # ground for sprayer
                # (2464, 2056, 3)
                """
                # ground for collection data
                if cfg.original_size_ROI is not None:
                    self.set_nearest_value(cam, 'OffsetY', cfg.original_size_ROI['y'])
                    self.set_nearest_value(cam, 'Height', cfg.original_size_ROI['y1']-cfg.original_size_ROI['y'])

                    self.set_nearest_value(cam, 'OffsetX', cfg.original_size_ROI['x'])
                    self.set_nearest_value(cam, 'Width', cfg.original_size_ROI['x1']-cfg.original_size_ROI['x'])

                # soil bin 
                # self.set_nearest_value(cam, 'Width', 2264)
                # self.set_nearest_value(cam, 'OffsetX', 200)

                # AcquisitionFrameRate = cam.AcquisitionFrameRate
                # print('AcquisitionFrameRate:', AcquisitionFrameRate)
                # TriggerSelector = cam.get_feature_by_name("TriggerSelector")
                # print('TriggerSelector:', TriggerSelector)
                # TriggerMode = cam.get_feature_by_name("TriggerMode")
                # TriggerSource = cam.get_feature_by_name("TriggerSource")
                # print('TriggerMode:', TriggerMode)
                # # feature = cam.get_feature_by_name("AcquisitionFrameRate")
                # AcquisitionFrameRate.set(30) #specifies 30FPS
                # # set the other features TriggerSelector and TriggerMode
                # TriggerSelector.set("FrameStart")
                # TriggerMode.set("Off")
                # FixedRate=30
                # TriggerSource.set(FixedRate)
            except Exception as e:
                # except (AttributeError, VimbaFeatureError):
                print(e)
                pass

            # try:
            #     cam.GevSCPSPacketSize = 9000  
            #     cam.AcquisitionFrameRateEnable = True
            #     cam.AcquisitionFrameRate = 30.0 
            # except Exception as e:
            #     # except (AttributeError, VimbaFeatureError):
            #     print(e)
            #     pass

            # Enable white balancing if camera supports it
            try:
                cam.BalanceWhiteAuto.set('Continuous')
            except Exception as e:
                # except (AttributeError, VimbaFeatureError):
                print(e)
                pass
            # Try to adjust GeV packet size. This Feature is only available for GigE - Cameras.
            try:
                stream = cam.get_streams()[0]
                stream.GVSPAdjustPacketSize.run()
                while not stream.GVSPAdjustPacketSize.is_done():
                    pass
            # except (AttributeError, VmbFeatureError):
            except Exception as e:
                pass

            self.setup_pixel_format(cam, window)
            # Query available, open_cv compatible pixel formats
            # prefer color formats over monochrome formats
            # try:
            #     cv_fmts = intersect_pixel_formats(cam.get_pixel_formats(), OPENCV_PIXEL_FORMATS)
            #     color_fmts = intersect_pixel_formats(cv_fmts, COLOR_PIXEL_FORMATS)

            #     if color_fmts:
            #         cam.set_pixel_format(color_fmts[0])
            #         print(f'Camera set_pixel_format: {color_fmts[0]}.')
            #         if window:
            #             window.statusBar().showMessage(f'Camera set_pixel_format: {color_fmts[0]}.')
            #     else:
            #         mono_fmts = intersect_pixel_formats(cv_fmts, MONO_PIXEL_FORMATS)

            #         if mono_fmts:
            #             cam.set_pixel_format(mono_fmts[0])
            #         else:
            #             if window:
            #                 window.statusBar().showMessage('Camera does not support a OpenCV compatible format natively. Abort.')
            #     success, img = self.capture()
            # except Exception as e:
            #     print(e)
            #     pass


    def setup_pixel_format(self, cam, window):
        # Query available pixel formats. Prefer color formats over monochrome formats
        cam_formats = cam.get_pixel_formats()
        cam_color_formats = intersect_pixel_formats(cam_formats, COLOR_PIXEL_FORMATS)
        convertible_color_formats = tuple(f for f in cam_color_formats
                                        if opencv_display_format in f.get_convertible_formats())

        cam_mono_formats = intersect_pixel_formats(cam_formats, MONO_PIXEL_FORMATS)
        convertible_mono_formats = tuple(f for f in cam_mono_formats
                                        if opencv_display_format in f.get_convertible_formats())

        # if OpenCV compatible color format is supported directly, use that
        if opencv_display_format in cam_formats:
            cam.set_pixel_format(opencv_display_format)

        # else if existing color format can be converted to OpenCV format do that
        elif convertible_color_formats:
            cam.set_pixel_format(convertible_color_formats[0])

        # fall back to a mono format that can be converted
        elif convertible_mono_formats:
            cam.set_pixel_format(convertible_mono_formats[0])

        else:
            if window:
                # abort('Camera does not support an OpenCV compatible format. Abort.')
                window.statusBar().showMessage('Camera does not support an OpenCV compatible format. Abort.')


    def check_camera(self):
        window = None
        if __name__ != '__main__':
            if cfg is not None:
                self.window = cfg.global_window
                window = self.window
        with Vimba.get_instance() as vimba:
            # interfaces = vimba.get_all_interfaces()
            # for interface in interfaces:
            #     for feat in interface.get_all_features():
            #         print(feat)
            with vimba.get_all_cameras()[0] as camera:
                self.setup_camera(camera)
                if window is not None:
                    if not camera:
                        window.statusBar().showMessage('No accessible cameras. Abort.')
                    else:
                        window.statusBar().showMessage("Camera is ready for access")
                    # self.cam = cam

    def set_exposure(self, window):
        if window:
            if window.Exposure_offBox.isChecked():
                print("off")
                window.ExpState = "Off"
            elif window.Exposure_onceBox.isChecked():
                window.ExpState = "Once"
            elif window.Exposure_continueBox.isChecked():
                print("set exposure continues")
                window.ExpState = "Continuous"
            else:
                pass


def demo():
    import numpy as np
    import cv2
    camera = CameraVimba()
    
    camera.check_camera()
    camera.setup_camera()
    with Vimba.get_instance() as vimba:
        with vimba.get_all_cameras()[0] as cam:
            cam.start_streaming(handler=handler, buffer_count=10)
            idx = 0
            while True:
                start_time_tot = time()
                res, color = camera.capture()
                frame_np = cv2.rotate(color, cv2.ROTATE_180)
                # frame_np = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
                resize_frame_for_display = cv2.resize(frame_np, (800, 800))
                cv2.imshow('Camera Feed', resize_frame_for_display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                cv2.namedWindow('Camera Feed', 1)
                # cv2.resizeWindow('Camera Feed', 800, 600)
                end_time_tot = time()
                cur_tot_fps = np.round(1.0 / (end_time_tot - start_time_tot), 2)
                if idx% 300 == 0:
                    print('camera fps:', cur_tot_fps)
                idx += 1


def main():
    demo()


if __name__ == '__main__':
    main()

# import numpy as np
# depth = frames.get_depth_frame()
# depth_data = depth.as_frame().get_data()
# np_image = np.asanyarray(depth_data)
