import os
import datetime
from PyQt5.QtCore import QCoreApplication, QObject, QRunnable, QThread, QThreadPool, pyqtSignal, QTimer, QRect
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QVBoxLayout, QSlider, QLabel, QFileDialog, QMainWindow, QButtonGroup, QApplication
from PyQt5.QtGui import QImage, QPixmap
import threading
import config as cfg
import cv2
import numpy as np
from time import time, sleep
import traceback


class ImageHandler:
    def __init__(self) -> None:
        self.mapped_rect = QRect()
        self.imgStatus = False  
        self.img_path_detect = []
        self.adjusted_image = None
        self.saturation_factor = 1.00
        self.hue_shift = 0
        self.value_factor = 1.00
        self.make_resize = True
        self.scaled_size = cfg.detect_size_with_pad
        self.original_size = cfg.original_size
        self.is_streaming = False

        self.is_new_video = True
        self.video_start = True
        self.video_origin = None
        self.video_origin_infrared = None
        self.video_detect = None
        self.last_frame_time = None

    def create_save_path(self):
        if cfg.global_window.save_check.isChecked():
            nowTime = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
            base_path = os.getcwd()
            if cfg.sys_name == "Windows":
                sep = '\\'
            else:
                sep = '/'
            pathD = os.path.join(base_path, f'result{sep}DetectedImages{sep}{nowTime}')
            pathJ = os.path.join(base_path, f'result{sep}json{sep}{nowTime}')
            pathO = os.path.join(base_path, f'result{sep}OriginalImages{sep}{nowTime}')
            pathV = os.path.join(base_path, f'result{sep}Video{sep}{nowTime}')
            if not os.path.exists(pathD):
                os.makedirs(pathD)
            if not os.path.exists(pathJ):
                os.makedirs(pathJ)
            if not os.path.exists(pathO):
                os.makedirs(pathO)
            if not os.path.exists(pathV):
                os.makedirs(pathV)
            print("save pathes created")
            self.img_path_origin = pathO
            self.img_path_detect = pathD
            self.json_save_dir = pathJ
            self.video_path_origin = pathV
            self.video_path_detect = pathV
            cfg.global_window.Inum = 1

    def resize_check(self, c_frame):
        if self.make_resize:
            mapped_rect = self.mapped_rect
            scale_factor_x = self.original_size['x'] / self.scaled_size[0]
            scale_factor_y = self.original_size['y'] / self.scaled_size[1]
            original_rect = QRect(
                int(mapped_rect.x() * scale_factor_x),
                int(mapped_rect.y() * scale_factor_y),
                int(mapped_rect.width() * scale_factor_x),
                int(mapped_rect.height() * scale_factor_y),
            )
            c_frame = c_frame[original_rect.y():original_rect.y() + original_rect.height(),
                                original_rect.x():original_rect.x() + original_rect.width()]
        return c_frame
    
    def finalize(self):
        if cfg.save_video:
            self.is_new_video = True
            if self.video_origin is not None:
                self.video_origin.release()
            if self.video_origin_infrared is not None:
                self.video_origin_infrared.release()
            if self.video_detect is not None:
                self.video_detect.release()
            self.video_origin = None
            self.video_origin_infrared = None
            self.video_detect = None

    def run_save_thread(self):
        t1 = threading.Thread(target=self.save_video_loop, args=(), daemon=True)  
        t1.start()  

    def save_video_loop(self):
        try:
            while True:
                # sleep(0.001)
                if cfg.global_window.save_check.isChecked():
                    try:
                        frame_info_for_save_dict = cfg.frame_info_for_save_queue.get(timeout=0.01)  # wait briefly for new frame
                    except Exception as e:
                        continue 
                    if cfg.save_origin_video:
                        frame_origin_for_save = frame_info_for_save_dict['frame_origin_for_save']
                    else:
                        frame_origin_for_save = None
                    if cfg.save_detect_video:
                        frame_detected_for_save = frame_info_for_save_dict['frame_detected_for_save']
                    else:
                        frame_detected_for_save = None
                    # results_save = frame_info_for_save_dict['results_save'] 
                    # frame_cnt_save = frame_info_for_save_dict['frame_cnt_save'] 
                    cfg.global_image_handler.save_frame(frame_origin=frame_origin_for_save, frame_detected=frame_detected_for_save, results=None)
                else:
                    cfg.global_image_handler.finalize()
                    cfg.frame_cnt_save = 0
                    sleep(0.25)
        except Exception as e:
            print(traceback.format_exc())
            print(e)

    def save_frame(self, frame_origin=None, frame_detected=None, results=None):
        if frame_detected is not None:
            frame_detected_bgr = frame_detected[:, :, ::-1]
        else:
            frame_detected_bgr = None
        if cfg.save_frame:
            self.save_frame_by_interval(frame_origin=frame_origin, frame_detected=frame_detected_bgr, results=results)
        if cfg.save_video:
            self.save_video(frame_origin=frame_origin, frame_detected=frame_detected_bgr, results=results)

    def save_video(self, frame_origin=None, frame_detected=None, results=None):
        current_time = cfg.global_timer.get_cur_time()
        if self.last_frame_time is not None:
            time_gap = cfg.global_timer.calculate_time_gap(current_time, self.last_frame_time)
        self.last_frame_time = current_time
        # if time_gap > 1:
        #     self.is_new_video = True

        if self.is_new_video:
            if cfg.sys_name == "Windows":
                fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
                file_ext = '.mp4'
                if cfg.save_origin_video:
                    h_origin, w_origin = frame_origin.shape[:2]
                    output_filename_o = os.path.join(self.video_path_origin,'video_origin_'+current_time+file_ext)
                    self.video_origin = cv2.VideoWriter(output_filename_o, fourcc, fps=cfg.save_video_target_fps, frameSize=(w_origin, h_origin))
                if cfg.has_infrared_channel:
                    fourcc_1_channel = cv2.VideoWriter_fourcc(*'MJPG') 
                    file_ext_1_channel = '.avi'
                    h_origin, w_origin = frame_origin.shape[:2]
                    output_filename_infrared = os.path.join(self.video_path_origin,'video_origin_infrared'+current_time+file_ext)
                    self.video_origin_infrared = cv2.VideoWriter(output_filename_infrared, fourcc, fps=cfg.save_video_target_fps, frameSize=(w_origin, h_origin))
                if cfg.save_detect_video:
                    h_detect, w_detect = frame_detected.shape[:2]
                    output_filename_d = os.path.join(self.video_path_origin,'video_detect_'+current_time+file_ext)
                    self.video_detect = cv2.VideoWriter(output_filename_d, fourcc, fps=cfg.save_video_target_fps, frameSize=(w_detect, h_detect))
            else:
                assert False, "not support now"
                # jetson
                # fourcc = cv2.VideoWriter_fourcc(*'MJPG') 
                # file_ext = '.avi'
                file_ext = '.mp4'
                output_filename_o = os.path.join(self.video_path_origin,'video_origin_'+current_time+file_ext)
                output_filename_d = os.path.join(self.video_path_origin,'video_detect_'+current_time+file_ext)
                gst_out_o = (
                    f'appsrc ! videoconvert ! '
                    f'nvv4l2h264enc bitrate=4000000 ! h264parse ! '
                    f'queue ! qtmux ! filesink location={output_filename_o} sync=false'
                )
                gst_out_d = (
                    f'appsrc ! videoconvert ! '
                    f'nvv4l2h264enc bitrate=4000000 ! h264parse ! '
                    f'queue ! qtmux ! filesink location={output_filename_d} sync=false'
                )
                # self.video_origin = cv2.VideoWriter(gst_out_o, cv2.CAP_GSTREAMER, 0, cfg.save_video_target_fps, (w_origin, h_origin))
                # self.video_detect = cv2.VideoWriter(gst_out_d, cv2.CAP_GSTREAMER, 0, cfg.save_video_target_fps, (w_detect, h_detect))

                fourcc = cv2.VideoWriter_fourcc(*'XVID') 
                file_ext = '.avi'
                output_filename_o = os.path.join(self.video_path_origin,'video_origin_'+current_time+file_ext)
                output_filename_d = os.path.join(self.video_path_origin,'video_detect_'+current_time+file_ext)
                # self.video_origin = cv2.VideoWriter(output_filename_o, fourcc, fps=cfg.save_video_target_fps, frameSize=(800, 800))
                if cfg.save_origin_video:
                    h_origin, w_origin = frame_origin.shape[:2]
                    self.video_origin = cv2.VideoWriter(output_filename_o, fourcc, fps=cfg.save_video_target_fps, frameSize=(w_origin, h_origin))
                if cfg.save_detect_video:
                    h_detect, w_detect = frame_detected.shape[:2]
                    self.video_detect = cv2.VideoWriter(output_filename_d, fourcc, fps=cfg.save_video_target_fps, frameSize=(w_detect, h_detect))
            # XVID -> avi, mp4v -> mp4
            self.is_new_video = False
        if cfg.save_origin_video:
            if cfg.has_infrared_channel:
                frame_origin_RGB = frame_origin[:, :, :3]
                frame_origin_infrared = frame_origin[:, :, -1]
                frame_origin_infrared = cv2.cvtColor(frame_origin_infrared, cv2.COLOR_GRAY2BGR)
                if self.video_origin is not None:
                    self.video_origin.write(frame_origin_RGB)
                if self.video_origin_infrared is not None:
                    self.video_origin_infrared.write(frame_origin_infrared)
            else:
                if self.video_origin is not None:
                    self.video_origin.write(frame_origin)


        if cfg.save_detect_video:
            if self.video_detect is not None:
                self.video_detect.write(frame_detected)
    
    def save_frame_by_interval(self, frame_origin=None, frame_detected=None, results=None):
        current_time = cfg.global_timer.get_cur_time(is_init=True)

        if current_time > cfg.global_timer.last_time + cfg.global_window.save_num.value()/2 and current_time % cfg.global_window.save_num.value() == 0:  # some  variable parameters
            img_name = "/"+str(cfg.global_window.Inum) + ".jpg"
            # cv2.imwrite(self.img_path_detect+img_name, final_frame)

            if frame_origin is not None:
                # cv2.imwrite(self.img_path_origin+img_name, cv2.resize(c_frame,(640,640))) #test
                cv2.imwrite(self.img_path_origin + img_name, frame_origin)

            if results is not None:
                cv2.imwrite(self.img_path_detect+img_name, cv2.cvtColor(frame_detected, cv2.COLOR_RGB2BGR))

            if results is not None:
                results_track = results['results_track']
                results_grade = results['results_grade']
                cfg.global_detector.save_json(results_track, self.json_save_dir+"/", cfg.global_window.Inum+'_results_track')
                cfg.global_detector.save_json(results_grade, self.json_save_dir+"/", cfg.global_window.Inum+'_results_grade')

            cfg.global_window.statusBar().showMessage("Saving file:"+img_name)
            cfg.global_window.Inum += 1
            cfg.global_timer.last_time = current_time

    # elif cfg.global_window.save_check.isChecked():
    #     if current_time > cfg.global_timer.last_time + cfg.global_window.save_num.value()/2 and current_time % cfg.global_window.save_num.value() == 0:  # some  variable parameters
    #         img_name = "/"+str(cfg.global_window.Inum) + ".jpg"
    #         cv2.imwrite(self.img_path_origin + img_name, frame_origin)
    #         self.statusBar().showMessage("Saving file:"+img_name)
    #         cfg.global_window.Inum += 1
    #         cfg.global_timer.last_time = current_time


    def adjust_image_in_hsv(self, frame):
        hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * self.saturation_factor, 0, 255)
        hsv_image[:, :, 0] = (hsv_image[:, :, 0] + self.hue_shift) % 360
        hsv_image[:, :, 2] = np.clip(hsv_image[:, :, 2] * self.value_factor, 0, 255)
        adjusted_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        return adjusted_image

    def open_img(self):  
        if self.imgStatus is True:
            self.timer_img.stop()
        self.imgPath, imgType = QFileDialog.getOpenFileName(self, "Open image file", "", "*.jpg;;*.png;;All Files(*)")
        if self.timer_img.isActive() is False:
            threading.Thread(target=self.img_streaming, args=(self.queue,), daemon=True).start()
            self.timer_img.start(20)
            cfg.global_window.statusBar().showMessage("img start")
            self.imgStatus = True
        else:
            # self.vclose = False
            self.timer_img.stop()
            # cam.stop_streaming()
            self.label_pic.clear()
            # self.OpenCamButton.setText("Open Camera")

    def img_streaming(self, queue):
        self.is_streaming = True
        while self.is_streaming:
            c_frame = cv2.imread(self.imgPath)
            print("put")
            queue.put(c_frame)

    def show_img(self):
        queue = cfg.global_camera_handler.queue
        if not queue.empty() and not cfg.global_window.detect_check.isChecked():
            c_frame = queue.get()
            cframe = c_frame.copy()
            c_frame = self.adjust_image_in_hsv(cframe)
            # print(c_frame.shape)
            # cv2.putText(c_frame, f'Time: {int(time())}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 6)
            label_width = self.label_pic.width()
            label_height = self.label_pic.height()
            c_frame = cv2.cvtColor(c_frame, cv2.COLOR_BGR2RGB)
            temp_imgSrc = QImage(c_frame, c_frame.shape[1], c_frame.shape[0], c_frame.shape[1] * 3,
                                 QImage.Format_RGB888)
            pixmap_imgSrc = QPixmap.fromImage(temp_imgSrc).scaled(label_width, label_height)
            self.label_pic.setPixmap(pixmap_imgSrc)
            print("show")
            # if self.makeResize:
            # self.label_pic.setPixmap(pixmap_imgSrc.copy(self.mapped_rect))
            if int(time() % cfg.global_window.save_num.value()) == 0 and cfg.global_window.save_check.isChecked():  # some  variable parameters
                img_name = "./" + str(cfg.global_window.Inum) + ".jpg"
                cv2.imwrite(self.img_path_detect + img_name, c_frame)
                cfg.global_window.statusBar().showMessage("Saving file:" + img_name)
                cfg.global_window.Inum += 1

        elif not queue.empty() and self.detect_check.isChecked():
            c_frame = queue.get()
            cframe = c_frame.copy()
            c_frame = self.adjust_image_in_hsv(cframe)

            start_time = time()
            c_framer = cv2.cvtColor(c_frame, cv2.COLOR_BGR2RGB)
            if self.cur_model == "yolov8bestS.pt":
                resize_frame = cv2.resize(c_framer, (800, 800))
            elif self.cur_model == "best_ckpt1.pth":
                resize_frame = cv2.resize(c_framer, (800, 800))
            elif self.cur_model == "yolov8bestS1.pt":
                resize_frame = cv2.resize(c_framer, (1500, 1500))
            else:
                resize_frame = cv2.resize(c_framer, (717, 669))
            cheight, cwidth, _ = c_frame.shape
            results = self.model_detect_one_frame(resize_frame)
            final_frame = self.plot_boxes(results, resize_frame)
            final_oriframe = self.plot_oriboxes(results, c_framer)
            end_time = time()
            fps = 1/np.round(end_time-start_time, 2)
            cv2.putText(final_frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            label_width = self.label_pic.width()
            label_height = self.label_pic.height()
            temp_imgSrc = QImage(final_frame, final_frame.shape[1], final_frame.shape[0], final_frame.shape[1] * 3,
                                 QImage.Format_RGB888)
            pixmap_imgSrc = QPixmap.fromImage(temp_imgSrc).scaled(label_width, label_height)
            self.label_pic.setPixmap(pixmap_imgSrc)
            if int(time() % cfg.global_window.save_num.value()) == 0 and cfg.global_window.save_check.isChecked():  # some  variable parameters

                img_name = "/"+str(cfg.global_window.Inum) + ".jpg"
                # cv2.imwrite(self.img_path_detect+img_name, final_frame)
                # restored_image = cv2.resize(resized_image, original_size)
                cv2.imwrite(self.img_path_detect+img_name, cv2.cvtColor(final_oriframe, cv2.COLOR_RGB2BGR))
                # cv2.imwrite (self.img_path_origin+img_name, cv2.resize(c_frame,(640,640))) #test
                cv2.imwrite(self.img_path_origin + img_name, c_frame)
                cfg.global_detector.save_json(results, self.json_save_dir+"/", cfg.global_window.Inum)
                cfg.global_window.statusBar().showMessage("Saving file:"+img_name)
                cfg.global_window.Inum += 1
