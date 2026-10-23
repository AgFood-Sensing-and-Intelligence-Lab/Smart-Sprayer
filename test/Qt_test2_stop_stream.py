from PyQt5.QtWidgets import *
import cv2
from vimba import *
from PyQt5.QtGui import *
from  PyQt5.QtCore import *
import cv2
import torch
import json
import os
import threading
from typing import Optional
#from time import time
import time
import numpy as np
import sys



def  ShowImage():
    image_path = textEdit1.toPlainText()
    image_path="example1.jpg"

    img_path = 'example1.jpg'  # 设置图片路径
    showImage = QPixmap(img_path).scaled(label_pic.width(), label_pic.height())  # 适应窗口大小
    label_pic.setPixmap(showImage)  # 显


def fault_show_function():
    dialog_fault = QDialog()
    image_path = "example1.jpg"
    pic = QPixmap(image_path)
    label_pic = QLabel("show", dialog_fault)
    label_pic.setPixmap(pic)
    label_pic.setGeometry(10, 10, 500, 500)
    label_pic.setScaledContents(True)
    dialog_fault.exec_()

def  Image_path_dialog():
    SP_text=textEdit1.toPlainText()
    if len(SP_text)==0:
        QMessageBox.about(window,"Information","Please input Image Path")

def show_detect_image():
    #model = torch.hub.load('/home/nvidia/DCW-main/YOLOv5', 'custom', path='bestL.pt', source='local')
    img = cv2.imread('example1.jpg')
    results = model(img, size=640)
    result_img=results.imgs
    labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    n = len(labels)
    x_shape, y_shape = img.shape[1], img.shape[0]
    for i in range(n):
        row = cord[i]
        if row[4] >= 0.3:
            x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(
                row[3] * y_shape)
            bgr = (0, 255, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), bgr, 2)
            cv2.putText(img, class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
            img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    label_width = label_pic.width()
    label_height = label_pic.height()
            # 将图片转换为QImage
    temp_imgSrc = QImage(img, img.shape[1], img.shape[0], img.shape[1] * 3,QImage.Format_RGB888)
    pixmap_imgSrc = QPixmap.fromImage(temp_imgSrc).scaled(label_width, label_height)

            # 使用label进行显示
    label_pic.setPixmap(pixmap_imgSrc)


    #print(img)
    #results.show()

def run_realTime():
    os.system("python realTime.py")

def sliderH_value():
    label_sliderH.setText("Hue:"+str(slider_H.value())) #做了个强转，不然报错：label框需要str类型值

def sliderS_value():
    label_sliderS.setText("Saturation:" + str(0.1*slider_S.value()))  # 做了个强转，不然报错：label框需要str类型值

def sliderG_value():
    label_sliderG.setText("Gamma:"+str(0.1*slider_G.value())) #做了个强转，不然报错：label框需要str类型值 0.1 in case int

##
def print_preamble():
    print('///////////////////////////////////////////////////////')
    print('/// Vimba API Asynchronous Grab with OpenCV Example ///')
    print('///////////////////////////////////////////////////////\n')


def print_usage():
    print('Usage:')
    print('    python asynchronous_grab_opencv.py [camera_id]')
    print('    python asynchronous_grab_opencv.py [/h] [-h]')
    print()
    print('Parameters:')
    print('    camera_id   ID of the camera to use (using first camera if not specified)')
    print()


def abort(reason: str, return_code: int = 1, usage: bool = False):
    print(reason + '\n')

    if usage:
        print_usage()

    sys.exit(return_code)


def parse_args() -> Optional[str]:
    args = sys.argv[1:]
    argc = len(args)

    for arg in args:
        if arg in ('/h', '-h'):
            print_usage()
            sys.exit(0)

    if argc > 1:
        abort(reason="Invalid number of arguments. Abort.", return_code=2, usage=True)

    return None if argc == 0 else args[0]


def get_camera(camera_id: Optional[str]) -> Camera:
    with Vimba.get_instance() as vimba:
        if camera_id:
            try:
                return vimba.get_camera_by_id(camera_id)

            except VimbaCameraError:
                abort('Failed to access Camera \'{}\'. Abort.'.format(camera_id))

        else:
            cams = vimba.get_all_cameras()
            if not cams:
                abort('No Cameras accessible. Abort.')

            return cams[0]


def setup_camera(cam: Camera):
    with cam:
        # Enable auto exposure time setting if camera supports it
        cam.ExposureAuto.set('Off')
        cam.ExposureMode.set('Timed')
        cam.ExposureTime.set('13996.768')

        # setting the color
        cam.BalanceWhiteAuto.set('Continuous')
        HueValue=slider_H.value()
        SaturationValue=0.1*slider_S.value()
        GammaValue=0.1*slider_G.value()
        cam.Hue.set(str(HueValue))
        cam.Saturation.set(str(SaturationValue))
        cam.Gamma.set(str(GammaValue))

        # setting image resolution (affects acquisition frame rate)
        # range of height (8-2056), range of width (8-2464)
        # cam.Height.set('2056')
        # cam.Width.set('2464')

        # setting the binning reduce the res
        cam.BinningHorizontal.set('2')
        cam.BinningVertical.set('2')

        # setting pixel format
        cam.set_pixel_format(PixelFormat.Bgr8)

        # setting bandwidth of the data that will be streaming, range from (16250000-450000000)
        # (affects acquisition frame rate)
        cam.DeviceLinkThroughputLimit.set('450000000')

        cam.TriggerSelector.set('FrameStart')

        # optional in this example but good practice as it might be needed for hardware triggering
        cam.TriggerActivation.set('RisingEdge')

        # Make camera listen to Software trigger
        cam.TriggerSource.set('Software')
        cam.TriggerMode.set('Off')


def score_frame(frame):
    """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
    global model
    global device
    i = 0
    while (os.path.exists("js_data/data%s.json" % i)):
        i += 1
    model.to(device)
    frame = [frame]
    results = model(frame)
    # results.save()
    parsed = json.loads(results.pandas().xyxy[0].to_json(orient="records"))
    f = open('js_data/data%s.json' % i, 'w')
    json.dump(parsed, f)
    f.close()
    labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    return labels, cord


def class_to_label(x):
    """
    For a given label value, return corresponding string label.
    :param x: numeric label
    :return: corresponding string label
    """
    global model
    classes = model.names
    return classes[int(x)]


def plot_boxes(results, frame):
    """
    Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
    :param results: contains labels and coordinates predicted by model on the given frame.
    :param frame: Frame which has been scored.
    :return: Frame with bounding boxes and labels ploted on it.
    """

    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    for i in range(n):
        row = cord[i]
        if row[4] >= 0.3:
            x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(
                row[3] * y_shape)
            bgr = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
            cv2.putText(frame, class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
    return frame

def  frame_handler(cam , frame ):
    opencv_frame=frame.as_opencv_image()
    label_width = label_pic.width()
    label_height = label_pic.height()
    # print(final_frame)
    # 将图片转换为QImage
    opencv_frame = cv2.cvtColor(opencv_frame, cv2.COLOR_BGR2RGB)
    print(type(opencv_frame))
    temp_imgSrc = QImage(opencv_frame, opencv_frame.shape[1], opencv_frame.shape[0], opencv_frame.shape[1] * 3,
                         QImage.Format_RGB888)
    pixmap_imgSrc = QPixmap.fromImage(temp_imgSrc).scaled(label_width, label_height)
    print(pixmap_imgSrc)

    # 使用label进行显示
    label_pic.setPixmap(pixmap_imgSrc)
    #cv2.imshow("sd",opencv_frame)
    cam.queue_frame(frame)


class Handler:
    def __init__(self):
        self.shutdown_event = threading.Event()

    def __call__(self, cam: Camera, frame: Frame):
        ENTER_KEY_CODE = 13
        key = cv2.waitKey(1)
        if key == ENTER_KEY_CODE:
            self.shutdown_event.set()
            return

        elif frame.get_status() == FrameStatus.Complete:
            print('{} acquired {}'.format(cam, frame), flush=True)
            msg = 'Stream from \'{}\'. Press <Enter> to stop stream.'
            start_time = time()
            opencv_frame = frame.as_opencv_image()
            opencv_frame=cv2.cvtColor(opencv_frame, cv2.COLOR_BGR2RGB)
            resize_frame = cv2.resize(opencv_frame, (640, 640))
            results = score_frame(resize_frame)
            final_frame = plot_boxes(results, resize_frame)
            end_time = time()
            fps = 1 / np.round(end_time - start_time, 2)

            cv2.putText(final_frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            #cv2.imshow(msg.format(cam.get_name()), final_frame)
            label_width = label_pic.width()
            label_height = label_pic.height()
            #print(final_frame)
            # 将图片转换为QImage
            final_frame = cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB)
            print(type(final_frame))
            temp_imgSrc = QImage(final_frame, final_frame.shape[1], final_frame.shape[0], final_frame.shape[1] * 3, QImage.Format_RGB888)
            pixmap_imgSrc = QPixmap.fromImage(temp_imgSrc).scaled(label_width, label_height)
            print(pixmap_imgSrc)

            # 使用label进行显示
            label_pic.setPixmap(pixmap_imgSrc)
            #label_pic.show()
            ###
            frame_id=frame.get_id()
            if frame_id%save_num.value()==0 and save_check.isChecked(): #some  variable parameters
                img_name = "/media/nvidia/64A5-F009/DetectedImages/" + str(frame_id) + ".jpg"
                cv2.imwrite(img_name, final_frame)
                print(frame.get_id())
        cam.queue_frame(frame)

def Run_realTime():
    print_preamble()
    cam_id = parse_args()


    with Vimba.get_instance():
        with get_camera(cam_id) as cam:
            # Start Streaming, wait for five seconds, stop streaming
            setup_camera(cam)
            for i in range(0,10):
            #handler = Handler()
            #cam.start_streaming(frame_handler)
            #handler.shutdown_event.wait()
            #cam.TriggerSoftware.run()

                c_frame=cam.get_frame()
                c_frame=c_frame.as_opencv_image()
                label_width = label_pic.width()
                label_height = label_pic.height()
            # print(final_frame)
            # 将图片转换为QImage
                c_frame = cv2.cvtColor(c_frame, cv2.COLOR_BGR2RGB)
            #print(type(c_frame))
                temp_imgSrc = QImage(c_frame, c_frame.shape[1], c_frame.shape[0], c_frame.shape[1] * 3,
                                 QImage.Format_RGB888)
                pixmap_imgSrc = QPixmap.fromImage(temp_imgSrc).scaled(label_width, label_height)
            #print(pixmap_imgSrc)

            # 使用label进行显示
                label_pic.setPixmap(pixmap_imgSrc)
                print(i)
                time.sleep(2)
            # cv2.imshow("sd",opencv_frame)

def camera_streaming(queue):
                global is_streaming
                is_streaming = True
                print("streaming started")
                with Vimba.get_instance() as vimba:
                    with vimba.get_all_cameras()[0] as camera:
                        while is_streaming:
                            frame = camera.get_frame()
                            frame = frame.as_opencv_image()
                            queue.put(frame)  # put the capture image into queue
                print("streaming stopped")


def camera_frame():
    global is_streaming
    is_streaming = True
    print("streaming started")
    print_preamble()
    cam_id = parse_args()

    with Vimba.get_instance():
        with get_camera(cam_id)as cam:
            if is_streaming:
                frame = cam.get_frame()
                frame = frame.as_opencv_image()
    return frame



def button_open_camera_clicked():
        if timer_camera.isActive() == False:  # 若定时器未启动
                timer_camera.start(30)  # 定时器开始计时30ms，结果是每过30ms从摄像头中取一帧显示
               # button_open_camera.setText('关闭相机')
        else:
            timer_camera.stop()  # 关闭定时器

def show_camera():
        image = camera_frame()  # 从视频流中读取
        label_width = label_pic.width()
        label_height = label_pic.height()

        #show = cv2.resize(self.image, (640, 480))  # 把读到的帧的大小重新设置为 640x480
        show = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 视频色彩转换回RGB，这样才是现实的颜色
        print(show)
        showImage = QImage(show.data, show.shape[1], show.shape[0],
                                         QImage.Format_RGB888)  # 把读取到的视频数据变成QImage形式
        print(showImage)
        label_pic.setPixmap(QPixmap.fromImage(showImage))# 往显示视频的Label里 显示QImage


        #temp_imgSrc = QImage(c_frame, c_frame.shape[1], c_frame.shape[0], c_frame.shape[1] * 3,
                             #QImage.Format_RGB888)


            #time.sleep(10)
           # cam.stop_streaming()




app=QApplication([])

model = torch.hub.load('/home/nvidia/DCW-main/YOLOv5', 'custom', path='bestS.pt', source='local')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
timer_camera = QTimer()

window=QMainWindow()
window.resize(900,550)
window.move(300,310)
window.setWindowTitle('Weed Detection')

qbtn1=QRadioButton(window)
qbtn1.setText("Image")
qbtn1.move(30,30)
#qbtn1.setChecked(True)
qbtn1.clicked.connect(Image_path_dialog)

qbtn2=QRadioButton(window)
qbtn2.setText("Video")
qbtn2.move(130,30)

qbtn=QRadioButton(window)
qbtn.setText("RealTime")
qbtn.move(230,30)

label = QLabel('YOLOWeeds',window)
label.resize(300,300)
label.move(50,100)

"""
image_path = "example1.jpg"
pic = QPixmap(image_path)
label_pic = QLabel("show", window)
label_pic.resize(400,400)
label_pic.setPixmap(pic)
label_pic.move(400,100)
label_pic.setScaledContents (True)
"""
label_pic = QLabel(window)  # 设置图片显示label
label_pic.setText(" Frame")
label_pic.setFixedSize(400, 400)  # 设置图片大小
label_pic.move(400, 100)  # 设置图片位置
label_pic.setStyleSheet("QLabel{background:white;}")  # 设置label底色
###slider###############
#label_pic.setGeometry(10, 10, 1019, 537)
slider_H= QSlider(window)
slider_H.setOrientation(Qt.Horizontal)
slider_H.setMinimum(-40)
slider_H.setMaximum(40)
slider_H.move(200,170)
slider_H.setSingleStep(2)
slider_H.setValue(0)# 步长slider.setValue(18)  # 当前值
#slider_H.setTickPosition.TicksBelow  # 设置刻度的位置，刻度在下方
slider_H.setTickInterval(5)  #
slider_H.setTickPosition(QSlider.TicksBelow)
slider_H.valueChanged.connect(sliderH_value)
label_sliderH=QLabel(window)
label_sliderH.setText("Hue:"+str(slider_H.value()))
label_sliderH.move(200,140)
###
slider_S= QSlider(window)
slider_S.setOrientation(Qt.Horizontal)
slider_S.setMinimum(0)
slider_S.setMaximum(20)
slider_S.move(200,220)
slider_S.setSingleStep(1)
slider_S.setValue(10)# 步长slider.setValue(18)  # 当前值
#slider_H.setTickPosition.TicksBelow  # 设置刻度的位置，刻度在下方
slider_S.setTickInterval(2)  #
slider_S.setTickPosition(QSlider.TicksBelow)
slider_S.valueChanged.connect(sliderS_value)
label_sliderS=QLabel(window)
label_sliderS.setText("Saturation:"+str(0.1*slider_S.value()))
label_sliderS.move(200,200)
###
slider_G= QSlider(window)
slider_G.setOrientation(Qt.Horizontal)
slider_G.setMinimum(4)
slider_G.setMaximum(24)
slider_G.move(200,270)
slider_G.setSingleStep(2)
slider_G.setValue(7)# 步长slider.setValue(18)  # 当前值
#slider_H.setTickPosition.TicksBelow  # 设置刻度的位置，刻度在下方
slider_G.setTickInterval(2)  #
slider_G.setTickPosition(QSlider.TicksBelow)
slider_G.valueChanged.connect(sliderG_value)
label_sliderG=QLabel(window)
label_sliderG.setText("Gamma:"+str(0.1*slider_G.value()))
label_sliderG.move(200,250)


combol1 = QComboBox(window)
combol1.move(270,80)
combol1.addItem("Select Model")
combol1.addItem("bestS")
combol1.addItem("bestX")
combol1.addItem("bestM")
combol1.addItem("bestN")
combol1.addItem("bestL")

textEdit1 = QPlainTextEdit(window)
textEdit1.setPlaceholderText(" Sources Path")
textEdit1.move(30,80)
textEdit1.resize(100,30)


textEdit2 = QPlainTextEdit(window)
textEdit2.setPlaceholderText(" Save Path")
textEdit2.move(150,80)
textEdit2.resize(100,30)


button=QPushButton('ShowImage',window)
button.move(30,150)
button.clicked.connect(ShowImage)

button1=QPushButton('DetectImage',window)
button1.move(30,200)
button1.clicked.connect(show_detect_image)

button2=QPushButton('ShowVideo',window)
button2.move(30,250)
button2.clicked.connect(run_realTime)

button3=QPushButton('DetectVideo',window)
button3.move(30,300)
button3.clicked.connect(run_realTime)

button4=QPushButton('realTime',window)
button4.move(30,350)
#button4.clicked.connect(Run_realTime)

button4.clicked.connect(button_open_camera_clicked)  # 若该按键被点击，则调用button_open_camera_clicked()
timer_camera.timeout.connect(show_camera)  # 若定时器结束，则调用show_camera()
#button_close.clicked.connect(self.close)  # 若该按键被点击，则调用close()，注意这个close是父类

save_check=QCheckBox(window)
save_check.setText("Save Result")
save_check.setChecked(True)
save_check.move(200,310)
print(save_check.isChecked())

save_num=QSpinBox(window)
save_num.setValue(2)
save_num.move(200,350)
save_num.setSuffix("  frame rate")

window.show()
app. exec_()

