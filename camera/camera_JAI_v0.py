import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))  # FasterRCNN
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))  # FasterRCNN

import vmbpy 
from queue import Queue
import numpy as np

import eBUS as eb
import sys
import os
import time
import json
import numpy as np
import cv2
import threading
import queue

from camera_abstract import CameraBase
import cv2
from functools import partial
from datetime import datetime


try:
    import JAI_control.sample.lib.PvSampleUtils as psu
    import config as cfg
except Exception as e:
    cfg = None
    print(e)

SAVE_DIR = r"D:\test"
BUFFER_COUNT = 512
kb = psu.PvKb()

save_queue = queue.Queue(3)
save_queue_infrared = queue.Queue(3)

g_display_im = None
g_end_streaming = False

PIXEL_TYPE_MONO8 = 0x01080001
PIXEL_TYPE_BAYERRG8 = 0x01080009  # 17301513

def extract_blockid(filename):
    try:
        return int(filename.split("_")[1])  
    except (IndexError, ValueError):
        return -1  


def save_worker():
    while True:
        item = save_queue.get()
        if item is None:
            break
        bin_path, meta_path, buffer_data, meta = item
        try:
            pass
            # with open(bin_path, "wb") as f:
            #     f.write(buffer_data)
            # with open(meta_path, "w") as f:
            #     json.dump(meta, f, indent=4)
        except Exception as e:
            print(f"[Save Error] {e}")
        save_queue.task_done()

# threading.Thread(target=save_worker, daemon=True).start()

class SourceStream:
    def __init__(self, device, connection_id, source_name):
        self.device = device
        self.connection_id = connection_id
        self.source_name = source_name
        self.stream = None
        self.pipeline = None
        self.running = False
        self.capture_thread = None
        self.display_queue = queue.Queue(maxsize=5)
        self.save_path = os.path.join(SAVE_DIR, source_name)
        os.makedirs(self.save_path, exist_ok=True)

        self.frame_count = 0
        self.display_interval = 5  

    def open(self):
        stack = eb.PvGenStateStack(self.device.GetParameters())
        stack.SetEnumValue("SourceSelector", self.source_name)

        result, channel = self.device.GetParameters().GetIntegerValue("SourceIDValue")
        if result.IsFailure():
            result, channel = self.device.GetParameters().GetIntegerValue("SourceStreamChannel")
            if result.IsFailure():
                print(f"[{self.source_name}] Cannot determine stream channel.")
                return False

        self.stream = eb.PvStreamGEV()
        if self.stream.Open(self.connection_id, 0, channel).IsFailure():
            print(f"[{self.source_name}] Failed to open stream.")
            return False

        ip = self.stream.GetLocalIPAddress()
        port = self.stream.GetLocalPort()
        self.device.SetStreamDestination(ip, port, channel)

        payload_size = self.device.GetPayloadSize()
        self.pipeline = eb.PvPipeline(self.stream)
        self.pipeline.SetBufferSize(payload_size)
        self.pipeline.SetBufferCount(BUFFER_COUNT)
        self.pipeline.Start()

        return True

    def start_acquisition(self):
        stack = eb.PvGenStateStack(self.device.GetParameters())
        stack.SetEnumValue("SourceSelector", self.source_name)
        self.device.StreamEnable()
        self.device.GetParameters().Get("AcquisitionStart").Execute()

    def stop_acquisition(self):
        stack = eb.PvGenStateStack(self.device.GetParameters())
        stack.SetEnumValue("SourceSelector", self.source_name)
        self.device.GetParameters().Get("AcquisitionStop").Execute()
        self.device.StreamDisable()

    def close(self):
        self.pipeline.Stop()
        self.stream.Close()

    def run(self):
        self.running = True
        while self.running and not kb.is_stopping():
            result, buffer, op_result = self.pipeline.RetrieveNextBuffer(1000)
            if result.IsOK() and op_result.IsOK():
                image = buffer.GetImage()
                if image:
                    width, height = image.GetWidth(), image.GetHeight()
                    pixel_type = image.GetPixelType()
                    ptr = image.GetDataPointer()
                    block_id = buffer.GetBlockID()
                    timestamp = datetime.now()

                    # 1. FAST RAW COPY (Creates a safe copy in Python memory)
                    if pixel_type == eb.PvPixelMono8 or pixel_type == 0x01080009: # Mono or Bayer
                        raw_image = np.ctypeslib.as_array(ptr, shape=(height, width)).copy()
                    elif pixel_type == eb.PvPixelRGB8:
                        raw_image = np.ctypeslib.as_array(ptr, shape=(height, width, 3)).copy()

                    # 2. RELEASE BUFFER IMMEDIATELY! (Frees the camera to grab the next frame)
                    self.pipeline.ReleaseBuffer(buffer)

                    # 3. NOW DO YOUR HEAVY LIFTING
                    np_image = raw_image 
                    if pixel_type == 0x01080009:  # BayerRG8
                        np_image = cv2.cvtColor(raw_image, cv2.COLOR_BayerRG2RGB)
                    elif pixel_type == eb.PvPixelRGB8:
                        np_image = cv2.cvtColor(raw_image, cv2.COLOR_RGB2BGR)

                    if getattr(cfg, 'frame_need_rotate_angle', None) is not None:
                        np_image = cv2.rotate(np_image, cfg.frame_need_rotate_angle)

                    meta = {
                        "block_id": int(block_id), "timestamp": timestamp,
                        "width": width, "height": height, "pixel_type": int(pixel_type)
                    }

                    # 4. PUT IN QUEUE
                    target_queue = save_queue_infrared if pixel_type == eb.PvPixelMono8 else save_queue
                    
                    try:
                        # Use timeout instead of False to prevent instant silent drops, 
                        # but don't block forever
                        target_queue.put((None, None, np_image, meta), timeout=0.01) 
                    except queue.Full:
                        pass # Queue is full, drop frame to keep stream real-time
            else:
                # Always release if retrieval failed but buffer is somewhat valid
                if buffer:
                    self.pipeline.ReleaseBuffer(buffer)

    def run_v0(self):
        self.running = True
        while self.running and not kb.is_stopping():
            result, buffer, op_result = self.pipeline.RetrieveNextBuffer(1000)
            if result.IsOK() and op_result.IsOK():
                image = buffer.GetImage()
                if image:
                    width, height = image.GetWidth(), image.GetHeight()
                    pixel_type = image.GetPixelType()
                    ptr = image.GetDataPointer()
                    block_id = buffer.GetBlockID()
                    # timestamp = int(time.time() * 1000)
                    timestamp = datetime.now()

                    bin_path = os.path.join(self.save_path, f"frame_{block_id}_{timestamp}.bin")
                    meta_path = bin_path.replace(".bin", ".json")
                    buffer_data = ptr[:buffer.GetSize()]
                    meta = {
                        "block_id": int(block_id),
                        "timestamp": timestamp,
                        "width": width,
                        "height": height,
                        "pixel_type": int(pixel_type),
                        "payload_size": int(buffer.GetSize())
                    }
                    # try:
                    #     save_queue.put((bin_path, meta_path, buffer_data, meta), True)
                    # except Exception as e:
                    #     print(e)

                    np_image = None
                    if pixel_type == eb.PvPixelMono8:
                        np_image = np.ctypeslib.as_array(ptr, shape=(height, width)).copy()
                        if cfg.frame_need_rotate_angle is not None:
                            np_image = cv2.rotate(np_image, cfg.frame_need_rotate_angle)
                        try:
                            save_queue_infrared.put((bin_path, meta_path, np_image, meta), False)
                        except Exception as e:
                            print(e)
                    elif pixel_type == 0x01080009:  # BayerRG8
                        bayer = np.ctypeslib.as_array(ptr, shape=(height, width)).copy()
                        np_image = cv2.cvtColor(bayer, cv2.COLOR_BayerRG2RGB)
                        if cfg.frame_need_rotate_angle is not None:
                            np_image = cv2.rotate(np_image, cfg.frame_need_rotate_angle)
                        try:
                            save_queue.put((bin_path, meta_path, np_image, meta), False)
                        except Exception as e:
                            print(e)
                    elif pixel_type == eb.PvPixelRGB8:
                        np_image = np.ctypeslib.as_array(ptr, shape=(height, width, 3)).copy()
                        np_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
                        if cfg.frame_need_rotate_angle is not None:
                            np_image = cv2.rotate(np_image, cfg.frame_need_rotate_angle)
                        try:
                            save_queue.put((bin_path, meta_path, np_image, meta), False)
                        except Exception as e:
                            print(e)
                    # self.frame_count += 1
                    # if self.frame_count % self.display_interval == 0 and np_image is not None:
                    #     self.display_queue.put((block_id, np_image))
                    # global g_display_im
                    # g_display_im = np_image.copy()
            self.pipeline.ReleaseBuffer(buffer)

    def display_loop(self):
        while self.running:
            try:
                block_id, img = self.display_queue.get(timeout=0.5)
                cv2.putText(img, f"{self.source_name} - ID: {block_id}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow(f"{self.source_name}", img)
                if cv2.waitKey(1) & 0xFF == 27:
                    self.running = False
            except queue.Empty:
                continue

    def start_thread(self):
        self.capture_thread = threading.Thread(target=self.run)
        self.capture_thread.start()
        threading.Thread(target=self.display_loop, daemon=True).start()

    def stop_thread(self):
        self.running = False
        if self.capture_thread:
            self.capture_thread.join()


class CameraJAI(CameraBase):
    def __init__(self) -> None:
        super().__init__()
        self.cam = None
        self.window = None
        self.sources = None
        self.device = None
        self.frame_idx = 0

    def capture(self):
        item = save_queue.get()
        item_infrared = save_queue_infrared.get()
        if item is None:
            return False, None, 0
        # bin_path, meta_path, buffer_data, meta = item
        bin_path, meta_path, image, meta = item
        bin_path, meta_path, image_infrared, meta = item_infrared
        # meta = g_meta
        # raw = g_buffer_data
        width = meta["width"]
        height = meta["height"]
        pixel_type = meta["pixel_type"]
        timestamp = meta["timestamp"]
        block_id = meta.get("block_id", "unknown")
        # image = None
        # if pixel_type == PIXEL_TYPE_MONO8:
        #     image = np.frombuffer(buffer_data, dtype=np.uint8).reshape((height, width))
        # elif pixel_type == PIXEL_TYPE_BAYERRG8:
        #     bayer = np.frombuffer(buffer_data, dtype=np.uint8).reshape((height, width))
        #     image = cv2.cvtColor(bayer, cv2.COLOR_BayerRG2RGB)
        # else:
        #     print(f"[Unsupported pixel type] {pixel_type}")
        #     assert False
        success = True
        timestamp = 0
        if cfg:
            cfg.camera_latest_frame = image
            cfg.camera_latest_frame_timestamp = timestamp
            cfg.camera_latest_frame_idx = self.frame_idx 
        self.frame_idx += 1
        images = [image, image_infrared]
        return success, images, timestamp

    def end(self):
        print("\n Stopping all streams...")
        try:
            for s in self.sources:
                s.stop_thread()
                # s.stop_acquisition()
                # s.close()
        except Exception as e:
            print(e)
        # save_queue.put(None)  # Stop save worker
        time.sleep(0.1)
        print("Done.")
        # if self.device:
        #     self.device.Disconnect()
        #     eb.PvDevice.Free(self.device)

    def config(self):
        pass

    def show_camera(self):
        pass

    def setup_camera(self, cam=None):
        window = self.window
        if self.device is None:
            connection_id = psu.PvSelectDevice()
            if not connection_id:
                print('CameraJAI has no connection_id')

            result, device = eb.PvDevice.CreateAndConnect(connection_id)
            if result.IsFailure():
                print("Failed to connect to device.")
                return
            self.device = device
        # Global parameter setup
        params = device.GetParameters()
        params.Get("AcquisitionMode").SetValue("Continuous")
        params.Get("TriggerMode").SetValue("Off")
        params.Get("AcquisitionFrameRate").SetValue(cfg.camera_cur_fps)

        for name in ["Source0", "Source1", "Source2"]:
            params.Get("SourceSelector").SetValue(name)
            params.Get("ExposureTime").SetValue(cfg.camera_time_expo)

        if self.sources is None:
            # Enumerate and open sources
            sources = []
            selector = device.GetParameters().GetEnum("SourceSelector")
            result, count = selector.GetEntriesCount()
            for i in range(count):
                result, entry = selector.GetEntryByIndex(i)
                if result.IsOK() and entry:
                    result, name = entry.GetName()
                    if result.IsOK():
                        stream = SourceStream(device, connection_id, name)
                        if stream.open():
                            sources.append(stream)
            self.sources = sources
            if not sources:
                print("No source streams opened.")
                return

        if cfg is not None:
            self.window = cfg.global_window
            window = self.window
            window.statusBar().showMessage('JAI camera is connected.')

    def check_camera(self):
        window = None

    def set_exposure(self, window):
        pass

    def camera_streaming(self, frame_queue):
        self.is_streaming = True
        print("streaming started")
        if __name__ != '__main__':
            cfg.global_window.statusBar().showMessage("streaming started")
            buffer_count=cfg.frame_buffer_count
        else:
            buffer_count=10
        frame_idx = 0

        print("\n Starting all streams...")
        for s in self.sources:
            s.start_acquisition()
            s.start_thread()
        cfg.cam_start_streaming_time = datetime.now()
        while self.is_streaming:
            if not hasattr(self, '_fps_frame_count'):
                self._fps_frame_count = 0
                self._fps_last_time = time.time()
                self._fps_value = 0
            try:
                res, frame_nps, timestamp = self.capture()
            except Exception as e:
                print(e)

            if frame_nps is not None:
                if cfg.use_camera_timestamp:
                    frame_timestamp = timestamp
                else:
                    frame_timestamp = datetime.now()
                cfg.camera_latest_frame = frame_nps
                cfg.camera_latest_frame_timestamp =frame_timestamp
                cfg.camera_latest_frame_idx = frame_idx
                frame_idx += 1
                frame_info = (frame_nps, frame_timestamp, frame_idx)
                try:
                    frame_queue.put_nowait(frame_info)  # put the capture image into queue
                except Exception as e:
                    try:
                        _ = frame_queue.get_nowait()  # drop oldest
                        with frame_queue.mutex:
                            frame_queue.queue.clear()
                    except frame_queue.Empty:
                        pass
                    frame_queue.put_nowait(frame_info)

            self._fps_frame_count += 1
            now = time.time()
            if now - self._fps_last_time >= 1.0:
                self._fps_value = np.round(self._fps_frame_count / (now - self._fps_last_time), 2)
                self._fps_last_time = now
                self._fps_frame_count = 0
            cfg.global_camera_handler.camera_fps = self._fps_value
        print("streaming stopped")
        if __name__ != '__main__':
            cfg.global_window.statusBar().showMessage("streaming stopped")


def demo():
    """
    must collect resources after use each time, otherwise, use eBUS PLayer to reopen it once.
    """
    import cv2
    cv2.namedWindow('im', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('im', 800, 600)
    cv2.namedWindow('im_infrared', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('im_infrared', 800, 600)
    import numpy as np
    camera = CameraJAI()
    
    camera.check_camera()
    camera.setup_camera()


    prev_time = time.time()
    im_last = None
    fps = 0.0
    while True:
        res, ims = camera.capture()
        im, im_infrared = ims
        # im = g_display_im
        # Update FPS only if frame changed
        if im is not None:
            if im_last is None or not np.array_equal(im_last, im):
                current_time = time.time()
                fps = 1 / max((current_time - prev_time), 0.001)
                prev_time = current_time
                im_last = im.copy()
            # Draw FPS
            cv2.putText(
                im,
                f"FPS: {fps:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            cv2.imshow('im', im)
            cv2.imshow('im_infrared', im_infrared)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
    camera.end()

def main():
    demo()


if __name__ == '__main__':
    main()

