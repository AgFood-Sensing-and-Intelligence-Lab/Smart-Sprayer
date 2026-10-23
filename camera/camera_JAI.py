import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))  # FasterRCNN
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))  # FasterRCNN

import vmbpy 
from queue import Queue, Full, Empty
import numpy as np

import eBUS as eb
import time
import json
import cv2
import threading
from datetime import datetime
import queue

from camera_abstract import CameraBase
from functools import partial
import traceback

try:
    import JAI_control.sample.lib.PvSampleUtils as psu
    import config as cfg
    camera_cur_fps = cfg.camera_cur_fps
    camera_time_expo = cfg.camera_time_expo
    frame_need_rotate_angle = cfg.frame_need_rotate_angle
except Exception as e:
    cfg = None
    camera_cur_fps = 30
    camera_time_expo = 10000
    frame_need_rotate_angle = cv2.ROTATE_90_COUNTERCLOCKWISE
    print(e)

SAVE_DIR = r"D:\test"
BUFFER_COUNT = 512
kb = psu.PvKb()

g_display_im = None
g_end_streaming = False

PIXEL_TYPE_MONO8 = 0x01080001
PIXEL_TYPE_BAYERRG8 = 0x01080009  # 17301513

def extract_blockid(filename):
    try:
        return int(filename.split("_")[1])  
    except (IndexError, ValueError):
        return -1  

class SourceStream:
    # def __init__(self, device, connection_id, source_name, rgb_queue, ir_queue):
    def __init__(self, device, connection_id, source_name, sync_callback):
        self.device = device
        self.connection_id = connection_id
        self.source_name = source_name
        self.sync_callback = sync_callback  # <--- Store the callback
        
        self.stream = None
        self.pipeline = None
        self.running = False

        self.capture_thread = None
        self.display_queue = Queue(maxsize=5)
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
        # while self.running and not kb.is_stopping():
        try:
            while self.running:
                result, buffer, op_result = self.pipeline.RetrieveNextBuffer(1000)
                if result.IsOK() and op_result.IsOK():
                    image = buffer.GetImage()
                    if image:
                        width, height = image.GetWidth(), image.GetHeight()
                        pixel_type = image.GetPixelType()
                        ptr = image.GetDataPointer()
                        block_id = int(buffer.GetBlockID()) # Ensure integer
                        payload_size = buffer.GetSize()
                        timestamp = datetime.now()

                        # 1. FAST RAW COPY
                        raw_image = None
                        if pixel_type == eb.PvPixelMono8 or pixel_type == PIXEL_TYPE_BAYERRG8:
                            raw_image = np.ctypeslib.as_array(ptr, shape=(height, width)).copy()
                        elif pixel_type == eb.PvPixelRGB8:
                            raw_image = np.ctypeslib.as_array(ptr, shape=(height, width, 3)).copy()

                        # 2. RELEASE BUFFER IMMEDIATELY
                        self.pipeline.ReleaseBuffer(buffer)

                        if raw_image is None:
                            continue

                        # 3. HEAVY LIFTING
                        np_image = raw_image
                        if pixel_type == PIXEL_TYPE_BAYERRG8:
                            np_image = cv2.cvtColor(raw_image, cv2.COLOR_BayerRG2RGB)
                        elif pixel_type == eb.PvPixelRGB8:
                            np_image = cv2.cvtColor(raw_image, cv2.COLOR_RGB2BGR)

                        if cfg is not None and getattr(cfg, 'frame_need_rotate_angle', None) is not None:
                            np_image = cv2.rotate(np_image, frame_need_rotate_angle)

                        meta = {
                            "block_id": block_id,
                            "timestamp": timestamp,
                            "width": width,
                            "height": height,
                            "pixel_type": int(pixel_type),
                            "payload_size": int(payload_size)
                        }

                        # 4. SEND TO SYNCHRONIZER
                        is_ir = (pixel_type == eb.PvPixelMono8)
                        self.sync_callback(block_id, np_image, meta, is_ir)
                else:
                    if buffer:
                        self.pipeline.ReleaseBuffer(buffer)
        except Exception as e:
            print(e)
            traceback.print_exc()

    def display_loop(self):
        while self.running:
            try:
                block_id, img = self.display_queue.get(timeout=0.5)
                cv2.putText(img, f"{self.source_name} - ID: {block_id}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow(f"{self.source_name}", img)
                if cv2.waitKey(1) & 0xFF == 27:
                    self.running = False
            except Empty:
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
        
        # --- NEW SYNC ARCHITECTURE ---
        self.paired_queue = Queue(maxsize=10) 
        self.sync_buffer = {} # Temporarily holds frames until their partner arrives
        self.sync_lock = threading.Lock()

    def _sync_frame_callback(self, block_id, image, meta, is_ir):
        """Called by the SourceStreams. Pairs RGB and IR frames by hardware ID."""
        with self.sync_lock:
            # Initialize the dict for this block_id if it doesn't exist
            if block_id not in self.sync_buffer:
                self.sync_buffer[block_id] = {}
            
            # Store the frame
            if is_ir:
                self.sync_buffer[block_id]['ir'] = (image, meta)
            else:
                self.sync_buffer[block_id]['rgb'] = (image, meta)
                
            # Check if we have both frames for this specific hardware trigger
            if 'rgb' in self.sync_buffer[block_id]:
                # We have a perfect pair! 
                rgb_data = self.sync_buffer[block_id]['rgb']
                ir_data = (None, None)
                if 'ir' in self.sync_buffer[block_id]:
                    ir_data = self.sync_buffer[block_id]['ir']
                
                packet = (rgb_data, ir_data, block_id)
                
                # Push the perfectly paired packet to the SINGLE queue
                try:
                    self.paired_queue.put_nowait(packet)
                except queue.Full:
                    try:
                        self.paired_queue.get_nowait() # Drop oldest paired frames
                    except queue.Empty:
                        pass
                    self.paired_queue.put_nowait(packet)
                    
                # Clean up memory: Delete this block_id and any older orphaned frames
                keys_to_delete = [k for k in self.sync_buffer.keys() if k <= block_id]
                for k in keys_to_delete:
                    del self.sync_buffer[k]
                    
            # Failsafe: Prevent memory leak if one stream completely dies
            if len(self.sync_buffer) > 15:
                oldest_key = next(iter(self.sync_buffer))
                del self.sync_buffer[oldest_key]

    def capture(self):
        # 1. Non-blocking safety check: Check if the queue has anything first
        # if self.paired_queue.empty():
        #     # Simply return indicating failure without stalling or logging a traceback
        #     return False, [None, None], 0

        try:
            # 2. Extract packet immediately without blocking the main thread
            # packet = self.paired_queue.get_nowait()
            packet = self.paired_queue.get(timeout=0.1)
        except queue.Empty:
            # Fallback for multithreading race conditions
            return False, [None, None], 0

        if packet is None:
            return False, [None, None], 0

        rgb_data, ir_data, block_id = packet
        image, meta = rgb_data
        image_infrared, meta_ir = ir_data

        timestamp = meta["timestamp"]
        success = True
            
        if cfg:
            cfg.camera_latest_frame = image
            cfg.camera_latest_frame_timestamp = timestamp
            cfg.camera_latest_frame_idx = self.frame_idx 
            
        self.frame_idx += 1
        images = [image, image_infrared]
        return success, images, timestamp

    def capture_v0(self):
        try:
            packet = None
            # We now only wait on ONE queue. Both frames are guaranteed to be in here.
            packet = self.paired_queue.get(timeout=0.1)
            # packet = self.paired_queue.get_nowait()
        except queue.Empty:
            # print("Warning: Camera queue timed out. No frame received.")
            # traceback.print_exc()
            return False, [None, None], 0
        if packet is None:
            return False, [None, None], 0

        rgb_data, ir_data, block_id = packet
        image, meta = rgb_data
        image_infrared, meta_ir = ir_data

        timestamp = meta["timestamp"]
        success = True
            
        if cfg:
            cfg.camera_latest_frame = image
            cfg.camera_latest_frame_timestamp = timestamp
            cfg.camera_latest_frame_idx = self.frame_idx 
            
        self.frame_idx += 1
        images = [image, image_infrared]
        return success, images, timestamp

    # ... (Keep end(), config(), show_camera(), check_camera(), set_exposure() the same) ...

    def check_camera(self):
            pass

    def set_exposure(self, window):
        pass

    def config(self):
            pass

    def show_camera(self):
        pass

    def end(self):
        print("\n Stopping all streams...")
        try:
            for s in self.sources:
                s.stop_thread()
                # s.stop_acquisition()
                # s.close()
        except Exception as e:
            print(e)
        time.sleep(0.1)
        print("Done.")

    def setup_camera(self, cam=None):
        # ... (Keep the connection and parameter setup the same) ...
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
        params.Get("AcquisitionFrameRate").SetValue(camera_cur_fps)

        for name in ["Source0", "Source1", "Source2"]:
            params.Get("SourceSelector").SetValue(name)
            params.Get("ExposureTime").SetValue(camera_time_expo)
        
        if self.sources is None:
            sources = []
            selector = device.GetParameters().GetEnum("SourceSelector")
            result, count = selector.GetEntriesCount()
            for i in range(count):
                result, entry = selector.GetEntryByIndex(i)
                if result.IsOK() and entry:
                    result, name = entry.GetName()
                    if result.IsOK():
                        # Pass the SYNC CALLBACK to the stream, not the queues
                        stream = SourceStream(device, connection_id, name, self._sync_frame_callback)
                        if stream.open():
                            sources.append(stream)
                if not cfg.has_infrared_channel:
                    break
            self.sources = sources
            if not sources:
                print("No source streams opened.")
                return
        if cfg is not None:
            self.window = cfg.global_window
            window = self.window
            window.statusBar().showMessage('JAI camera is connected.')

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
        if __name__ != '__main__':
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
                traceback.print_exc()

            if frame_nps is not None:
                frame_timestamp = datetime.now()
                if __name__ != '__main__':
                    if cfg.use_camera_timestamp:
                        frame_timestamp = timestamp
                    cfg.camera_latest_frame = frame_nps[0]
                    cfg.camera_latest_frame_timestamp =frame_timestamp
                    cfg.camera_latest_frame_idx = frame_idx
                frame_idx += 1
                frame_info = (frame_nps, frame_timestamp, frame_idx)
                try:
                    frame_queue.put_nowait(frame_info)  # put the capture image into queue
                except Exception as e:
                    try:
                        _ = frame_queue.get_nowait()  # drop oldest
                    except Empty:
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
            cfg.global_camera_handler.camera_fps = self._fps_value
            cfg.global_window.statusBar().showMessage("streaming stopped")


def demo():
    import cv2
    cv2.namedWindow('im', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('im', 800, 600)
    cv2.namedWindow('im_infrared', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('im_infrared', 800, 600)
    
    camera = CameraJAI()
    camera.check_camera()
    camera.setup_camera()

    prev_time = time.time()
    im_last = None
    fps = 0.0
    frame_queue = Queue(maxsize=2)  # queue for video frames

    # Start the streaming engine in the background
    threading.Thread(target=camera.camera_streaming, args=(frame_queue,), daemon=True).start()
    
    while True:
        try:
            # READ FROM THE QUEUE, NOT camera.capture()
            frame_info = frame_queue.get(timeout=1.0)
            ims, timestamp, frame_idx = frame_info
            
            if ims is None:
                continue
                
            im, im_infrared = ims
            
            # Update FPS
            current_time = time.time()
            fps = 1 / max((current_time - prev_time), 0.001)
            prev_time = current_time
                
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

        except Empty:
            # Queue is empty, just wait for the next frame
            pass

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # Stop streams before destroying windows
    camera.is_streaming = False
    camera.end()
    cv2.destroyAllWindows()


def main():
    demo()

if __name__ == '__main__':
    main()