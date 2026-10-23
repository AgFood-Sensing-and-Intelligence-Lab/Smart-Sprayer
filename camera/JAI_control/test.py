#!/usr/bin/env python3
"""
https://www.jai.com/support-software/jai-software/
eBUS-SDK-64-bit-for-JAI.6.6.1.7475.zip
install ebus_python-6.6.1-7475_jai-py38-none-win_amd64.whl
"""
import eBUS as eb
import sys
import os
import time
import json
import numpy as np
import cv2
import threading
import queue

# sys.path.append("../sample/lib")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../sample/lib'))  # FasterRCNN

import sample.lib.PvSampleUtils as psu

BUFFER_COUNT = 512
SAVE_DIR = r"D:\test"
kb = psu.PvKb()

# 保存队列线程
save_queue = queue.Queue()

def save_worker():
    while True:
        item = save_queue.get()
        if item is None:
            break
        bin_path, meta_path, buffer_data, meta = item
        try:
            with open(bin_path, "wb") as f:
                f.write(buffer_data)
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=4)
        except Exception as e:
            print(f"[Save Error] {e}")
        save_queue.task_done()

threading.Thread(target=save_worker, daemon=True).start()

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
                    timestamp = int(time.time() * 1000)

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
                    save_queue.put((bin_path, meta_path, buffer_data, meta))

                    np_image = None
                    if pixel_type == eb.PvPixelMono8:
                        np_image = np.ctypeslib.as_array(ptr, shape=(height, width)).copy()
                    elif pixel_type == 0x01080009:  # BayerRG8
                        bayer = np.ctypeslib.as_array(ptr, shape=(height, width)).copy()
                        np_image = cv2.cvtColor(bayer, cv2.COLOR_BayerRG2RGB)
                    elif pixel_type == eb.PvPixelRGB8:
                        np_image = np.ctypeslib.as_array(ptr, shape=(height, width, 3)).copy()
                        np_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)

                    # self.frame_count += 1
                    # if self.frame_count % self.display_interval == 0 and np_image is not None:
                    #     self.display_queue.put((block_id, np_image))

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

def main():
    print("▶ Streaming from multiple sources. Press ESC to stop.")
    connection_id = psu.PvSelectDevice()
    if not connection_id:
        return

    result, device = eb.PvDevice.CreateAndConnect(connection_id)
    if result.IsFailure():
        print("❌ Failed to connect to device.")
        return

    # Global parameter setup
    params = device.GetParameters()
    params.Get("AcquisitionMode").SetValue("Continuous")
    params.Get("TriggerMode").SetValue("Off")
    params.Get("AcquisitionFrameRate").SetValue(30)

    for name in ["Source0", "Source1", "Source2"]:
        params.Get("SourceSelector").SetValue(name)
        params.Get("ExposureTime").SetValue(33000)

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

    if not sources:
        print("❌ No source streams opened.")
        return
    print("\n⏹ Starting all streams...")
    for s in sources:
        s.start_acquisition()
        s.start_thread()

    kb.start()
    while not kb.kbhit():
        time.sleep(0.1)
    kb.getch()
    kb.stop()

    print("\n⏹ Stopping all streams...")
    for s in sources:
        s.stop_thread()
        s.stop_acquisition()
        s.close()

    save_queue.put(None)  # Stop save worker
    time.sleep(1)
    cv2.destroyAllWindows()

    print("✅ Done.")
    device.Disconnect()
    eb.PvDevice.Free(device)

if __name__ == "__main__":
    main()
