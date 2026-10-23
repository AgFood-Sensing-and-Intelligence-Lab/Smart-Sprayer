import serial
import time
import os
import numpy as np
import threading


BAUD_RATE = 9600
PREFIX = 'signal_nozzle_'

is_sending_signal_dict_1_12 = {}
is_sending_signal_dict_13_24 = {}
do_close_valve_dict = {}


log_path = './arduino_code/valve_control/signal/speed/speed_info.txt'

def log_xalg(*info, log_path=log_path, show=True, end=None):
    from datetime import datetime
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S.%f")[:-3]
    if show:
        if end is not None:
            print(current_time, *info, end=end)
        else:
            print(current_time, *info)
    if log_path:
        f_log = open(log_path, 'a')
        print(current_time, *info, file=f_log)
        f_log.close()

def write_one_signal_file(signal_file, data):
    if os.path.exists(signal_file):
        assert os.path.isfile(signal_file)
        if data != '' and data is not None :
            with open(signal_file, 'w') as f:
                f.write(data)
                f.flush()
    else:
        directory_path = os.path.dirname(signal_file)
        os.makedirs(directory_path, exist_ok=True)
        # Create the file
        with open(signal_file, "w") as f:
            f.write("")
        assert os.path.isfile(signal_file)
        # set_full_access_recursive(directory_path)
    return data

def receive_text_from_arduino(ser, arduino_idx):
    data = None
    if ser.in_waiting > 0:
        data = ser.readline().decode("utf-8", errors="ignore").strip()
        if data:
            log_xalg(f"Received from Arduino {arduino_idx}:", data)
    return data

def valve_serial_send_port_1to12_jetson():
    try:
        SERIAL_PORT = 'COM5'  # Update this as needed
        SIGNAL_PATH = './arduino_code/valve_control/speed/speed_info.txt'
        log_xalg("Opening serial port:", SERIAL_PORT)
        with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) as ser:
            log_xalg('Serial connection established.')

            while True:
                time.sleep(0.5)
                data= receive_text_from_arduino(ser, arduino_idx=1)
                # write_one_signal_file(SIGNAL_PATH, data)

    except serial.SerialException as e:
        log_xalg(f"Serial error: {e}")
    except KeyboardInterrupt:
        log_xalg("\nStopped by user.")

def main():
    t1 = threading.Thread(target=valve_serial_send_port_1to12_jetson, args=())  
    t1.start()  
    while True:
        time.sleep(0.01)

if __name__ == '__main__':
    main()