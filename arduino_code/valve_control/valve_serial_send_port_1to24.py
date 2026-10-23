import serial
import time
import os
import threading
import sys

"""
For Linux:
    ls /dev/ttyACM*
    sudo chmod a+rw /dev/ttyACM0
    python arduino_code/valve_control/valve_serial_send_port1.py

For Windows:
    Change SERIAL_PORT to something like 'COM8'
"""
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

BAUD_RATE = 115200
PREFIX = 'signal_nozzle_'

# FIX 1: Disabled writing to disk. Only prints to terminal to save I/O bandwidth.
def log_xalg(*info, show=True, end=None):
    from datetime import datetime
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S.%f")[:-3]
    if show:
        if end is not None:
            print(current_time, *info, end=end)
        else:
            print(current_time, *info)

# FIX 2: Optimized file reading. Only opens files that actually have data.
def check_one_signal_file(signal_file, valve_idx):
    command = ''
    try:
        if os.path.exists(signal_file):
            if os.path.getsize(signal_file) > 0:
                valve_txt = valve_idx_to_text_dict[valve_idx]
                # command = f'{valve_txt}-1'
                with open(signal_file, 'r') as f:
                    command = f.read().strip()
                if command != '':
                    with open(signal_file, 'w') as f:
                        f.write('')
                        f.flush()
        else:
            directory_path = os.path.dirname(signal_file)
            os.makedirs(directory_path, exist_ok=True)
            with open(signal_file, "w") as f:
                f.write("")
    except PermissionError:
        # The main program is currently writing to this file. 
        # We will just ignore it and check again on the next loop.
        pass
    except Exception as e:
        # Catch any other unexpected file system errors
        pass
        
    return command

def receive_text_from_arduino(ser, arduino_idx):
    if ser.in_waiting > 0:
        data = ser.readline().decode("utf-8", errors="ignore").strip()
        if data:
            log_xalg(f"Received from Arduino {arduino_idx}:", data)

def valve_serial_send_port_1to12():
    try:
        valve_idxes = range(2, 26, 2)  # Even indices
        SERIAL_PORT = 'COM7'  # Update this as needed
        SIGNAL_DIR = './arduino_code/valve_control/signal/even'
        
        log_xalg("Opening serial port:", SERIAL_PORT)
        arduino_idx = 1
        with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) as ser:
            log_xalg('Serial connection established on', SERIAL_PORT)

            while True:
                # FIX 3: Yield CPU. 10ms sleep prevents starving the main camera program.
                time.sleep(0.01)
                # receive_text_from_arduino(ser, arduino_idx=arduino_idx)
                
                for valve_idx in valve_idxes:
                    signal_file = os.path.join(SIGNAL_DIR, f"{PREFIX}{valve_idx}.txt")
                    command = ''
                    command = check_one_signal_file(signal_file, valve_idx)
                    
                    if command != '':
                        ser.write((command + '\n').encode('utf-8'))
                        log_xalg('Sent to Arduino', arduino_idx, ':', command)

    except serial.SerialException as e:
        log_xalg(f"Serial error on port 1: {e}")
    except KeyboardInterrupt:
        pass

def valve_serial_send_port_13to24():
    try:
        valve_idxes = range(1, 24, 2)  # Odd indices
        SERIAL_PORT = 'COM8'  # Update this as needed
        SIGNAL_DIR = './arduino_code/valve_control/signal/odd'
        
        log_xalg("Opening serial port:", SERIAL_PORT)
        arduino_idx = 2
        with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) as ser:
            log_xalg('Serial connection established on', SERIAL_PORT)
            
            while True:
                # FIX 3: Yield CPU. 10ms sleep prevents starving the main camera program.
                time.sleep(0.01)
                # receive_text_from_arduino(ser, arduino_idx=arduino_idx)
                
                for valve_idx in valve_idxes:
                    signal_file = os.path.join(SIGNAL_DIR, f"{PREFIX}{valve_idx}.txt")
                    command = ''
                    command = check_one_signal_file(signal_file, valve_idx)
                    
                    if command != '':
                        ser.write((command + '\n').encode('utf-8'))
                        log_xalg('Sent to Arduino', arduino_idx, ':', command)

    except serial.SerialException as e:
        log_xalg(f"Serial error on port 2: {e}")
    except KeyboardInterrupt:
        pass

def main():
    t1 = threading.Thread(target=valve_serial_send_port_1to12, daemon=True)  
    t1.start()  
    t2 = threading.Thread(target=valve_serial_send_port_13to24, daemon=True)  
    t2.start()  
    
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        log_xalg("\nStopped by user.")

if __name__ == '__main__':
    main()