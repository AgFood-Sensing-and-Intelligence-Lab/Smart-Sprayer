import serial
import time
import os
import numpy as np
import threading


"""
For Linux:
    ls /dev/ttyACM*
    sudo chmod a+rw /dev/ttyACM0
    python arduino_code/valve_control/valve_serial_send_port1.py

For Windows:
    Change SERIAL_PORT to something like 'COM4'
"""

BAUD_RATE = 9600
PREFIX = 'signal_nozzle_'

is_sending_signal_dict_1_12 = {}
is_sending_signal_dict_13_24 = {}
do_close_valve_dict = {}


log_path = './log/log.txt'

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

def set_full_access_recursive(folder_path):
    """
    Sets full read, write, and execute permissions (777) for all files and
    directories within the specified folder and its subdirectories.
    """
    # Define the desired permissions (octal 777 for rwx for owner, group, and others)
    permissions = 0o777

    for root, dirs, files in os.walk(folder_path):
        # Set permissions for the current directory
        try:
            os.chmod(root, permissions)
            log_xalg(f"Set permissions for directory: {root}")
        except OSError as e:
            log_xalg(f"Error setting permissions for directory {root}: {e}")

        # Set permissions for all files in the current directory
        for file_name in files:
            file_path = os.path.join(root, file_name)
            try:
                os.chmod(file_path, permissions)
                log_xalg(f"Set permissions for file: {file_path}")
            except OSError as e:
                log_xalg(f"Error setting permissions for file {file_path}: {e}")

def check_one_signal_file(signal_file):
    command = ''
    if os.path.exists(signal_file):
        assert os.path.isfile(signal_file)
        with open(signal_file, 'r') as f:
            command = f.read().strip()
        if command != '':
            with open(signal_file, 'w') as f:
                f.write('')
                f.flush()
    else:
        directory_path = os.path.dirname(signal_file)
        os.makedirs(directory_path, exist_ok=True)
        # Create the file
        with open(signal_file, "w") as f:
            f.write("")
        assert os.path.isfile(signal_file)
        # set_full_access_recursive(directory_path)
    return command

def receive_text_from_arduino(ser, arduino_idx):
    if ser.in_waiting > 0:
        data = ser.readline().decode("utf-8", errors="ignore").strip()
        if data:
            log_xalg(f"Received from Arduino {arduino_idx}:", data)

def close_valve_signal(nozzle_idx, valve_idx, is_sending_signal_dict):
    is_sending_signal_dict[nozzle_idx] = True
    global valvedo_close_valve_dict
    valve_opening_duration = 0.2
    time.sleep(valve_opening_duration)
    do_close_valve_dict[valve_idx] = nozzle_idx

def close_valve(nozzle_idx, valve_idx, is_sending_signal_dict, ser):
    close_command = f"{nozzle_idx}-0"
    log_xalg('Rewrite file to close valve:', close_command)
    ser.write((close_command + '\n').encode('utf-8'))
    do_close_valve_dict[valve_idx] = None
    is_sending_signal_dict[nozzle_idx] = False
    pass

def valve_serial_send_port_1to12():
    try:
        global is_sending_signal_dict_1_12, do_close_valve_dict
        valve_idxes = range(2, 26, 2)  # Valve indices from 1 to 12

        SERIAL_PORT = 'COM7'  # Update this as needed
        SIGNAL_DIR = './arduino_code/valve_control/signal/even'
        log_xalg("Opening serial port:", SERIAL_PORT)
        arduino_idx = 1
        with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) as ser:
            log_xalg('Serial connection established.')

            while True:
                time.sleep(0.001)
                receive_text_from_arduino(ser, arduino_idx=arduino_idx)
                for valve_idx in valve_idxes:
                    if valve_idx not in do_close_valve_dict:
                        do_close_valve_dict[valve_idx] = None
                    signal_file = os.path.join(SIGNAL_DIR, f"{PREFIX}{valve_idx}.txt")
                    # assert os.path.exists(signal_file)
                    command = check_one_signal_file(signal_file)
                    if command !='':
                        ser.write((command + '\n').encode('utf-8'))
                        log_xalg('Sent to Arduino', arduino_idx,':', command)
                        time.sleep(0.005)
                        nozzle_idx = command.split('-')[0]
                        if nozzle_idx not in is_sending_signal_dict_1_12:
                            is_sending_signal_dict_1_12[nozzle_idx] = False
                        # if is_sending_signal_dict_1_12[nozzle_idx] is False:
                        #     t1 = threading.Thread(target=close_valve_signal, args=(nozzle_idx, valve_idx, is_sending_signal_dict_1_12), daemon=True)  
                        #     t1.start()  
                for valve_idx in valve_idxes:
                    if do_close_valve_dict[valve_idx] is not None:
                        pass
                        # close_valve(nozzle_idx, valve_idx, is_sending_signal_dict_1_12, ser)
                        # time.sleep(0.005)

    except serial.SerialException as e:
        log_xalg(f"Serial error: {e}")
    except KeyboardInterrupt:
        log_xalg("\nStopped by user.")

def valve_serial_send_port_13to24():
    try:
        global is_sending_signal_dict_13_24, do_close_valve_dict
        valve_idxes = range(1, 24, 2)  
        SERIAL_PORT = 'COM8'  # Update this as needed
        SIGNAL_DIR = './arduino_code/valve_control/signal/odd'
        log_xalg("Opening serial port:", SERIAL_PORT)
        with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) as ser:
            log_xalg('Serial connection established.')
            arduino_idx = 2
            while True:
                time.sleep(0.001)
                receive_text_from_arduino(ser, arduino_idx=arduino_idx)
                for valve_idx in valve_idxes:
                    if valve_idx not in do_close_valve_dict:
                        do_close_valve_dict[valve_idx] = None
                    signal_file = os.path.join(SIGNAL_DIR, f"{PREFIX}{valve_idx}.txt")
                    # assert os.path.exists(signal_file)
                    command = check_one_signal_file(signal_file)
                    if command !='':
                        ser.write((command + '\n').encode('utf-8'))
                        log_xalg('Sent to Arduino', arduino_idx,':', command)
                        time.sleep(0.005)
                        nozzle_idx = command.split('-')[0]
                        if nozzle_idx not in is_sending_signal_dict_13_24:
                            is_sending_signal_dict_13_24[nozzle_idx] = False
                        # if is_sending_signal_dict_13_24[nozzle_idx] is False:
                        #     t1 = threading.Thread(target=close_valve_signal, args=(nozzle_idx, valve_idx, is_sending_signal_dict_13_24), daemon=True)  
                        #     t1.start()  
                for valve_idx in valve_idxes:
                    if do_close_valve_dict[valve_idx] is not None:
                        pass
                        # close_valve(nozzle_idx, valve_idx, is_sending_signal_dict_13_24, ser)
                        # time.sleep(0.005)

    except serial.SerialException as e:
        log_xalg(f"Serial error: {e}")
    except KeyboardInterrupt:
        log_xalg("\nStopped by user.")

def main():
    t1 = threading.Thread(target=valve_serial_send_port_1to12, args=(), daemon=True)  
    t1.start()  
    t2 = threading.Thread(target=valve_serial_send_port_13to24, args=(), daemon=True)  
    t2.start()  
    while True:
        time.sleep(0.01)

if __name__ == '__main__':
    main()