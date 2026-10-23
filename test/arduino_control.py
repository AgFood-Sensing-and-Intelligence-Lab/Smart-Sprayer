import serial 
import time
from pathlib import Path
import os
import time

#with serial.Serial('/dev/ttyACM1', 9600, timeout=1) as ser:
with serial.Serial('COM5', 9600, timeout=1) as ser:
    print('start serial send')
    cur_dir = Path.cwd()
    signal_read_file_path = os.path.join(str(cur_dir), 'sensing', 'signal_read.txt')
    signal_write_file_path = os.path.join(str(cur_dir), 'sensing', 'signal_write.txt')
    signal_new_sample_path = os.path.join(str(cur_dir), 'sensing', 'signal_new_sample.txt')
    # os.makedirs(signal_read_file_path, exist_ok=True)
    # os.makedirs(signal_write_file_path, exist_ok=True)
    with open(signal_read_file_path, 'w') as f:
        f.write('')
    with open(signal_write_file_path, 'w') as f:
        f.write('0')
    signal_new_sample = False
    while True:
        if ser.in_waiting > 0:
            data = ser.readline().decode("ascii")
            data = data.split('\r\n')[0]
            print('receive:', data, 'signal_new_sample', signal_new_sample)

            if not signal_new_sample:
                with open(signal_read_file_path, 'w') as f:
                    f.write(data)
                    signal_new_sample = True
            else:
                with open(signal_new_sample_path, 'r') as f:
                    data_new_sample = f.read()
                    if data_new_sample == '':
                        signal_new_sample = False
                        print('signal_new_sample', signal_new_sample)
        signal_used = False
        if ser.out_waiting == 0:
            with open(signal_write_file_path, 'r') as f:
                send_txt = f.read()
                if send_txt in '1':
                    print('send_txt:', send_txt)
                    ser.write(bytes('1', 'utf-8'))
                    signal_used = True
                if send_txt in '2':
                    print('send_txt:', send_txt)
                    ser.write(bytes('2', 'utf-8'))
                    signal_used = True
            if signal_used:
                with open(signal_write_file_path, 'w') as f:
                    f.write('0')
                    signal_used = False
        time.sleep(0.01)