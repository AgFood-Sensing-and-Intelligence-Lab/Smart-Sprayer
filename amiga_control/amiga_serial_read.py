import serial 
import time
'''
sudo chmod a+rw /dev/ttyACM2
python amiga_control/amiga_serial_read.py 
'''

with serial.Serial('/dev/ttyACM0', 115200, timeout=1) as ser:
    print('serial send')
    while True:
        # if ser.in_waiting > 0:
        if True:
            data = ser.readline().decode("ascii")
            print(data)
            with open('./amiga_control/states.txt', 'w') as f:
                xxx = f.write(data)
        time.sleep(0.1)
        # print(ser.readline())
