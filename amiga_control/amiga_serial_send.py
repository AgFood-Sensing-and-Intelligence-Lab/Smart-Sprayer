import serial 
import time
# sudo chmod a+rw /dev/ttyACM2

with serial.Serial('/dev/ttyACM2', 115200, timeout=1) as ser:
    first_xxx = '0'
    print('serial send')
    while True:
        # xxx=input("LED on?")
        if ser.in_waiting > 0:
            data = ser.readline().decode("ascii")
            print(data)
            # with open('speed.txt', 'w') as f:
            #     f.write(data)

        if ser.out_waiting == 0:
            with open('./amiga_control/signal.txt', 'r') as f:
                xxx = f.read()
            if first_xxx == xxx:
                continue
            else:
                first_xxx = xxx
            if xxx == '0':
                ser.write(bytes('0', 'utf-8'))
            if xxx == '1':
                # ser.write(bytes('1', 'utf-8'))
                print('no port 1')
            if xxx == '2':
                ser.write(bytes('2', 'utf-8'))
            if xxx == '3':
                ser.write(bytes('3', 'utf-8'))
            if xxx == '4':
                ser.write(bytes('4', 'utf-8'))
            if xxx == '5':
                ser.write(bytes('5', 'utf-8'))
            if xxx == '6':
                ser.write(bytes('6', 'utf-8'))
            if xxx == '7':
                ser.write(bytes('7', 'utf-8'))
            if xxx == '8':
                ser.write(bytes('8', 'utf-8'))
            if xxx == '9':
                ser.write(bytes('9', 'utf-8'))
            if xxx == '10':
                ser.write(bytes('a', 'utf-8'))
            if xxx == '11':
                ser.write(bytes('b', 'utf-8'))
            if xxx == '12':
                ser.write(bytes('c', 'utf-8'))
            if xxx == '13':
                ser.write(bytes('d', 'utf-8'))
            print('input:', xxx)
            with open('./signal.txt', 'w') as f:
                xxx = f.write('')
        time.sleep(0.1)
        # print(ser.readline())
