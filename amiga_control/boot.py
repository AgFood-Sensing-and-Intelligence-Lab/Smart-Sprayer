# SPDX-FileCopyrightText: 2017 Limor Fried for Adafruit Industries
#
# SPDX-License-Identifier: MIT
'''
import storage
storage.erase_filesystem()
'''
# https://docs.circuitpython.org/en/9.0.x/shared-bindings/storage/index.html

"""CircuitPython Essentials Storage logging boot.py file"""
import board
import digitalio
import storage

# get_board_type() Adafruit Feather M4 CAN with same51j19a

# For Gemma M0, Trinket M0, Metro M0/M4 Express, ItsyBitsy M0/M4 Express
# switch = digitalio.DigitalInOut(board.D2)

# For Feather M0/M4 Express
switch = digitalio.DigitalInOut(board.D5)

# For Circuit Playground Express, Circuit Playground Bluefruit
# switch = digitalio.DigitalInOut(board.D7)

switch.direction = digitalio.Direction.INPUT
switch.pull = digitalio.Pull.UP

# If the switch pin is connected to ground CircuitPython can write to the drive
# storage.remount("/", readonly=switch.value)
# storage.remount("/", readonly=False, disable_concurrent_write_protection=True)
storage.remount("/", readonly=True)

# import usb_cdc
# usb_cdc.enable(data=True)