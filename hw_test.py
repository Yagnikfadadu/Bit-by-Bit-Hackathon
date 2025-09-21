import chipwhisperer as cw
import random
import csv
import os
import numpy as np

# --- 1. CONNECT TO HARDWARE ---
try:
    # First and only time to connect
    scope = cw.scope()
    target = cw.target(scope)
    print("Hardware connected.")
except IOError as e:
    print(f"Could not connect to ChipWhisperer hardware: {e}")
    # Exit the script if hardware connection fails
    exit()

scope.default_setup()

# --- 2. PROGRAM THE TARGET ---
prog = cw.programmers.STM32FProgrammer
print("Programming target...")
cw.program_target(scope, prog, "simpleserial_rsa-CW308_STM32F3.hex")
print("Programming done.")

# --- 3. PERFORM THE ATTACK/VERIFICATION ---
# Use the already-connected 'scope' and 'target' objects
c_int = 26353
ct_bytes = c_int.to_bytes(2, 'big')

# These lines were causing the error. They are now removed.
# scope = cw.scope()
# target = cw.target(scope)

target.simpleserial_write('p', ct_bytes)
resp = target.simpleserial_read('r', 2)

decimal_value = int.from_bytes(resp, byteorder='big')
print("Decimal Value", decimal_value)

# --- 4. DISCONNECT (Best Practice) ---
scope.dis()
target.dis()
