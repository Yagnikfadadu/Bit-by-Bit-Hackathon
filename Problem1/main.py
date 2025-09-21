import chipwhisperer as cw
import random
import csv
import os
import numpy as np

# --- 1. CONNECT TO HARDWARE ---
try:
    scope = cw.scope()
    target = cw.target(scope)
    print("Hardware connected.")
except IOError as e:
    print(f"Could not connect to ChipWhisperer hardware: {e}")
    exit()

scope.default_setup()

# --- 2. PROGRAM THE TARGET ---
prog = cw.programmers.STM32FProgrammer
print("Programming target...")
cw.program_target(scope, prog, "simpleserial_rsa-CW308_STM32F3.hex")
print("Programming done.")

# --- 3. CONFIGURE CAPTURE SETTINGS ---
scope.clock.adc_src = "clkgen_x1"
scope.adc.samples = 800  # adjust as needed

# --- 4. CAPTURE ONCE ---
RSA_N = 64507
NUM_TRACES = 300   # total traces to capture once

random.seed(0xCAFEBABE)
rows = []

# Create header for CSV
header = ['ciphertext_hex'] + [f'sample_{i + 1}' for i in range(scope.adc.samples)]

print(f"Starting capture of {NUM_TRACES} traces...")
for i in range(NUM_TRACES):
    c_int = random.randint(0, RSA_N - 1)
    ct_bytes = c_int.to_bytes(2, 'big')

    scope.arm()
    target.simpleserial_write('p', ct_bytes)
    ret = scope.capture()

    if ret:
        print(f"\nCapture timed out for trace {i + 1}")
        continue

    trace = scope.get_last_trace()
    rows.append([ct_bytes.hex()] + trace.tolist())
    print(f"Captured trace {i + 1}/{NUM_TRACES}", end='\r')

print("\nCapture complete.")

# --- SAVE TO ONE CSV ---
OUT_CSV = "single_trace.csv"
with open(OUT_CSV, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)

print(f"All {NUM_TRACES} traces saved to {OUT_CSV}")

# --- DISCONNECT ---
scope.dis()
target.dis()
print("Disconnected from hardware.")