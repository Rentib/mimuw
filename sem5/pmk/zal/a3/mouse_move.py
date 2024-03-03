import time
import serial
import subprocess

device = "/dev/ttyACM0"
baudrate = 9600
ser = serial.Serial(device, baudrate)
print(f"Reading from {device}. Press Ctrl+C to exit.")

running = True

def move_mouse(x, y, diff):
    print(f"Moving mouse by ({x}, {y})")
    subprocess.run([
        "ydotool", "mousemove", "--", str(x), str(y)
    ])


start_time = time.time()
while running:
    try:
        data = ser.readline().decode('utf-8')
        y, x = data.split(' ')
        x = int(x)
        y = -int(y)
        diff = time.time() - start_time
        move_mouse(x, y, diff)
        start_time = time.time()
    except KeyboardInterrupt:
        running = False
    except:
        print("Error reading serial data.")

ser.close()
