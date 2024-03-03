'''
Łączenie na GNU / Linux:
    - przydatne paczki (archlinux):
        - bluez
        - bluez-utils
        - bluez-rfcomm
    - wymagana jest obsługa protokołu RFCOMM
    - należy wcześniej sparować urządzenie (kod 1234)
        - bluetoothctl scan on
        - bluetoothctl pair <bt_addr>
        - bluetoothctl scan off
    - dopiero po sparowaniu ten skrypt ma szansę zadzaiałać
'''

import bluetooth
import threading

# Adres bluetooth mikrokontrolera.
bt_addr = "98:D3:51:FD:A3:F0"

# Będziemy się łączyć po porcie 1.
port = 0x1

# Protokół obsługiwany przez mikrokontroler to RFCOMM.
sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)

# Funkcja do ciągłego odbioru danych.
def recv_loop():
    while True:
        data = sock.recv(512)
        try:
            print(data.decode('utf-8'), end="")
        except UnicodeDecodeError:
            print("[!] couldn't decode received data in utf-8")

# Funkcja do ciągłego odczytu danych z wejścia i ich wysyłania.
def send_loop():
    while True:
        data = input("> ").strip()
        try:
            sock.send(data.encode('utf-8'))
        except:
            print("[!] couldn't encode provided data in utf-8")

if __name__ == "__main__":
    print(f"[i] Connecting to microcontroller {bt_addr} port {port}...")

    try:
        # Zwiąż gniazdo sieciowe z adresem mikrokontrolera i wybranym portem.
        sock.connect((bt_addr, port))
    except KeyboardInterrupt:
        print("[i] exiting...")
        sock.close()

    print("[i] ...connnected!")

    recv_loop_thread = threading.Thread(target=recv_loop)
    recv_loop_thread.daemon = True
    recv_loop_thread.start()
    try:
        send_loop()
    except KeyboardInterrupt:
        print("[i] exiting...")
        sock.close()
