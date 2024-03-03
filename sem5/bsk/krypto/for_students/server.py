#!/usr/bin/env python3
from socketserver import TCPServer, ThreadingMixIn, BaseRequestHandler
import hashlib
# from Crypto.Cipher import AES
# from Crypto.Random import get_random_bytes, random
from Cryptodome.Cipher import AES
from Cryptodome.Random import get_random_bytes, random

from utils import read_until, pad, unpad
from secrets import FLAG1, FLAG2, FLAG3


class ThreadedTCPServer(ThreadingMixIn, TCPServer):
    allow_reuse_address = True


class TCPHandler(BaseRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def read_line(self):
        return read_until(self.sock, b'\n').decode().strip()


    def read_line_hex(self):
        return bytes.fromhex(self.read_line())


    def send(self, text):
        self.sock.sendall(text.encode())


    def send_line(self, text):
        self.sock.sendall((text + '\n').encode())


    def send_line_hex(self, data):
        self.sock.sendall((data.hex() + '\n').encode())


    def handle(self):
        self.sock = self.request
        print(f'{self.client_address} connected!')
        try:
            self.send('Welcome to the crypto task game!\n')
            self.send('''
Which challenge do you want to solve?
 1) NCG
 2) Block cipher (two flags here)
''')
            while True:
                self.send('> ')
                choice = self.read_line()
                if choice in ['1', '2']:
                    break
                self.send('???\n')

            if choice == '1':
                self.chall1()
            elif choice == '2':
                self.chall2()
            else:
                raise RuntimeError("Not a valid challenge")
            self.send('bye!\n')
        except (EOFError, ConnectionResetError):
            pass


    def finish(self):
        print(f'{self.client_address} disconnected!')


    # -------- NCG challenge --------
    # LCG was easy, because it was linear. What if it wasn't? Welcome to the
    # very secure Non-linear Congruential Generator™!
    class NcgPrng:
        def __init__(self):
            self.state = random.randint(0, 2**64-1)
            self.a = random.randint(0, 2**64-1)
            self.c = random.randint(0, 2**64-1)
            self.m = 2**64

        def next(self):
            """Gets a next 64bit random number."""
            self.state = (self.state * self.a ^ self.c) % self.m
            return self.state


    def chall1(self):
        rng = self.NcgPrng()
        for i in range(5):
            self.send_line(f'{rng.next()}')
        self.send_line('What\'s next?')
        answer = self.read_line().strip()
        try:
            answer = int(answer)
        except ValueError:
            self.send_line('malformed answer')
            return
        if answer == rng.next():
            self.send_line(FLAG1)
        else:
            self.send_line('nope')


    # -------- Block cipher challenge --------
    def send_encrypted(self, key, msg):
        iv = b'so_random_iv_wow'
        aes = AES.new(key, AES.MODE_CBC, iv)
        self.send_line_hex(iv + aes.encrypt(pad(msg)))


    def recv_encrypted(self, key):
        msg = self.read_line_hex()
        aes = AES.new(key, AES.MODE_CBC, msg[:16])
        return unpad(aes.decrypt(msg[16:]))


    def chall2(self):
        key = get_random_bytes(16)
        self.send_encrypted(key, b'Hello')
        while True:
            try:
                msg = self.recv_encrypted(key).strip()
                print(msg)
                if msg.startswith(b'hash?'):
                    print('hash?')
                    self.send_encrypted(key, hashlib.sha256(msg[5:]).digest())
                elif msg.startswith(b'flag?'):
                    print('flag?')
                    self.send_encrypted(key, FLAG2.encode())
                elif msg.startswith(b'FLAG!'):
                    print('FLAG!')
                    self.send_encrypted(key, FLAG3.encode())
                else:
                    print('unknown command')
                    self.send_encrypted(key, b'unknown command')
            except Exception:
                print('exception')
                self.send_line('???')
                break


def main():
    PORT = 13371
    try:
        with ThreadedTCPServer(('0.0.0.0', PORT), TCPHandler) as server:
            print(f'Server started on port {PORT}')
            server.serve_forever()
    except KeyboardInterrupt: # Ctrl-C
        print('Exiting...')

if __name__ == '__main__':
    main()
