"""
Numer indeksu: 438247
Nazwa zadania: Zadanie zaliczeniowe - kryptografia
Flaga1:        flag{still-not-a-csprng} 
Flaga2:        flag{sh0rt-fl4G}
Flaga3:        flag{p4dding-1s-h4rd-but-re4ly-just-s1gn-y0ur-c1phert3xts}
opis:          opis.md
"""

from pwn import *

context.log_level = 'error' # pwn ma być cicho

server = sys.argv[1]
port = int(sys.argv[2])

flag_bytes = [bytes([i]) for i in range(0, 256)] # TODO: zmienić żeby były tylko znaki występujące we flagach

def get_connection(): return remote(server, port)

def xor(a, b): return bytes([ac ^ bc for ac, bc in zip(a, b)])

def pad(msg):
    byte = 16 - len(msg) % 16
    return msg + bytes([byte] * byte)

################################################################################
#                                FLAGA 1                                       #
################################################################################

while True:
    conn = get_connection()
    for _ in range(5): conn.recvline()
    conn.recvn(2)
    conn.sendline(b'1')

    # challange 1
    # dane: m=1<<64, s1, s2, s3, s4, s5
    # szukamy a, c takie, że:
    # s_{i+1} = (s_i * a ^ c) mod m
    # m=1<<64, więc powinno działać dla każdego k=0..64 mod 1<<k
    # idziemy od najmniejszych bitów, testujemy kombinacje (a, c) i sprawdzamy, czy zachodzą równania
    # s_{i+1} mod 1<<k = (s_i * a ^ c) mod 1<<k \forall_{k=0..64}
    # cały czas będie dość mało możliwości tak rzędu O(1), więc znajdziemy odp w O(64*1) = O(1)

    s = [int(conn.recvline().decode()) for _ in range(5)]
    conn.recvline()

    possible = [(0,0)]
    for k in range(0, 65):
        old, cur, possible = possible, [], []
        # dodajemy wszystkie możliwe kombinacje bitów a, c
        for (a,c) in old:
            cur.append((a | (0<<k), c | (0<<k)))
            cur.append((a | (0<<k), c | (1<<k)))
            cur.append((a | (1<<k), c | (0<<k)))
            cur.append((a | (1<<k), c | (1<<k)))
        # wszystkie 4 równania muszą zachodzić mod 2^k
        possible = [(a, c) for (a, c) in cur if all(s[i + 1] % (1<<k) == (s[i] * a ^ c) % (1<<k) for i in range(4))]
    (a,c) = possible[0]
    answer = (s[4] * (a % (1<<64)) ^ (c % (1 << 64))) % (1<<64)
    conn.sendline(str(answer).encode())
    flag1 = conn.recvline().decode().strip()
    conn.close()
    # CHYBA jest o równanie za mało, więc czasami nie znajdzie poprawnej odpowiedzi
    # wtedy trzeba spróbować jeszcze raz
    if 'flag{' in flag1:
        print(flag1)
        break

################################################################################
#                                FLAGA 2                                       #
################################################################################

def get_block_based_on_5_bytes(conn, plaintext, iv, enc, fill, query, msg):
    # funkcja query to wyrocznia
    k = 0
    while len(plaintext) < 16 and '}' not in plaintext.decode():
        k += 1
        for i in flag_bytes: # sprawdzamy wszystkie możliwe bajty
            l = len(plaintext) + 1
            iv2 = xor(xor(plaintext + i, iv[:l]), fill * k + query) # bierzemy nową wiadomość z dopisanym nowym bajtem
            conn.sendline((iv2.hex() + iv[l:].hex() + enc.hex()).encode())
            tmp = conn.recvline().decode().strip()
            if tmp == msg: # jak response się zgadza, to mamy poprawny bajt
                plaintext += i
                break
    return plaintext

conn = get_connection()
for _ in range(5): conn.recvline()
conn.recvn(2)
conn.sendline(b'2') # challange 2

# odbieramy zaszyfrowane 'Hello'
msg = conn.recvline().decode().strip()
b = bytes.fromhex(msg)
iv, enc = b[:16], b[16:]

# wysylamy "flag?"
e = xor(pad(b'Hello'), iv)
iv = xor(e, pad(b'flag?'))
conn.sendline((iv.hex() + enc.hex()).encode())

# odbieramy zaszyfrowaną flagę
msg = conn.recvline().decode().strip()
b = bytes.fromhex(msg)
iv, enc = b[:16], b[16:]

flag2 = get_block_based_on_5_bytes(conn, b"flag{", iv, enc, b' ', b'flag?', msg)
print(flag2.decode())

conn.close()

################################################################################
#                                FLAGA 3                                       #
################################################################################

conn = get_connection()
for _ in range(5): conn.recvline()
conn.recvn(2)
conn.sendline(b'2')

# odbieramy zaszyfrowane 'Hello'
msg = conn.recvline().decode().strip()
b = bytes.fromhex(msg)
iv, enc = b[:16], b[16:]

# wysylamy "FLAG!"
e = xor(pad(b'Hello'), iv)
iv = xor(e, pad(b'FLAG!'))
conn.sendline((iv.hex() + enc.hex()).encode())

# odbieramy zaszyfrowaną flagę
msg = conn.recvline().decode().strip()
b = bytes.fromhex(msg)
iv, enc = b[:16], b[16:]

flag3 = get_block_based_on_5_bytes(conn, b"flag{", iv, enc, b' ', b'FLAG!', msg)

# flaga może być dłuższa niż 1 blok, więc musimy znaleźć kolejne bloki
ct = enc
block_num = 1
while '}' not in flag3.decode():
    # używamy 'hash?' jako wyroczni żeby znaleźć pierwsze 5 bajtów bloku
    # potem używamy 'FLAG!' żeby wyznaczyć kolejne bajty na podstawie pierwszych 5 (tak jak w drugiej fladze)

    # IVa + CT_{i-1} + CT_i + IVb + CT_{i-1} -> hash? + PT_i
    prefix=b'hash?'
    ctim1 = '' # CT_{i-j}
    for j in range(5):
        ctim1 = enc[16*(block_num-1):16*block_num]
        cti   = enc[16*block_num:16*(block_num+1)]
        iva   = xor(xor(flag3[-16:], iv), 11 * b' ' + b'hash?')
        ivb   = xor(xor(flag3[-16:], iv), 15 * b' ' + bytes([32 + 15-j]))

        conn.sendline((iva.hex() + ctim1.hex() + cti.hex() + ivb.hex() + ctim1.hex()).encode())
        msg2 = conn.recvline().decode().strip()
        b2 = bytes.fromhex(msg2)
        iv2, enc2 = b2[:16], b2[16:]

        for i in flag_bytes:
            iv2 = xor(xor(flag3[-16:], iv), (8-j) * b' ' + prefix + i + bytes([2,2]))
            conn.sendline((iv2.hex() + ctim1.hex()).encode())
            tmp = conn.recvline().decode().strip()
            if tmp == msg2: # jak response się zgadza, to mamy poprawny bajt
                prefix += i
                break
        if prefix[-1] == b'}'[0]: # jeśli to już koniec to dodajemy ostatni blok
            break
    if prefix[-1] == b'}'[0]: # jeśli to już koniec to dodajemy ostatni blok
        flag3 += prefix[5:]
        break

    iv = ctim1 # IV dla i-tego bloku to CT_{i-1}
    flag3 += get_block_based_on_5_bytes(conn, prefix[5:], iv, enc[16*block_num:], b' ', b'FLAG!', msg)
    block_num += 1

print(flag3.decode())
conn.close()
