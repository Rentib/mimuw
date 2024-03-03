def read_until(s, suffix):
    res = b''
    while not res.endswith(suffix):
        # quite slow, but should be enough for us
        d = s.recv(1)
        if len(d) == 0:
            raise EOFError()
        res += d
    return res


def xor(a, b):
    return bytes([ac ^ bc for ac, bc in zip(a, b)])


def split_by(data, cnt):
    return [data[i : i+cnt] for i in range(0, len(data), cnt)]


def pad(msg):
    byte = 16 - len(msg) % 16
    return msg + bytes([byte] * byte)


def unpad(msg):
    if not msg:
        return b''
    return msg[:-msg[-1]]
