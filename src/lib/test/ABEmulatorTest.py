#!/usr/bin/python

import socket
import struct
import numpy as np

ip = '127.0.0.1'
port = 2001
size = 8208         # packet size

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind((ip, port))

for i in range(8):
    data, addr = sock.recvfrom(size)
    header = struct.unpack(">Q", data[0:8])[0]
    counter = struct.unpack("<1Q", data[0:8])
    counter = (counter[0] & 0x0000FFFFFFFFFFFF)
    integCount = header >> 16
    print integCount, counter

sock.close()

