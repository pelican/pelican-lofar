#!/usr/bin/python

import socket
import struct
import numpy as np

ip = "127.0.0.1"
port = 9999
size = 8208         # packet size

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind((ip, port))

for i in range(32):
    data, addr = sock.recvfrom(size)
    header = struct.unpack(">Q", data[0:8])[0]
    integCount = header >> 16
    sq = struct.unpack(">B", data[6:7])[0]
    #print sq, integCount
    for j in range(8):
        x = struct.unpack(">B", data[j:j+1])[0]
        print x,
    print ""

sock.close()

