#!/usr/bin/python

import socket
import struct
import numpy as np

udp_ip='127.0.0.1'
udp_port=2001
size=8208 #packet size

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind((udp_ip, udp_port))

for i in range(8):
    data, addr = sock.recvfrom(size)

    a = np.array(struct.unpack("<8208B", data), dtype=np.uint8)
    #a = a[8:-8] 	# skip 8-byte header and footer
    counter = struct.unpack("<1Q", data[0:8])
    counter = (counter[0] & 0x0000FFFFFFFFFFFF)
    print counter,
    for j in range(24):
        print a[j],
    print "\n",
    #a = np.array(struct.unpack('<4104H', data), dtype=np.int16)
    #a = a[4:-4] 	# skip 8-byte header and footer
    #XX = a[0::4]
    #YY = a[1::4]
    #XYre = a[2::4]
    #XYim = a[3::4]

    #print XX[0], XX[1], YY[0], YY[1]

sock.close()

