#!/usr/bin/env python
import sys
import math

#print len(sys.argv)
#print sys.argv
cl=sys.argv
if len(cl) !=3:
    sys.exit("Please provide J2000 coords at HH MM SS.SSS DD MM SS.SS")

RAin=cl[1]
DECin=cl[2]
RAa=RAin.split(':')
DECa=DECin.split(':')
# print RAa, DECa


RAH=float(RAa[0])
RAM=float(RAa[1])
RAS=float(RAa[2])
decD=float(DECa[0])
decM=float(DECa[1])
decS=float(DECa[2])

RA=(RAH/24+RAM/24/60+RAS/86400)*2*math.pi
dec=(decD+decM/60+decS/3600)*math.pi/180.0
print
print
print '%.6f,%.6f' % (RA, dec)
