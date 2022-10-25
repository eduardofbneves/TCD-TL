# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

"""
    Column1: Device ID
    Column2: accelerometer x
    Column3: accelerometer y
    Column4: accelerometer z
    Column5: gyroscope x
    Column6: gyroscope y
    Column7: gyroscope z
    Column8: magnetometer x
    Column9: magnetometer y
    Column10: magnetometer z
    Column11: Timestamp
    Column12: Activity Label
"""

import csv
import numpy as np
import pandas as pd 

loc = []

for i in range (0,14):
    loc.append("dev2_wrist/part" + str(i) + "dev2.csv")
    
print(loc)

df = pd.read_csv('dev2_wrist/part0dev2.csv', sep=',', header=None)
array = df.to_numpy()



"""
with open('dev2_wrist/part0dev2.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')

type(spamreader)
"""
