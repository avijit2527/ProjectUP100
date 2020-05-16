#!/usr/bin/env python
# coding: utf-8
# @author Avijit Roy

# Importing the required module
import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pickle

from Utilities import Utilities as ut


x = datetime.now()
now = str(x)[0:10]


df1 = pd.read_excel("../Dataset/Ayodhya_2019_20_Street_crime_events.xlsx")
df1['CreateTime'] = pd.to_datetime(df1['CreateTime'])
df1 = df1.dropna()
df1 = (df1[df1['Long'] > 81])
df1 = (df1[df1['Lat'] > 26.4])
# print(df1.columns)
# print(df1.dtypes)


BBox = ((df1.Long.min(), df1.Long.max(),
         df1.Lat.min(), df1.Lat.max()))

# print(BBox)


ruh_m = plt.imread('../Figure/Ayodha.png')
fig, ax = plt.subplots(figsize=(8, 7))
ax.scatter(df1['Long'], df1['Lat'], zorder=1, alpha=1, c='b', s=10)
ax.set_title('Plotting Spatial Data on Map')
ax.set_xlim(BBox[0], BBox[1])
ax.set_ylim(BBox[2], BBox[3])
ax.imshow(ruh_m, zorder=0, extent=BBox, aspect='equal')
if not os.path.exists("../Figure/%s" % (now)):
    os.makedirs("../Figure/%s" % (now))

plt.savefig("../Figure/%s.png" % (now))
plt.close()


latLong = (ut.lat_long_to_grid(df1["Lat"], df1["Long"], 35, 12))

latLong2D = []

for lat, lng in zip(latLong[0], latLong[1]):
    latLong2D.append([lat, lng])

np.save("../Files/latlongGrid",latLong2D)

print(latLong2D)



