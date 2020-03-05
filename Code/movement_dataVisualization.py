#!/usr/bin/env python
# coding: utf-8
# @author Avijit Roy



#Importing the required module
import numpy as np
import pandas as pd
import os
import datetime
import matplotlib.pyplot as plt

from Utilities import Utilities as ut


x = datetime.datetime.now()
now = str(x)[0:10] 



df = pd.read_excel("../Dataset/UP100/Disc1/DATA DACOITY.xlsx")
convert_dict = {'LAT': float, 
                'LONG': float
               } 

zones = (df.Zone.unique())
zone = "KANPUR"
ruh_m = plt.imread('../Figure/%s.png'%(zone))
new_df = df[df["Zone"] == zone]
new_df = new_df.astype(convert_dict) 
new_df.to_excel("../Dataset/%s.xlsx"%(zone)) 
BBox = ((new_df.LONG.min(), new_df.LONG.max(), new_df.LAT.min(), new_df.LAT.max()))
print(new_df["Zone"].size)
print(new_df.head())
print(BBox)

fig, ax = plt.subplots(figsize = (8,7))
ax.scatter(new_df['LONG'],new_df['LAT'] , zorder=1, alpha= 1, c='b', s=10)
ax.set_title('Plotting Spatial Data on %s Map'%(zone))
ax.set_xlim(BBox[0],BBox[1])
ax.set_ylim(BBox[2],BBox[3])
ax.imshow(ruh_m, zorder=0, extent = BBox, aspect= 'equal')
if not os.path.exists("../Figure/%s"%(now)):
    os.makedirs("../Figure/%s"%(now))
    
plt.savefig("../Figure/%s/%s.png"%(now,zone))
plt.close()
    
#print(ut.lat_long_to_grid(new_df["LAT"],new_df["LONG"],133,100))
print(pd.to_datetime(new_df['Date/Time']))
