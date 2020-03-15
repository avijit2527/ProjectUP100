#!/usr/bin/env python
# coding: utf-8
# @author Avijit Roy



#Importing the required module
import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pickle

from Utilities import Utilities as ut


x = datetime.now()
now = str(x)[0:10] 


df1 = pd.read_excel("../Dataset/UP100/Disc1/DATA DACOITY.xlsx") 
df2 = pd.read_excel("../Dataset/UP100/Disc1/DATA ROBBERY.xlsx")
df3 = pd.read_excel("../Dataset/UP100/Disc1/THEFT DATA.xlsx")
frames = [df1, df2, df3]
df = pd.concat(frames)
convert_dict = {'LAT': float, 
                'LONG': float
               } 

zones = (df.Zone.unique())
zone = "KANPUR CITY"
#ruh_m = plt.imread('../Figure/%s.png'%(zone))
new_df = df[df["District"] == zone]
new_df = new_df.astype(convert_dict)
new_df = new_df.reindex() 
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
#ax.imshow(ruh_m, zorder=0, extent = BBox, aspect= 'equal')
if not os.path.exists("../Figure/%s"%(now)):
    os.makedirs("../Figure/%s"%(now))
    
plt.savefig("../Figure/%s/%s.png"%(now,zone))
plt.close()
     
latLong = (ut.lat_long_to_grid(new_df["LAT"],new_df["LONG"],13,10))
datetime = (pd.to_datetime(new_df['Date/Time']).apply(lambda dt: datetime(dt.year, dt.month, dt.day, dt.hour)).values)
print(datetime.size)
rewardStateMap = {}
for x,y,z in zip(latLong[0].tolist(),latLong[1].tolist(),datetime):
    #print(x,y,z)
    if z in rewardStateMap:
        rewardStateMap[z].append([x,y])      
    else:
        rewardStateMap[z] = [[x,y]]
#print(type(rewardStateMap))
with open("../Files/rewardMap", 'wb') as fp:
    pickle.dump(rewardStateMap, fp, protocol=pickle.HIGHEST_PROTOCOL)
#np.save("../Files/rewardMap",rewardStateMap)

dti = pd.date_range('2019-01-12', periods=2*24, freq='H').to_frame()
print(dti)
dti.to_excel("../Dataset/DateRange.xlsx")