#!/usr/bin/env python
# coding: utf-8
# @author Avijit Roy



#Importing the required module
import numpy as np
import pandas as pd
import os
import datetime
import matplotlib.pyplot as plt


x = datetime.datetime.now()
now = str(x)[0:10] 





df = pd.read_excel("../Dataset/UP100/Disc1/DATA DACOITY.xlsx")
zones = (df.Zone.unique())
for zone in zones:
    ruh_m = plt.imread('../Figure/%s.png'%(zone))
    new_df = df[df["Zone"] == zone]
    BBox = ((new_df.LONG.min(), new_df.LONG.max(), new_df.LAT.min(), new_df.LAT.max()))
    print(zone)
    print(new_df.head())
    print(BBox)



    fig, ax = plt.subplots(figsize = (8,7))
    ax.scatter(df['LONG'],df['LAT'] , zorder=1, alpha= 1, c='b', s=10)
    ax.set_title('Plotting Spatial Data on %s Map'%(zone))
    ax.set_xlim(BBox[0],BBox[1])
    ax.set_ylim(BBox[2],BBox[3])
    ax.imshow(ruh_m, zorder=0, extent = BBox, aspect= 'equal')
    if not os.path.exists("../Figure/%s"%(now)):
        os.makedirs("../Figure/%s"%(now))
    
    plt.savefig("../Figure/%s/%s.png"%(now,zone))
    plt.close()