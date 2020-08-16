#!/usr/bin/env python
# coding: utf-8
# @author Avijit Roy


# In[ ]:

#Importing the required module
import numpy as np
import pandas as pd
import os
import datetime
import matplotlib.pyplot as plt

class Utilities:
    def __init__(self):
        pass
    
    @staticmethod
    def lat_long_to_grid(df_lat,df_long, width, height):
        _, hist_lat = np.histogram(df_lat, bins = height)
        np.save("../Files/hist_lat",hist_lat)
        _, hist_long = np.histogram(df_long, bins = width)
        np.save("../Files/hist_long",hist_long)
        #print(np.digitize(df_long, hist_long))

        latLong2D = []
        for lat, lng in zip(np.digitize(df_lat, hist_lat), np.digitize(df_long, hist_long)):
            latLong2D.append([lat, lng])
        return list(latLong2D) , hist_lat, hist_long;
     
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
