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
import folium

from pymongo import MongoClient
import json
from pprint import pprint
from bson.objectid import ObjectId
import datetime


from Utilities import Utilities as ut



df1 = pd.read_excel("../Dataset/GOMTINAGAR_2020.xlsx");
df = df1[["latitude","longitude"]];
df = df.dropna();
print(df.size);
print(df.columns);

BBox = ((df.longitude.min(), df.longitude.max(),
         df.latitude.min(), df.latitude.max()))
print(BBox)

my_map = folium.Map(location=[26.40, 79.85], zoom_start=10)

subset = df[['latitude', 'longitude']]
tuples = [tuple(x) for x in subset.to_numpy()]
for mytuple in tuples:
    folium.Marker(mytuple, icon=folium.Icon( icon='car', prefix='fa')).add_to(my_map);
    print("kjgh");

my_map.add_child(folium.LatLngPopup());


my_map.save("../Results/traj_GOMTINAGAR.html")