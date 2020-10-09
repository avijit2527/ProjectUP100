#!/usr/bin/env python
# coding: utf-8
# @author Avijit Roy


import numpy as np
import random

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import pandas as pd
import folium
import numpy as np
from pymongo import MongoClient
from folium import plugins

def getCrimes(db,zone, leftLat,leftLng,rightLat,rightLng):
    crimes = db.crimes.find({'zone':zone, 'lat' : {'$lte': leftLat, '$gte': rightLat}, 'lng' : {'$lte': rightLng, '$gte': leftLng}});
    df = pd.DataFrame(crimes);
    return df;

client = MongoClient(port=27017);    
db=client.conFusion;

df_crime = getCrimes(db, "KNC", 26.60,79.70,26.25,80.09);

df_crime = df_crime[['lat', 'lng']]

X = (df_crime.to_numpy());

dbscan = DBSCAN(eps=0.0025, min_samples=2).fit(X)
core_samples_mask = np.zeros_like(dbscan.labels_, dtype=bool)
core_samples_mask[dbscan.core_sample_indices_] = True
labels = dbscan.labels_
print(labels);

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)

filter_arr = [];
for x in labels:
	filter_arr.append(x != -1);

cluster_arr = X[filter_arr];

idx = np.random.randint(len(cluster_arr), size=24);
final_points = cluster_arr[idx,:];

print(final_points);

df = pd.DataFrame(final_points, columns=["Latitude", "Longitude"])

#print(df.head())
df.to_excel("../Results/trajectory_DBSCAN_" + ".xlsx")