import pandas as pd
import folium
import numpy as np
from pymongo import MongoClient
from geopy.distance import distance

import random


from folium import plugins

def getCrimes(db,zone, leftLat,leftLng,rightLat,rightLng):
    crimes = db.crimes.find({'zone':zone, 'lat' : {'$lte': leftLat, '$gte': rightLat}, 'lng' : {'$lte': rightLng, '$gte': leftLng}});
    df = pd.DataFrame(crimes);
    return df;

def getDistance(tuples):
    tot_dist = 0.0;
    for i in range(len(tuples)-1):
        tot_dist += distance(tuples[i],tuples[i+1]).km;
    return tot_dist;


client = MongoClient(port=27017);    
db=client.conFusion;

df_crime = getCrimes(db, "KNC", 26.7459, 79.9290, 26.6883, 80.0508);

df = pd.read_excel("../Results/trajectory_-1_GMTNGR.xlsx")
df_crime = df_crime[['lat', 'lng']]
print(df_crime)

allAgents = df.AgentId.unique()
colormap = {0: "darkgreen", 1: "orange", 2: "lightgray", 3: "lightred", 4: "cadetblue", 5: "teal", 6: "purple", 7: "magenta", 8: "blue", 9: "black",
            19: "indianred", 10: "violet", 11: "darkorange", 12: "olive", 13: "forestgreen", 14: "darkslategrey", 15: "purple", 16: "magenta", 17: "black", 18: "red",
            20: "sienna", 21: "black", 22: "crimson", 23: "violet", 24: "blue", 25: "teal", 26: "purple", 27: "magenta", 28: "black", 29: "red",
            30: "green", 31: "violet", 32: "red", 33: "green", 34: "blue", 35: "teal", 36: "purple", 37: "magenta", 38: "orange", 39: "darkgreen"}
my_map = folium.Map(location=[26.40, 79.85], zoom_start=10)
#print(df)
'''
count = 0;   
dist = 0;   
for agent in allAgents:
    temp = df[df['AgentId'] == agent]
    subset = temp[['Latitude', 'Longitude']]
    tuples = [tuple(x) for x in subset.to_numpy()]
    for mytuple in tuples:
        folium.Marker(mytuple, icon=folium.Icon(color=colormap[count], icon='car', prefix='fa')).add_to(my_map);
    dist += (getDistance(tuples));
    count = count + 1;

print(dist)

'''
df_manual = pd.read_excel("../Results/zone_3_trajectory_0_KNC_Mannual.xlsx")
subset = df_manual[['Latitude', 'Longitude']]
tuples = [tuple(x) for x in subset.to_numpy()]
for mytuple in tuples:
    folium.Marker(mytuple, icon=folium.Icon(color="black", icon='asterisk', prefix='fa')).add_to(my_map);

poly = folium.PolyLine(locations = tuples, color = "gray", width = 50)
poly.add_to(my_map)


def get_float_list(start, stop, size):
    result = []
    unique_set = set()
    for i in range(size):
        x = round(random.uniform(start, stop),4)
        while x in unique_set:
            x = round(random.uniform(start, stop),4)
        unique_set.add(x)
        result.append(x)

    return result

print("List of unique random numbers between 100, 1000")
print(get_float_list( 26.6883,26.7459, 24))


tuples = [tuple(x) for x in df_crime.to_numpy()]
folium.plugins.HeatMap(tuples).add_to(my_map)
my_map.add_child(plugins.HeatMap(tuples, radius=25))
my_map.add_child(folium.LatLngPopup());

#my_map.save("../Results/traj.html")
