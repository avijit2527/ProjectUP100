import pandas as pd
import folium
import numpy as np

df = pd.read_excel("../Results/trajectory.xlsx")
print(df.head())
allAgents = df.AgentId.unique()
colormap = {0:"green", 1:"black",2: "red",3: "green",4: "blue",5: "cyan",6: "yellow",7: "magenta", 8: "black",9: "red"}
my_map = folium.Map(location=[26.5, 79.0],zoom_start = 7)
print(allAgents)
for agent in allAgents:
    temp = df[df['AgentId'] == agent]
    subset = temp[['Latitude', 'Longitude']]
    tuples = [tuple(x) for x in subset.to_numpy()]
    folium.PolyLine(tuples, color = colormap[agent], weight=2.5, opacity=1).add_to(my_map)
    
my_map.save("../Results/traj.html")
