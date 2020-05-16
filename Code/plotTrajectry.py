import pandas as pd
import folium
import numpy as np

df = pd.read_excel("../Results/trajectory_Ayodhya.xlsx")
df_crime = pd.read_excel(
    "../Dataset/Ayodhya_2019_20_Street_crime_events.xlsx")[['Lat', 'Long']]
print(df_crime)



allAgents = df.AgentId.unique()
colormap = {0: "green", 1: "black", 2: "violet", 3: "green", 4: "blue", 5: "teal", 6: "purple", 7: "magenta", 8: "black", 9: "red",
            19: "violet", 10: "violet", 11: "red", 12: "green", 13: "blue", 14: "teal", 15: "purple", 16: "magenta", 17: "black", 18: "red",
            20: "green", 21: "black", 22: "red", 23: "violet", 24: "blue", 25: "teal", 26: "purple", 27: "magenta", 28: "black", 29: "red",
            30: "green", 31: "violet", 32: "red", 33: "green", 34: "blue", 35: "teal", 36: "purple", 37: "magenta", 38: "black", 39: "red"}
my_map = folium.Map(location=[26.7, 82], zoom_start=10)
print(allAgents)
for agent in allAgents:
    temp = df[df['AgentId'] == agent]
    subset = temp[['Latitude', 'Longitude']]
    tuples = [tuple(x) for x in subset.to_numpy()]
    folium.PolyLine(
        tuples, color=colormap[agent], weight=2.5, opacity=1).add_to(my_map)


tuples = [tuple(x) for x in df_crime.to_numpy()]
for mytuple in tuples:
    folium.CircleMarker(location=[mytuple[0], mytuple[1]],
                  fill_color='#000000', radius=1).add_to(my_map)

my_map.save("../Results/traj.html")
