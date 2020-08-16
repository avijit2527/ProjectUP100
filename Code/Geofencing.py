#!/usr/bin/env python
# coding: utf-8
# @author Avijit Roy

import shapefile;
import folium;

sf = shapefile.Reader("../Dataset/Geofencing/PRJ/PRAYAGRAJ_PS"); #PRAYAGRAJ_DISTRICT
s = sf.shape(0);
print(sf.shapeRecords()[1]);  #[0].record[25:27]

'''records = sf.shapeRecords();

points = [];
for x in records:
    points.append((x.shape.points[1][1],x.shape.points[1][0]));


my_map = folium.Map(location=[25.41, 81.82], zoom_start=10);

for x in points:
    folium.Marker(x).add_to(my_map);
#folium.Polygon(points, color="red", weight=2.5, opacity=1).add_to(my_map)
my_map.save("../Results/Geofencing.html")'''
