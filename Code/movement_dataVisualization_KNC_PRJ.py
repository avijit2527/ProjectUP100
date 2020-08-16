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

from pymongo import MongoClient
import json
from pprint import pprint
from bson.objectid import ObjectId
import datetime


from Utilities import Utilities as ut



df1 = pd.read_excel("../Dataset/KNC_PRJ.xlsx");
df1['Createtime'] = pd.to_datetime(df1['Createtime'])
df = df1[["EventId","Createtime","Type","Region","latitude","longitude"]];
df = df.dropna();
print(df.dtypes);
print(df["Type"].unique());
print(df["Region"].unique());


client = MongoClient(port=27017);
db=client.conFusion;

x = datetime.datetime.now();


for step in df.itertuples():
    print(step[2])
    db.crimes.insert_one({"eventId":step[1],"crimeTime":step[2],"type":step[3],"zone":step[4],"lat":step[5],"lng":step[6],"createdAt":x,"updatedAt":x});