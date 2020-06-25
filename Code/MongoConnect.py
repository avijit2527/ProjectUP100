#This reads from xlsx and updates mongodb

from pymongo import MongoClient
import json
from pprint import pprint
import pandas as pd
from bson.objectid import ObjectId
import datetime


client = MongoClient(port=27017)
db=client.conFusion
# Issue the serverStatus command and print the results
allDocs = db.vehicles.find({})
#for document in allDocs: 
#    pprint(document)



df = pd.read_excel("../Results/trajectory_Ayodhya.xlsx")

print(df["TimeSlot"][0])
x = datetime.datetime.now()
for step in df.itertuples():
    print(step[2])

    db.vehicles.update_one({"vehicleId" : str(step[2])},{"$push": {"locations": {"_id":ObjectId(),"createdAt":x,"updatedAt":x,"timeSlot":step[3],"lat":step[4],"lng":step[5]}}})