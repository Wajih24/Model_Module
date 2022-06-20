import os
import time
import json
import joblib
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
from sklearn.cluster import KMeans
from lightgbm import LGBMClassifier
from flask import Flask, request, jsonify
import pandas as pd


model_directory = 'model'
model_file_name = '%s/model_updated.pickle' % model_directory
model_columns_file_name = '%s/model_columns_updated.pickle' % model_directory
model_columns = joblib.load(model_columns_file_name)
clf = joblib.load(model_file_name)
model_labels = {'Playertype':{'type_achiever': 0,  'type_disruptor': 1,  'type_freeSpirit': 2,   'type_player': 3}}


f = open("input.json","r", encoding='utf8')

input = json.loads(f.read())
keys = list(input.keys())

# Prediction on all the User_IDs
for j in keys:
   sample = {}
   for i in input[j] :
      if i in model_columns:
         try: 
            sample[i] = int(input[j][i])
         except :
            sample[i] = 0
         

   query = pd.DataFrame(sample,index=[0])
   prediction = clf.predict_proba(query)[0]
   lab = dict((v,k) for k,v in model_labels['Playertype'].items())
   
   prediction = list(prediction)

   input[j]["Player_Type"] = lab[prediction.index(max(prediction))] # Add the predcition to the user row
   input[j]['type_achiever'] = round(prediction[0]*100,2)
   input[j]['type_disruptor'] = round(prediction[1]*100,2)
   input[j]['type_freeSpirit'] = round(prediction[2]*100,2)
   input[j]['type_player'] = round(prediction[3]*100,2)


with open("output.json", "w",encoding='utf8') as write_file:
    json.dump(input, write_file, indent=4,ensure_ascii = False)
