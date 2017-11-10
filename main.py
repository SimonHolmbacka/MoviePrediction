import sqlite3
import numpy as np
import pandas as pd
from pandas.tests.io.parser import usecols, skiprows
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras import optimizers
from keras.layers.core import Activation, Dense
# from keras.layers import LSTM
from keras.layers import Input
from keras.models import model_from_json
from keras.utils import np_utils
import matplotlib.pyplot as plt
import time
from random import randint
import matplotlib
# from seq2seq.models import SimpleSeq2Seq
#Local imports
# import ElisaMakeMinMax
import ModelConfig
import ConfigParser
from seq2seq.models import SimpleSeq2Seq
# import ExtendData
# from ExtendData import Extend_more_alarms

# git clone https://www.github.com/datalogai/recurrentshop.git
# cd recurrentshop
# python setup.py install
#sudo pip install git+https://github.com/farizrahman4u/seq2seq.git
#CREATE INDEX tag_cellid ON main (CELLID);

def GetDatabase(path):
    elisabase = sqlite3.connect(path)
    elisabase.text_factory = str
    cursor = elisabase.cursor()
    return cursor

# def GetIDs(path): 
#     _id = np.genfromtxt(path, delimiter=',',dtype=str) 
#     return _id
# 
# def GetInterestingIDs(path): 
#     _id = np.genfromtxt(path, delimiter=',',dtype=str) 
#     return _id
# 
# def GetMinMax(path):
#     _minmax = np.genfromtxt(path, delimiter=',',dtype=float)
#     return _minmax

#Create new MaxMin values for nomalization
#They are saved in csv file
# it will save as /home/simon/Documents/LiClipse Workspace/MinMax<projname>"
# def CreateMinMax(db):
#     elisabase = sqlite3.connect(db)
#     elisabase.text_factory = str
#     cursor = elisabase.cursor()  
#     ElisaMakeMinMax.MainFunction(cursor)

def MakeConfig():
    Config = ConfigParser.ConfigParser()
    Config.read('/home/simon/Documents/LiClipse Workspace/MoviePrediction/Conf.txt')
    config = ModelConfig.MyConfig()
    config.setTrainColumns([Config.getint("Columns",'TrainColumnBegin'),Config.getint("Columns",'TrainColumnEnd')])
    config.setTargetColumns([Config.getint("Columns",'TargetColumnBegin'),Config.getint("Columns",'TargetColumnEnd')])
    config.setDatabase(GetDatabase(Config.get("Source",'Database')))
    config.setWindows([Config.getint("Settings",'HistoryWindow'),Config.getint("Settings",'FutureWindow')])
    config.setTrainIndex([Config.getint("Settings",'TrainIndexBegin'),Config.getint("Settings",'TrainIndexEnd')])
    config.setTestIndex([Config.getint("Settings",'TestIndexBegin'),Config.getint("Settings",'TestIndexEnd')])
    config.setEpochs(Config.getint("Settings",'Epochs'))
    # CreateMinMax(Config.get("Source",'Database'))
#     config.setMinMax(GetMinMax(Config.get("Source",'MinMax')))
#     config.setIdSource(GetIDs(Config.get("Source",'IdSource')))
#     config.setIIdSource(GetInterestingIDs(Config.get("Source",'InterestingIdSource')))    
    return config,Config




def getnewdata(cellid,config):    
sql = "select movieId from main WHERE userId == '%s'" % (cellid)
temp = config.getDatabase().execute(sql)
temp = temp.fetchall()
dataset = np.array(temp)
#     dataset = dataset[0:20]
    del temp
    #Check that we have enough data to make a proper window
    if (dataset.shape[0]-2 < config.getWindows()[0]+config.getWindows()[0]):
        print "Not enough data in timeseries!"
        return 0,0,0
    else:
        #Normalize the part that are numbers
        #Delete 2 first columns: ID,Time
        #ID is given as input and is the same for all items.
        #Time is assumed to be ordered in the database, and is therefore in a sequence here
        #Current implementation has one hour sampling rate
#         dataset = np.delete(dataset, range(0,1), axis=1) 
#         dataset[dataset=='']='0' #Convert all blanks to zeros
        
        #TODO HOW TO SPLIT FLOATS AND TEXT IN A GOOD WAY
#         numeric_part = np.array(dataset[:,0:392],dtype=np.float64)
#         text_part = dataset[:,392:396]
        
        #Normalize the data with minmax scaling
#         MinMax = config.getMinMax()
#         numeric_part = (numeric_part - MinMax[1]) / (MinMax[0] - MinMax[1])
#         numeric_part = np.nan_to_num(numeric_part)      
#         dataset = None
#         dataset = np.append(numeric_part, text_part,axis=1)
        
        #TODO HOW TO CONVERT TEXT TO NUMBER (IF NEEDED)
#         TextIndex = 392
#         dataset[:,TextIndex][dataset[:,TextIndex] != 'CELL OPERATION DEGRADED'] = 0
#         dataset[:,TextIndex][dataset[:,TextIndex] == 'CELL OPERATION DEGRADED'] = 1
        
        #Set training and test columns
        InBEGIN = config.getTrainColumns()[0]
        InEND = config.getTrainColumns()[1]
        OutBEGIN = config.getTargetColumns()[0]         
        OutEND = config.getTargetColumns()[1]
        history_window = config.getWindows()[0]
        future_window = config.getWindows()[1]        
            
        X_train = dataset[:,InBEGIN:InEND]      #Input data
        Y_train = dataset[:,OutBEGIN:OutEND]    #Output data
            
        #Transform NxM matrix into sliding window structure 
        X = None
        nb_samples = len(X_train) - history_window - future_window
        input_list = [np.expand_dims(np.atleast_2d(X_train[i:history_window+i,:]), axis=0) for i in range(0, nb_samples)] #nb_samples=length of total input
        X = np.concatenate(input_list, axis=0)
        Y = None
        nb_samples = len(Y_train) - history_window - future_window 
        input_list = [np.expand_dims(np.atleast_2d(Y_train[history_window+i:history_window+i+future_window,:]), axis=0) for i in range(0, nb_samples)] 
        Y = np.concatenate(input_list, axis=0)
        
        #Make sure the data is a number
        #TODO SHOULD THIS BE IMPLICITLY FLOAT OR SHOULD THE USER SELECT?
        X = np.array(X,dtype=int)
        Y = np.array(Y,dtype=int)
        return X,Y,dataset

config,Config = MakeConfig()
sql = "select distinct userid from main order by userid asc"
temp = config.getDatabase().execute(sql)
temp = temp.fetchall()
dataset = np.array(temp)
f_handle = file("/home/simon/Documents/LiClipse Workspace/MovieUsers.csv", 'w')
np.savetxt(f_handle, dataset,  delimiter=",",fmt="%s")
f_handle.close()
ids = dataset
ids = ids[:,0]
# id = GetIDs(config.getIdSource())
# iid = GetInterestingIDs(config.getIIdSource())

#GET A SHAPE FOR X AND Xt, THIS SHOULD BE DONE IN A BETTER WAY

# model.fit(X, Y,initial_epoch=0, epochs=1,batch_size=1,shuffle=True, validation_data=(X,Y))
# model.fit(X, Y,initial_epoch=rounds, epochs=rounds+1,batch_size=1,shuffle=True, validation_data=(Xt,Yt))
# X,Y,ds = getnewdata(1,config)
# temp = X.reshape(1,X.shape[1],X.shape[2])
# out = model.predict(X, batch_size=1) 

# out = None
# for i in range(0,46292):
#     X,Y,ds = getnewdata(iids[i],config)
#     if(np.isscalar(X) == False):
#         print i
#         out = np.append(out,iids[i])
# 
# iids = out
# iids = iids[1:iids.shape[0]]
# f_handle = file("iids.csv", 'w')
# np.savetxt(f_handle, iids,  delimiter=",",fmt="%s")
# f_handle.close()    

iids = np.genfromtxt('/home/simon/Documents/LiClipse Workspace/MoviePrediction/iids.csv', delimiter=',',dtype=int) 


X,Y,ds = getnewdata(iids[2],config)
Xt,Yt,ds = getnewdata(iids[randint(config.getTestIndex()[0],config.getTestIndex()[1])],config)

model = Sequential()
s2s = SimpleSeq2Seq(batch_input_shape=(1, X.shape[1], X.shape[2]), hidden_dim=1, output_length=config.getWindows()[1], output_dim=1)
model.add(s2s) 
model.add(Dense(50,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(99999,activation='softmax'))
opt = optimizers.Nadam()
model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

for rounds in range(0,config.getEpochs()):
    for index in range(config.getTrainIndex()[0],config.getTrainIndex()[1]):
        print "Index ",index," Round ", rounds
        X,Y,ds = getnewdata(iids[index],config)
        r = randint(config.getTestIndex()[0],config.getTestIndex()[1])
        print "random", r
        Xt,Yt,ds = getnewdata(iids[r],config)
        model.fit(X, Y,initial_epoch=rounds, epochs=rounds+1,batch_size=1,shuffle=True, validation_data=(Xt,Yt))
print "ok"


X,Y,ds = getnewdata(iids[0],config)
out = model.predict(X, batch_size=1) 
#             if(np.sum(Y[:,:,0]) > 0.0):
#                 pred = np.zeros((Y.shape[0],config.getWindows()[1])) #First is current time last is +t time steps
#                 for i in range(0,X.shape[0]):
#                     temp = X[i].reshape(1,X.shape[1],X.shape[2])    
#                     out = model.predict(temp, batch_size=1)   
#                     pred[i] = out[0,:,0]             
#                 plt.ion()
#                 plt.clf()        
#                 plt.plot(pred[:,0],color='blue')      
#                 plt.plot(Y[:,0,0],color='orange')
#                 plt.axis([0,Y.shape[0],0,1.1])
#                 plt.pause(0.05)
            
            
            
#             model_json = model.to_json()
#             with open(Config.get("Target",'modelfile')+".json", "w") as json_file:
#                 json_file.write(model_json)
#             model.save_weights(Config.get("Target",'modelfile')+".h5")



#   
