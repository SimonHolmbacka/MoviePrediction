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
from keras.layers import Input
from keras.models import model_from_json
from keras.utils import np_utils
import matplotlib.pyplot as plt
import time
from random import randint
import matplotlib

import ModelConfig
import ConfigParser
from seq2seq.models import SimpleSeq2Seq


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
    return config,Config




def getnewdata(cellid,config):    
    sql = "select movieId from main WHERE userId == '%s'" % (cellid)
    temp = config.getDatabase().execute(sql)
    temp = temp.fetchall()
    dataset = np.array(temp)
    del temp
    #Check that we have enough data to make a proper window
    if (dataset.shape[0]-2 < config.getWindows()[0]+config.getWindows()[0]):
        print "Not enough data in timeseries!"
        return 0,0,0
    else:
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
        X = np.array(X,dtype=float)
        Y = np.array(Y,dtype=float)
        return X,Y,dataset
        
def updatedata(dataset):    
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
    X = np.array(X,dtype=float)
    Y = np.array(Y,dtype=float)
    return X,Y,dataset
    
config,Config = MakeConfig()

# out = None
# for i in range(0,iids.shape[0]):
#     X,Y,ds = getnewdata(iids[i],config)
#     if(np.isscalar(X) == False):
#         print i
#         out = np.append(out,iids[i])
# iids = out
# iids = iids[1:iids.shape[0]]
# f_handle = file("/home/simon/Documents/LiClipse Workspace/MoviePrediction/iids.csv", 'w')
# np.savetxt(f_handle, iids,  delimiter=",",fmt="%s")
# f_handle.close()    

iids = np.genfromtxt('/home/simon/Documents/LiClipse Workspace/MoviePrediction/iids.csv', delimiter=',',dtype=int) 


X,Y,ds = getnewdata(iids[2],config)
Xt,Yt,ds = getnewdata(iids[randint(config.getTestIndex()[0],config.getTestIndex()[1])],config)

model = Sequential()
s2s = SimpleSeq2Seq(batch_input_shape=(1, X.shape[1], X.shape[2]), hidden_dim=4, output_length=config.getWindows()[1], output_dim=1)
model.add(s2s) 
model.add(Dense(50,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2999,activation='softmax'))
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


X,Y,ds = getnewdata(iids[1000],config)
out = model.predict(X, batch_size=1) 
moviearray = np.array(out[0,0,:])

for i in range(0,10):
    bestindex = np.argmax(moviearray)
    bestvalue = moviearray[bestindex]
    moviearray = np.delete(moviearray, range(bestindex,bestindex+1)) 
    
    imdbbase = sqlite3.connect('/home/simon/Documents/LiClipse Workspace/MoviePrediction/Links.sqlite')
    imdbbase.text_factory = str
    cursor = imdbbase.cursor()
    sql = "select distinct imdbId from main WHERE movieid == '%s'" % (bestindex)
    temp = cursor.execute(sql)
    temp = temp.fetchall()
    imdbid = np.array(temp)[0][0]
    
    url ="http://www.imdb.com/title/tt"+str(imdbid)+"/"
    webbrowser.open_new_tab(url)
    
    ds = ds[1:ds.shape[0]]
    ds = np.append(ds,bestindex)  
    ds = ds.reshape(174,1)
    X,Y,ds = updatedata(ds)    
    out = model.predict(X, batch_size=1) 
    moviearray = np.array(out[0,0,:])
    time.sleep(5)


