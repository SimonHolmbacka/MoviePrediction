'''
Created on Oct 29, 2017

@author: simon
'''

class MyConfig(object):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        
    def setTargetColumns(self, c):
        self.targetColumns = c
    def getTargetColumns(self):
        return self.targetColumns
    
    def setTrainColumns(self, c):
        self.trainColumns = c
    def getTrainColumns(self):
        return self.trainColumns
    
    def setTrainIndex(self,c):
        self.trainIndex = c
    def getTrainIndex(self):
        return self.trainIndex
    
    def setTestIndex(self,c):
        self.testIndex = c
    def getTestIndex(self):
        return self.testIndex
        
    def setEpochs(self,c):
        self.epochs = c
    def getEpochs(self):
        return self.epochs    
        
    def setMinMax(self, c):
        self.minmax = c
    def getMinMax(self):
        return self.minmax
    
    def setIdSource(self, c):
        self.idsource = c
    def getIdSource(self):
        return self.idsource

    def setIIdSource(self, c):
        self.iidsource = c
    def getIIdSource(self):
        return self.iidsource
    
    #The cursor to the database
    def setDatabase(self, c):
        self.database = c
    def getDatabase(self):
        return self.database
    
    #History, Future
    def setWindows(self, c):
        self.windows = c
    def getWindows(self):
        return self.windows
    