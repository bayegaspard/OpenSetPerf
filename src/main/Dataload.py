import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
import os
import time

CLASSLIST = {0: 'BENIGN', 1: 'Infiltration', 2: 'Bot', 3: 'PortScan', 4: 'DDoS', 5: 'FTP-Patator', 6: 'SSH-Patator', 7: 'DoS slowloris', 8: 'DoS Slowhttptest', 9: 'DoS Hulk', 10: 'DoS GoldenEye', 11: 'Heartbleed', 12: 'Web Attack – Brute Force', 13: 'Web Attack – XSS', 14:'Web Attack – Sql Injection'}
LISTCLASS = {CLASSLIST[x]:x for x in range(15)}
PROTOCOLS = {"udp":0,"tcp":1}
CHUNKSIZE = 10000

def get_class_names(lst):
    new_class_list = []
    for i in lst:
        new_class_list.append(CLASSLIST[i])
    return new_class_list

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

device = get_default_device()

def chunkprocess(x:pd.DataFrame):
    #This separates the data from the labels

    data = x
    data = data.drop("label",inplace=False, axis=1)         #This removes the labels
    data = data.drop("Unnamed: 0", inplace=False, axis=1)      #I do not know what unnamed is it was not showing up earlier
    data = torch.tensor(data.to_numpy())

    label = x
    label = label["label"]                          #This selects the label
    label = torch.tensor(label.to_numpy())    #The int is because the loss function is expecting ints

    return (data,label)





#note, this is a very modified version of a dataloader found in https://www.youtube.com/watch?v=ZoZHd0Zm3RY
class Dataset(Dataset):
    def __init__(self, path:str, use:list=None, unknownData=False):
        #path is the string path that is the main datafile
        #use is the list of integers corrispoding with the class dictionary above that you want to load.
        #Unknown Data is if the dataset should only give unknown labels.
        
        #If you want to make a dataloader that only reads benign data:
        #  train = Dataload.Dataset("Payload_data_CICIDS2017_Sorted",use=[0])
        #"Payload_data_CICIDS2017_Sorted" is the main name for where the chunked data folder is
        #use = [0] means that we are only using CLASSLIST[0] which is benign

        self.unknownData=unknownData
        self.path = path
        self.length = None
        self.listOfCounts = None


        #This is setting what classes are considered to be knowns.
        if use is not None:
            self.use = [False for i in range(len(CLASSLIST))] 
            self.usedDict = {}
            use.sort()
            for case in use:
                self.use[case] = True
                #OK this requires you to have the use list be sorted, but otherwise it works.
                self.usedDict[len(self.usedDict)] = CLASSLIST[case]
        else:
            self.use = [True for i in range(len(CLASSLIST))] 
            self.usedDict = CLASSLIST
        
        #this will check if the file is chunked and chunk it if it is not
        checkIfSplit(path)

    def __len__(self) -> int:
        if self.listOfCounts is None:
            self.listOfCounts = pd.read_csv(self.path+"counts.csv", index_col=0)
            #This removes all of the unused classes
            self.listOfCounts = self.listOfCounts.loc[self.use]
        if self.length is None:
            self.length = self.listOfCounts.sum().item()
        return self.length


    def __getitem__(self, index) -> tuple([torch.tensor,torch.tensor]):
        #For debug
        if index==self.length:
            print(index)

        #Now it needs to figure out what type of data it is using.
        chunktype = 0
        while index>=self.listOfCounts.iat[chunktype,0]:
            index -= self.listOfCounts.iat[chunktype,0]
            chunktype+=1
        chunkNumber = index//CHUNKSIZE
        index = index%CHUNKSIZE
        #This is needed incase it is not a full chunk so the max value is 0
        index = index%self.listOfCounts.iat[chunktype,0]

        t_start = time.time()
        chunk = pd.read_csv(self.path+f"/chunk{self.usedDict[chunktype]}{chunkNumber}.csv", index_col=False,chunksize=1,skiprows=index).get_chunk()
        t_total = time.time()-t_start
        if t_total>1:
            print(f"load took {t_total:.2f} seconds")

        data, labels = self.seriesprocess(chunk.iloc[0])  
        
        #print(f"index: {index} does not exist in chunk: {chunkNumber} of type: {chunktype} ")

        item = data,labels

        return item

    def seriesprocess(self,x:pd.Series) -> tuple([torch.tensor,torch.tensor]):
        #this separates the data from the labels with series

        data = x.iloc[:len(x)-1]
        data = torch.tensor(data.to_numpy())

        label = x.iloc[len(x)-1]         #This selects the label
        label = torch.tensor(int(label),dtype=torch.long)    #The int is because the loss function is expecting ints
        label.unsqueeze_(0)              #This is to allow it to be two dimentional
        if self.unknownData:
            label2 = torch.tensor(15,dtype=torch.long).unsqueeze_(0)    #unknowns are marked as unknown
        else:
            label2 = label.clone()
        label = torch.cat([label2,label], dim=0)


        return (data,label)



def checkIfSplit(path):
    def classConvert(x):
        return LISTCLASS[x]
    def protocalConvert(x):
        return PROTOCOLS[x]
    if not os.path.exists(os.path.join(path,"")): 
        os.mkdir(path)


        #this stores the data in dataframes
        runningDataFrames = []
        for c in CLASSLIST:
            runningDataFrames.append(pd.DataFrame())

        #this is just to keep track of how many files exist of each class
            filecount = [0]*len(runningDataFrames)

        data = pd.read_csv(path+".csv", chunksize=CHUNKSIZE,converters={"protocol":protocalConvert,"label":classConvert})
        for chunk in data:
            #this is the new version that splits the classes

            for j in range(len(runningDataFrames)):
                mask = chunk["label"]==j         #this deturmines if things are in this class
                runningDataFrames[j] = pd.concat((runningDataFrames[j],chunk[mask]))
                if len(runningDataFrames[j])>=10000:
                    runningDataFrames[j][:10000].to_csv(path+f"/chunk{CLASSLIST[j]}{filecount[j]}.csv",index_label=False,index=False)
                    runningDataFrames[j] = runningDataFrames[j][10000:]
                    filecount[j] += 1

        count = [x*10000 for x in filecount]
        for j in range(len(runningDataFrames)):
            runningDataFrames[j].to_csv(path+f"/chunk{CLASSLIST[j]}{filecount[j]}.csv",index_label=False,index=False)
            count[j] += len(runningDataFrames[j])
        
        count = pd.DataFrame(count)
        count.to_csv(path+"counts.csv",index_label=False)
        
        