import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
import os
import time
import Config
from sklearn.cluster import AgglomerativeClustering

CLASSLIST = {0: 'BENIGN', 1: 'Infiltration', 2: 'Bot', 3: 'PortScan', 4: 'DDoS', 5: 'FTP-Patator', 6: 'SSH-Patator', 7: 'DoS slowloris', 8: 'DoS Slowhttptest', 9: 'DoS Hulk', 10: 'DoS GoldenEye', 11: 'Heartbleed', 12: 'Web Attack – Brute Force', 13: 'Web Attack – XSS', 14:'Web Attack – Sql Injection'}
LISTCLASS = {CLASSLIST[x]:x for x in range(15)}
PROTOCOLS = {"udp":0,"tcp":1}
CHUNKSIZE = 10000

def classConvert(x):
    return LISTCLASS[x]
def protocalConvert(x):
    return PROTOCOLS[x]

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


#NOTE: There are three diffrent dataset loaders here. I think I could get them all to be the same but that would be difficult
#Class Div Dataset is the original and is quite diffrently stored compaired to the other two.


#note, this is a very modified version of a dataloader found in https://www.youtube.com/watch?v=ZoZHd0Zm3RY
class ClassDivDataset(Dataset):
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
        self.maxclass = Config.parameters["MaxPerClass"][0]
        
        

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
        self.checkIfSplit()

    def __len__(self) -> int:
        if self.listOfCounts is None:
            self.listOfCounts = pd.read_csv(self.path+"counts.csv", index_col=0)
            #This removes all of the unused classes
            self.listOfCounts = self.listOfCounts.loc[self.use]
        if self.length is None:
            self.length = self.listOfCounts.sum().item()
        return self.length


    def __getitem__(self, index) -> tuple([torch.Tensor,torch.Tensor]):
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

    def seriesprocess(self,x:pd.Series) -> tuple([torch.Tensor,torch.Tensor]):
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
    
    def checkIfSplit(self):
        path = self.path
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

















class ClusterDivDataset(ClassDivDataset):
    def __init__(self, path:str, use:list=None, unknownData=False):
        #path is the string path that is the main datafile
        #use is the list of integers corrispoding with the class dictionary above that you want to load.
        #Unknown Data is if the dataset should only give unknown labels.
        
        #If you want to make a dataloader that only reads benign data:
        #  train = Dataload.Dataset("Payload_data_CICIDS2017_Sorted",use=[0])
        #"Payload_data_CICIDS2017_Sorted" is the main name for where the chunked data folder is
        #use = [0] means that we are only using CLASSLIST[0] which is benign

        self.unknownData=unknownData
        self.path = path+"_Clustered"
        self.length = None
        self.listOfCounts = None
        self.perclassgroups = None
        self.clusters = 32
        self.minclass = 0
        


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
        self.checkIfSplit(path)
            
    def __len__(self) -> int:
        if self.listOfCounts is None:
            self.listOfCounts = pd.read_csv(self.path+"/counts.csv", index_col=None)
            #Limit the size of the model
            self.listOfCounts.mask(self.listOfCounts>self.maxclass,self.maxclass,inplace=True)
            #This removes all of the unused classes
            self.listOfCounts = self.listOfCounts.loc[self.use]
            #count how many classes we have the minumim number of examples in.
            self.perclassgroups = (self.listOfCounts>self.minclass).sum(axis=1)
        if self.length is None:
            self.length = self.listOfCounts[self.listOfCounts>self.minclass].sum().sum().item()
            self.length = int(self.length)
        return self.length


    def __getitem__(self, index) -> tuple([torch.Tensor,torch.Tensor]):
        #For debug
        if index==self.length:
            print(index)

        #Now it needs to figure out what type of data it is using.
        chunktype = 0
        chunkNumber = 0
        classNumber = 0
        while index>=self.listOfCounts.iat[chunktype,chunkNumber]:

            if self.listOfCounts.iat[chunktype,chunkNumber]>self.minclass:
                classNumber+=1
                index -= self.listOfCounts.iat[chunktype,chunkNumber]

            #look at next chunk
            chunkNumber+=1
            
            #looked at all of the chunks of this type (32 chunks)
            if chunkNumber>=self.clusters:
                chunktype+=1
                chunkNumber = 0
        

        t_start = time.time()
        #print(f"/chunk{self.usedDict[chunktype]}-type{chunkNumber:03d}.csv")
        #print(f"supposed length:{self.listOfCounts.iat[chunktype,chunkNumber]}, Index:{index}")
        chunk = pd.read_csv(self.path+f"/chunk{self.usedDict[chunktype]}-type{chunkNumber:03d}.csv", index_col=False,chunksize=1,skiprows=index).get_chunk()
        t_total = time.time()-t_start
        if t_total>1:
            print(f"load took {t_total:.2f} seconds")

            
        data, labels = self.seriesprocess(chunk.iloc[0],classNumber)  

        
        #print(f"index: {index} does not exist in chunk: {chunkNumber} of type: {chunktype} ")

        item = data,labels

        return item
    
    def checkIfSplit(self,path):
        if not os.path.exists(self.path): 
            print("Generating clustered data folder.")
            os.mkdir(self.path)

            #Create an dataframe to store how many of each cluster there is
            counts = pd.DataFrame(0, index=range(15),columns=range(self.clusters))
            #Read the data and make sure that the protocols and labels are numbers. Then convert them to integers
            data = pd.read_csv(path+".csv",converters={"protocol":protocalConvert,"label":classConvert})
            for x in range(15):
                X = data.astype(int)
                X = X[X["label"]==x]
                X = X.sample(n=10000 if 10000<len(X) else len(X))
                X2 = X.to_numpy()

                # setting distance_threshold=0 ensures we compute the full tree.
                model = AgglomerativeClustering(distance_threshold=None, n_clusters=self.clusters if self.clusters<len(X) else 1, compute_distances=True)

                model = model.fit(X2)
                lst = model.labels_
                bincount = np.bincount(lst,minlength=self.clusters)
                counts.iloc[x] = bincount
                for i in range(self.clusters):
                    X3 = X[lst==i]
                    X3.to_csv(self.path+f"/chunk{CLASSLIST[x]}-type{i:03d}.csv",index_label=False,index=False)
            counts.to_csv(f"{self.path}/counts.csv",index_label=False,index=False)

    def seriesprocess(self,x:pd.Series,classNumber:int) -> tuple([torch.Tensor,torch.Tensor]):
        #this separates the data from the labels with series

        data = x.iloc[:len(x)-1]
        data = torch.tensor(data.to_numpy())

        label = torch.tensor(int(classNumber),dtype=torch.long)    #The int is because the loss function is expecting ints
        label.unsqueeze_(0)              #This is to allow it to be two dimentional
        if self.unknownData:
            #label2 = torch.tensor(self.perclassgroups.sum().item(),dtype=torch.long).unsqueeze_(0)    #unknowns are marked as unknown
            label2 = torch.tensor(15,dtype=torch.long).unsqueeze_(0)
        else:
            label2 = x.iloc[len(x)-1]         #This selects the label
            label2 = torch.tensor(int(label2),dtype=torch.long)    #The int is because the loss function is expecting ints
            label2.unsqueeze_(0)              #This is to allow it to be two dimentional
        label = torch.cat([label,label2], dim=0)


        return (data,label)


















class ClusterLimitDataset(ClusterDivDataset):
    def __init__(self, path:str, use:list=None, unknownData=False):
        #path is the string path that is the main datafile
        #use is the list of integers corrispoding with the class dictionary above that you want to load.
        #Unknown Data is if the dataset should only give unknown labels.
        
        #If you want to make a dataloader that only reads benign data:
        #  train = Dataload.Dataset("Payload_data_CICIDS2017_Sorted",use=[0])
        #"Payload_data_CICIDS2017_Sorted" is the main name for where the chunked data folder is
        #use = [0] means that we are only using CLASSLIST[0] which is benign

        self.unknownData=unknownData
        self.path = path+"_Clustered"
        self.length = None
        self.listOfCounts = None
        self.perclassgroups = None
        self.clusters = 32
        self.minclass = 0
        self.maxclass = Config.parameters["MaxPerClass"][0]
        


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
        self.checkIfSplit(path)
            
    def __len__(self) -> int:
        if self.listOfCounts is None:
            self.listOfCounts = pd.read_csv(self.path+"/counts.csv", index_col=None)
            #This is the limit part of the class name
            self.listOfCounts.mask(self.listOfCounts>self.maxclass,self.maxclass,inplace=True)
            #This removes all of the unused classes
            self.listOfCounts = self.listOfCounts.loc[self.use]
            #count how many classes we have the minumim number of examples in.
            self.perclassgroups = (self.listOfCounts>self.minclass).sum(axis=1)
        if self.length is None:
            self.length = self.listOfCounts[self.listOfCounts>self.minclass].sum().sum().item()
            self.length = int(self.length)
        return self.length


    def __getitem__(self, index) -> tuple([torch.Tensor,torch.Tensor]):
        #For debug
        if index==self.length:
            print(index)

        #Now it needs to figure out what type of data it is using.
        chunktype = 0
        chunkNumber = 0
        classNumber = 0
        while index>=self.listOfCounts.iat[chunktype,chunkNumber]:

            if self.listOfCounts.iat[chunktype,chunkNumber]>self.minclass:
                classNumber+=1
                index -= self.listOfCounts.iat[chunktype,chunkNumber]

            #look at next chunk
            chunkNumber+=1
            
            #looked at all of the chunks of this type (32 chunks)
            if chunkNumber>=self.clusters:
                chunktype+=1
                chunkNumber = 0
        

        t_start = time.time()
        #print(f"/chunk{self.usedDict[chunktype]}-type{chunkNumber:03d}.csv")
        #print(f"supposed length:{self.listOfCounts.iat[chunktype,chunkNumber]}, Index:{index}")
        chunk = pd.read_csv(self.path+f"/chunk{self.usedDict[chunktype]}-type{chunkNumber:03d}.csv", index_col=False,chunksize=1,skiprows=index).get_chunk()
        t_total = time.time()-t_start
        if t_total>1:
            print(f"load took {t_total:.2f} seconds")

            
        data, labels = self.seriesprocess(chunk.iloc[0])  

        
        #print(f"index: {index} does not exist in chunk: {chunkNumber} of type: {chunktype} ")

        item = data,labels

        return item
    


    def seriesprocess(self,x:pd.Series) -> tuple([torch.Tensor,torch.Tensor]):
        #this separates the data from the labels with series

        data = x.iloc[:len(x)-1]
        data = torch.tensor(data.to_numpy())

        label = x.iloc[len(x)-1]         #This selects the label
        label = torch.tensor(int(label),dtype=torch.long)    #The int is because the loss function is expecting ints
        label.unsqueeze_(0)              #This is to allow it to be two dimentional
        if self.unknownData:
            #label2 = torch.tensor(self.perclassgroups.sum().item(),dtype=torch.long).unsqueeze_(0)    #unknowns are marked as unknown
            label2 = torch.tensor(15,dtype=torch.long).unsqueeze_(0)
        else:
            label2 = label.clone()
        label = torch.cat([label2,label], dim=0)


        return (data,label)


        
        

