import pandas as pd
import torch
from torch.utils.data import Dataset, IterableDataset,TensorDataset, DataLoader
import numpy as np
import os
import time
import Config
from sklearn.cluster import AgglomerativeClustering
from itertools import filterfalse
from tqdm import tqdm
import copy
import random

#List of conversions:
if Config.parameters["Dataset"][0] == "Payload_data_CICIDS2017":
    CLASSLIST = {'BENIGN': 0, 'Infiltration': 1, 'Bot': 2, 'PortScan': 3, 'DDoS': 4, 'FTP-Patator': 5, 'SSH-Patator': 6, 'DoS slowloris': 7, 'DoS Slowhttptest': 8, 'DoS Hulk': 9, 'DoS GoldenEye': 10, 'Heartbleed': 11, 'Web Attack – Brute Force': 12, 'Web Attack – XSS': 13, 'Web Attack – Sql Injection': 14}
elif Config.parameters["Dataset"][0] == "Payload_data_UNSW":
    CLASSLIST = {"normal": 0,"backdoor": 1, "dos": 2, "exploits": 3, "fuzzers": 4, "generic": 5, "analysis": 6, "reconnaissance": 7, "shellcode": 8, "worms": 9}
else:
    print("ERROR, Dataset not implemented")
#PROTOCOLS = {"udp":0,"tcp":1}
PROTOCOLS = {"udp":0,"tcp":1,"others":2,"ospf":3,"sctp":4,"gre":5,"swipe":6,"mobile":7,"sun-nd":8,"sep":9,"unas":10,"pim":11,"secure-vmtp":12,"pipe":13,"etherip":14,"ib":15,"ax.25":16,"ipip":17,"sps":18,"iplt":19,"hmp":20,"ggp":21,"ipv6":22,"rdp":23,"rsvp":24,"sccopmce":25,"egp":26,"vmtp":27,"snp":28,"crtp":29,"emcon":30,"nvp":31,"fire":32,"crudp":33,"gmtp":34,"dgp":35,"micp":36,"leaf-2":37,"arp":38,"fc":39,"icmp":40,"other":41}
LISTCLASS = {CLASSLIST[x]:x for x in CLASSLIST.keys() if x not in Config.UnusedClasses}
CHUNKSIZE = 10000

def groupDoS(x):
    if False and Config.parameters["Dataset"][0] == "Payload_data_CICIDS2017":
        x[x>=7 and x<=10] = 7
        x[x>10] = x[x>10] - 3
    return x

def classConvert(x):
    """
    Does a conversion based on the dictionaries
    """
    return CLASSLIST[x]

def protocalConvert(x):
    """
    Does a conversion based on the dictionaries
    """
    return PROTOCOLS[x]

def get_class_names(lst):
    """
    Goes through a list of integer classes and turns them back into string class names.
    """
    new_class_list = []
    for i in lst:
        new_class_list.append(LISTCLASS[i])
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



#Try to store all of the data in memory instead?
def recreateDL(dl:torch.utils.data.DataLoader,shuffle=True):
    xList= []
    yList= []
    for xs,ys in tqdm(dl,desc="Loading Dataloader into memory"):
        #https://github.com/pytorch/pytorch/issues/11201#issuecomment-486232056
        xList.append(copy.deepcopy(xs))
        del(xs)
        yList.append(copy.deepcopy(ys))
        del(ys)
    xList = torch.cat(xList)
    yList = torch.cat(yList)
    combinedList = TensorDataset(xList,yList)
    if Config.parameters["num_workers"][0] > 0:
        persistant_workers = True
    else:
        persistant_workers = False
    return DataLoader(combinedList, Config.parameters["batch_size"][0], shuffle=shuffle, num_workers=Config.parameters["num_workers"][0],pin_memory=True, persistent_workers=persistant_workers)


device = get_default_device()


#NOTE: There are three diffrent dataset loaders here. I think I could get them all to be the same but that would be difficult
#Class Div Dataset is the original and is quite diffrently stored compaired to the other two.


#note, this is a very modified version of a dataloader found in https://www.youtube.com/watch?v=ZoZHd0Zm3RY
class ClassDivDataset(Dataset):
    def __init__(self, path:str, use:list=None, unknownData=False):
        """
        initializes the dataloader.
        
        parameters: 
            path is the string path that is the main datafile
            use is the list of integers corrispoding with the class dictionary above that you want to load.
            Unknown Data is if the dataset should only give unknown labels.
        
        If you want to make a dataloader that only reads benign data:
          train = Dataload.Dataset("Payload_data_CICIDS2017_Sorted",use=[0])
        "Payload_data_CICIDS2017_Sorted" is the main name for where the chunked data folder is
        use = [0] means that we are only using LISTCLASS[0] which is benign
        """

        self.unknownData=unknownData
        self.path = path
        self.countspath = path+"counts.csv"
        self.length = None
        self.listOfCounts = None
        self.maxclass = Config.parameters["MaxPerClass"][0]
        self.data_length = 1504
        if "MaxSamples" in Config.parameters:
            self.totalSamples = Config.parameters["MaxSamples"][0]
        
        

        #This is setting what classes are considered to be knowns.
        if use is not None:
            # self.use is a booliean array of a length for all of the classes in the file (not just the classes being used)
            self.use = [False for i in range(Config.parameters["CLASSES"][0])] 
            self.usedDict = {}
            use.sort()
            for case in use:
                if case not in Config.UnusedClasses:
                    self.use[case] = True
                    #OK this requires you to have the use list be sorted, but otherwise it works.
                    self.usedDict[len(self.usedDict)] = LISTCLASS[case]
        else:
            # self.use is a booliean array of a length for all of the classes in the file (not just the classes being used)
            self.use = [(i not in Config.UnusedClasses) for i in range(Config.parameters["CLASSES"][0])] 
            # self.usedDict is a continuous number mapping to file names
            self.usedDict = {keypair[0]: LISTCLASS[keypair[1]] for keypair in enumerate(LISTCLASS.keys())}
        
        #this will check if the file is chunked and chunk it if it is not
        self.checkIfSplit(path)


    def __len__(self) -> int:
        """
        Finds and saves the length of the dataloader.

        returns an intger of the length of the data.
        
        """
        if self.listOfCounts is None:
            self.listOfCounts = pd.read_csv(self.countspath, index_col=0)
            self.totalListOfCounts = self.listOfCounts.copy()
            self.listOfCountsOffset = self.listOfCounts.copy()
            if "MaxSamples" in Config.parameters:
                #MAX SAMPLES DEFINITION:
                #Max Samples limits the total number of samples to self.totalSamples
                #It then distributes those samples to match with the percentages given in percentages.csv as best as possible.
                maxperclass = pd.read_csv("datasets/percentages.csv", index_col=None)
                maxperclass = maxperclass.iloc[Config.parameters["loopLevel"][0],:len(self.listOfCounts)]
                maxperclass = ((torch.tensor(maxperclass)/100)*self.totalSamples).ceil()
                for x in range(len(self.listOfCounts)):
                    if self.listOfCounts.iloc[x,0] > maxperclass[x].item():
                        self.listOfCounts.iloc[x,0] = maxperclass[x].item()
                        # self.listOfCountsOffset.iloc[x,0] = self.listOfCountsOffset.iloc[x,0]-maxperclass[x].item()
                        # if self.listOfCountsOffset.iloc[x,0]>0:
                        #     self.listOfCountsOffset.iloc[x,0] = random.randrange(self.listOfCountsOffset.iloc[x,0].item())
                        # else:
                        #     self.listOfCountsOffset.iloc[x,0] = 0
                #This removes all of the unused classes
                self.listOfCounts = self.listOfCounts.loc[self.use]
                print(f"Items per class: \n{self.listOfCounts}")
            else:
                #add max per class
                if isinstance(self.maxclass,int):
                    #Huh, this only runs if Config MaxPerClass is an integer. 
                    # But it takes x samples of each class where x is MaxPerClass
                    self.listOfCounts.mask(self.listOfCounts>self.maxclass,self.maxclass,inplace=True)
                elif self.maxclass == "File":
                    #Not quite sure how this works, 
                    # I think it is assuming there are 100 samples and just taking the number of samples listed in percentages.csv
                    self.maxclass = pd.read_csv("datasets/percentages.csv", index_col=0)
                    self.listOfCounts.mask(self.listOfCounts>self.maxclass,self.maxclass,inplace=True)
                #This removes all of the unused classes
                self.listOfCounts = self.listOfCounts.loc[self.use]

            #creates a random offset
            self.listOfCountsOffset = self.listOfCountsOffset.loc[self.use]
            self.listOfCountsOffset = self.listOfCountsOffset-self.listOfCounts
            if Config.datasetRandomOffset:
                self.listOfCountsOffset = self.listOfCountsOffset.applymap(lambda x: 0 if x<=0 else random.randrange(x))
            else:
                self.listOfCountsOffset = self.listOfCountsOffset.applymap(lambda x: 0)

        if self.length is None:
            self.length = self.listOfCounts.sum().item()
        return self.length


    def __getitem__(self, index) -> tuple([torch.Tensor,torch.Tensor]):
        """
        Gets the item at integer index. Note: you should not be calling this directly. Torch dataloaders will do that for you.

        parameters:
            index - the index of the item to retrieve.
        
        returns:
            tuple containing two tensors:
                first tensor - data, the data associated with a label
                second tensor is two dimentional:
                    first column - modified label, label with all unknown classes replaced with 15 for unknown.
                    second column - true label, the actual label of the class in integer form.
        
        """
        #For debug
        if index==self.length:
            print(index)

        #Now it needs to figure out what type of data it is using.
        chunktype = 0
        while index>=self.listOfCounts.iat[chunktype,0]:
            index -= self.listOfCounts.iat[chunktype,0]
            chunktype+=1
        
        index_before_offset = index
        chunkNumber = (index+self.listOfCountsOffset.iat[chunktype,0])//CHUNKSIZE
        index = (index+self.listOfCountsOffset.iat[chunktype,0])%CHUNKSIZE
        #This is needed incase it is not a full chunk so the max value is 0
        index = index%self.totalListOfCounts.iat[chunktype,0]

        t_start = time.time()
        try:
            chunk = pd.read_csv(self.path+f"/chunk{self.usedDict[chunktype]}{chunkNumber}.csv", index_col=False,chunksize=1,skiprows=index).get_chunk()
        except:
            print(f"Original index {index_before_offset}, Chunk Type{self.usedDict[chunktype]},Chunk Type {chunktype}, Chunk Number {chunkNumber}, Offset {self.listOfCountsOffset.iat[chunktype,0]}",flush=True)
            raise
        t_total = time.time()-t_start
        if t_total>1:
            print(f"load took {t_total:.2f} seconds")

        data, labels = self.seriesprocess(chunk.iloc[0])  
        
        #print(f"index: {index} does not exist in chunk: {chunkNumber} of type: {chunktype} ")

        item = data,labels

        return item

    def seriesprocess(self,x:pd.Series) -> tuple([torch.Tensor,torch.Tensor]):
        """
        This separates the data from the labels with series
        
        parameters:
            x - series to turn into tensors.
        
        returns:
            tuple containing:
                data - torch tensor of data values for a label
                label - the true class of the item in integer form.
        """

        data = x.iloc[:len(x)-1]
        data = torch.tensor(data.to_numpy())

        label = x.iloc[len(x)-1]         #This selects the label
        label = torch.tensor(int(label),dtype=torch.long)    #The int is because the loss function is expecting ints
        label.unsqueeze_(0)              #This is to allow it to be two dimentional
        if self.unknownData:
            label2 = torch.tensor(Config.parameters["CLASSES"][0],dtype=torch.long).unsqueeze_(0)    #unknowns are marked as unknown
        else:
            label2 = groupDoS(label.clone())
        label = torch.cat([label2,label], dim=0)


        return (data,label)
    
    def checkIfSplit(self, path=None):
        """
        This checks if the data is in the correct format, if it is not in the correct format it will generate the correct format.
        The correct format is clustered by type into chunks with a csv conainging the counts of all of the classes.

        Parameters:
            path - string containing the path to look for the dataset.
        
        """
        if path is None:
            path = self.path
        if not os.path.exists(os.path.join(path,"")): 
            os.mkdir(path)


            #this stores the data in dataframes
            runningDataFrames = []
            for c in LISTCLASS:
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
                        runningDataFrames[j][:10000].to_csv(path+f"/chunk{LISTCLASS[j]}{filecount[j]}.csv",index_label=False,index=False)
                        runningDataFrames[j] = runningDataFrames[j][10000:]
                        filecount[j] += 1

            count = [x*10000 for x in filecount]
            for j in range(len(runningDataFrames)):
                runningDataFrames[j].to_csv(path+f"/chunk{LISTCLASS[j]}{filecount[j]}.csv",index_label=False,index=False)
                count[j] += len(runningDataFrames[j])
            
            count = pd.DataFrame(count)
            count.to_csv(path+"counts.csv",index_label=False)

















class ClusterDivDataset(ClassDivDataset):
    """
    This version of the dataset is not directly used. 
    It is focused on creating multiple subclasses for each class and giving unique idenification number to each.
    However, the better implemented ClusterLimitDataset does inhearit some of its functionality from this version.
    
    """
    def __init__(self, path:str, use:list=None, unknownData=False, randomOffset=True):
        super().__init__(path,use,unknownData)
        """
        path is the string path that is the main datafile
        use is the list of integers corrispoding with the class dictionary above that you want to load.
        Unknown Data is if the dataset should only give unknown labels.
        
        If you want to make a dataloader that only reads benign data:
          train = Dataload.Dataset("Payload_data_CICIDS2017_Sorted",use=[0])
        "Payload_data_CICIDS2017_Sorted" is the main name for where the chunked data folder is
        use = [0] means that we are only using LISTCLASS[0] which is benign
        """
        

        self.perclassgroups = None
        self.clusters = 32
        self.minclass = 0
        self.path = path+"_Clustered"
        self.countspath = self.path+"/counts.csv"
        self.data_length = 1504

            
    def __len__(self) -> int:
        if self.listOfCounts is None:
            self.listOfCounts = pd.read_csv(self.countspath, index_col=0)
            self.totalListOfCounts = self.listOfCounts.copy()
            self.listOfCountsOffset = self.listOfCounts.copy()
            if "MaxSamples" in Config.parameters:
                #MAX SAMPLES DEFINITION:    (effectively percentage based distribution)
                #Max Samples limits the total number of samples to self.totalSamples
                #It then distributes those samples to match with the percentages given in percentages.csv as best as possible.
                maxclass = pd.read_csv("datasets/percentages.csv", index_col=None)
                maxclass = maxclass.iloc[Config.parameters["loopLevel"][0],:len(self.listOfCounts)]
                maxclass = ((torch.tensor(maxclass)/100)*self.totalSamples).ceil()
                for y in range(self.listOfCounts.shape[0]):
                    x=0
                    cutofflist = self.listOfCounts.iloc[y].copy()
                    cutofflist[cutofflist>x]  = x
                    while cutofflist.sum()<maxclass[y] and x<maxclass[y]:
                        x+=1
                        cutofflist = self.listOfCounts.iloc[y].copy()
                        cutofflist[cutofflist>x]  = x
                    self.listOfCounts.iloc[y] = cutofflist
                #This removes all of the unused classes
                self.listOfCounts = self.listOfCounts.loc[self.use]
                print(f"Items per class: \n{self.listOfCounts.sum(axis=1)}")
            else:
                if isinstance(Config.parameters["MaxPerClass"][0],int):
                    for y in range(self.listOfCounts.shape[0]):
                        x=0
                        cutofflist = self.listOfCounts.iloc[y].copy()
                        cutofflist[cutofflist>x]  = x
                        while cutofflist.sum()<self.maxclass and x<self.maxclass:
                            x+=1
                            cutofflist = self.listOfCounts.iloc[y].copy()
                            cutofflist[cutofflist>x]  = x
                        self.listOfCounts.iloc[y] = cutofflist
                else:
                    #This is a diffrent version of doing a MaxPerClass
                    #It revolves around the maxperclass being a percentage of the total number of samples of that class
                    #this preserves the distribution but may not make sense in all cases.
                    maxclass = [self.maxclass]*Config.parameters["CLASSES"][0]
                    maxclass = (torch.tensor(maxclass))
                    self.listOfCounts = torch.tensor(self.listOfCounts.to_numpy())
                    #test = torch.stack([maxclass]*listOfCounts.size()[1]).T
                    maxclass = self.listOfCounts.mul(torch.stack([maxclass]*self.listOfCounts.size()[1]).T).ceil()
                    self.listOfCounts[self.listOfCounts>maxclass] = maxclass[self.listOfCounts>maxclass].to(torch.long)
                    
                    self.listOfCounts = self.listOfCounts.numpy()
                    self.listOfCounts = pd.DataFrame(self.listOfCounts)
                #This removes all of the unused classes
                self.listOfCounts = self.listOfCounts.loc[self.use]
            
            #creates a random offset
            self.listOfCountsOffset = self.listOfCountsOffset.loc[self.use]
            self.listOfCountsOffset = self.listOfCountsOffset-self.listOfCounts
            if Config.datasetRandomOffset:
                self.listOfCountsOffset = self.listOfCountsOffset.applymap(lambda x: 0 if x<=0 else random.randrange(x))
            else:
                self.listOfCountsOffset = self.listOfCountsOffset.applymap(lambda x: 0)
        if self.length is None:
            self.length = self.listOfCounts.sum().sum().item()

        self.perclassgroups = (self.listOfCounts>self.minclass).sum(axis=1)
        return self.length


    def __getitem__(self, index) -> tuple([torch.Tensor,torch.Tensor]):
        """
        Gets the item at integer index. Note: you should not be calling this directly. Torch dataloaders will do that for you.

        parameters:
            index - the index of the item to retrieve.
        
        returns:
            tuple containing two tensors:
                first tensor - data, the data associated with a label
                second tensor is two dimentional:
                    first column - modified label, label with all unknown classes replaced with 15 for unknown.
                    second column - true label, the actual label of the class in integer form.
        
        """
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
        index_before_offset = index

        t_start = time.time()
        #print(f"/chunk{self.usedDict[chunktype]}-type{chunkNumber:03d}.csv")
        #print(f"supposed length:{self.listOfCounts.iat[chunktype,chunkNumber]}, Index:{index}")
        try:
            chunk = pd.read_csv(self.path+f"/chunk{self.usedDict[chunktype]}-type{chunkNumber:03d}.csv", index_col=False,chunksize=1,skiprows=index).get_chunk()
        except:
            print(f"Original index {index_before_offset}, Index with offset {index}, Chunk Number {chunkNumber}, Offset {self.listOfCountsOffset.iat[chunktype,0]}",flush=True)
            raise
        t_total = time.time()-t_start
        if t_total>1:
            print(f"load took {t_total:.2f} seconds")

            
        data, labels = self.seriesprocess(chunk.iloc[0],classNumber)  

        
        #print(f"index: {index} does not exist in chunk: {chunkNumber} of type: {chunktype} ")

        item = data,labels

        return item
    
    def checkIfSplit(self,path):
        """
        This checks if the data is in the correct format, if it is not in the correct format it will generate the correct format.
        The correct format is clustered by type into chunks with a csv conainging the counts of all of the classes.

        Parameters:
            path - string containing the path to look for the dataset.
        
        """
        if not os.path.exists(self.path+"_Clustered"): 
            print("Generating clustered data folder.")
            os.mkdir(self.path+"_Clustered")
            
            #Create an dataframe to store how many of each cluster there is
            counts = pd.DataFrame(0, index=range(Config.parameters["CLASSES"][0]),columns=range(self.clusters))
            #Read the data and make sure that the protocols and labels are numbers. Then convert them to integers
            data = pd.read_csv(path+".csv",converters={"protocol":protocalConvert,"label":classConvert})
            path=path+"_Clustered"
            for x in range(Config.parameters["CLASSES"][0]):
                X = data.astype(int)
                X = X[X["label"]==x]
                X = X.sample(n=100000 if 100000<len(X) else len(X))
                X2 = X.to_numpy()

                # setting distance_threshold=0 ensures we compute the full tree.
                model = AgglomerativeClustering(distance_threshold=None, n_clusters=self.clusters if self.clusters<len(X) else 1, compute_distances=True)

                model = model.fit(X2)
                lst = model.labels_
                bincount = np.bincount(lst,minlength=self.clusters)
                counts.iloc[x] = bincount
                for i in range(self.clusters):
                    X3 = X[lst==i]
                    X3.to_csv(self.path+"_Clustered"+f"/chunk{LISTCLASS[x]}-type{i:03d}.csv",index_label=False,index=False)
            counts.to_csv(f"{path}/counts.csv",index_label=False)

    def seriesprocess(self,x:pd.Series,classNumber:int) -> tuple([torch.Tensor,torch.Tensor]):
        """
        This separates the data from the labels with series
        
        parameters:
            x - series to turn into tensors.
        
        returns:
            tuple containing:
                data - torch tensor of data values for a label
                label - the true class of the item in integer form.
        """
        data = x.iloc[:len(x)-1]
        data = torch.tensor(data.to_numpy())

        label = torch.tensor(int(classNumber),dtype=torch.long)    #The int is because the loss function is expecting ints
        label.unsqueeze_(0)              #This is to allow it to be two dimentional
        if self.unknownData:
            #label2 = torch.tensor(self.perclassgroups.sum().item(),dtype=torch.long).unsqueeze_(0)    #unknowns are marked as unknown
            label2 = torch.tensor(Config.parameters["CLASSES"][0],dtype=torch.long).unsqueeze_(0)
        else:
            label2 = groupDoS(x.iloc[len(x)-1])         #This selects the label
            label2 = torch.tensor(int(label2),dtype=torch.long)    #The int is because the loss function is expecting ints
            label2.unsqueeze_(0)              #This is to allow it to be two dimentional
        label = torch.cat([label,label2], dim=0)


        return (data,label)


















class ClusterLimitDataset(ClusterDivDataset):
    """
    This version of the dataset will use agglomerative clustering to split each class into 32 subclasses.
    It will then take Config.parameters["MaxPerClass"] sample points from each subclass to compile the dataset.
    If a subclass does not have that number of samples it will take the maximum number of samples.

    The thought behind this is that the catagories we have are broad,
     so we want to take examples form each of the diffrent styles within these classes to train our model on.
    """
    def __init__(self, path:str, use:list=None, unknownData=False):
        """
        path is the string path that is the main datafile
        use is the list of integers corrispoding with the class dictionary above that you want to load.
        Unknown Data is if the dataset should only give unknown labels.
        
        If you want to make a dataloader that only reads benign data:
          train = Dataload.Dataset("Payload_data_CICIDS2017_Sorted",use=[0])
        "Payload_data_CICIDS2017_Sorted" is the main name for where the chunked data folder is
        use = [0] means that we are only using LISTCLASS[0] which is benign
        
        """
        self.perclassgroups = None
        self.clusters = 32
        self.minclass = 0
        self.data_length = 1504
        super().__init__(path,use,unknownData)
        
        

        
            
    def __len__(self) -> int:
        if self.length is None:
            super().__len__()
            self.perclassgroups = (self.listOfCounts>self.minclass).sum(axis=1)
            # print(self.listOfCounts)
        return self.length


    def __getitem__(self, index) -> tuple([torch.Tensor,torch.Tensor]):
        """
        Gets the item at integer index. Note: you should not be calling this directly. Torch dataloaders will do that for you.

        parameters:
            index - the index of the item to retrieve.
        
        returns:
            tuple containing two tensors:
                first tensor - data, the data associated with a label
                second tensor is two dimentional:
                    first column - modified label, label with all unknown classes replaced with 15 for unknown.
                    second column - true label, the actual label of the class in integer form.
        
        """
        #For debug
        if index==self.length:
            print(index)

        #Now it needs to figure out what type of data it is using.
        chunktype = 0
        chunkNumber = 0
        classNumber = 0
        while index>self.listOfCounts.iat[chunktype,chunkNumber]:

            if self.listOfCounts.iat[chunktype,chunkNumber]>self.minclass:
                classNumber+=1
                index -= self.listOfCounts.iat[chunktype,chunkNumber]

            #look at next chunk
            chunkNumber+=1
            
            #looked at all of the chunks of this type (32 chunks)
            if chunkNumber>=self.clusters:
                chunktype+=1
                chunkNumber = 0
        
        index_before_offset = index
        index += self.listOfCountsOffset.iat[chunktype,chunkNumber].item()

        t_start = time.time()
        #print(f"/chunk{self.usedDict[chunktype]}-type{chunkNumber:03d}.csv")
        #print(f"supposed length:{self.listOfCounts.iat[chunktype,chunkNumber]}, Index:{index}")
        try:
            chunk = pd.read_csv(self.path+f"/chunk{self.usedDict[chunktype]}-type{chunkNumber:03d}.csv", index_col=False,chunksize=1,skiprows=index-1).get_chunk()
        except:
            print(f"Original index {index_before_offset}, Index with offset {index}, Chunk Number {chunkNumber}, Offset {self.listOfCountsOffset.iat[chunktype,0]}",flush=True)
            raise
        t_total = time.time()-t_start
        if t_total>1:
            print(f"load took {t_total:.2f} seconds")

            
        data, labels = self.seriesprocess(chunk.iloc[0])  

        
        #print(f"index: {index} does not exist in chunk: {chunkNumber} of type: {chunktype} ")

        item = data,labels

        return item
    


    def seriesprocess(self,x:pd.Series) -> tuple([torch.Tensor,torch.Tensor]):
        """
        This checks if the data is in the correct format, if it is not in the correct format it will generate the correct format.
        The correct format is clustered by type into chunks with a csv conainging the counts of all of the classes.

        Parameters:
            path - string containing the path to look for the dataset.
        
        """

        data = x.iloc[:len(x)-1]
        data = torch.tensor(data.to_numpy())

        label = x.iloc[len(x)-1]         #This selects the label
        label = torch.tensor(int(label),dtype=torch.long)    #The int is because the loss function is expecting ints
        label.unsqueeze_(0)              #This is to allow it to be two dimentional
        if self.unknownData:
            #unknowns are marked as unknown
            label2 = torch.tensor(Config.parameters["CLASSES"][0],dtype=torch.long).unsqueeze_(0)
        else:
            label2 = groupDoS(label.clone())
        label = torch.cat([label2,label], dim=0)


        return (data,label)


        

#https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset
class DatasetWithFlows(IterableDataset):
    def __init__(self, path="", use:list=None, unknownData=False, state_worker_loads = False):
        """
        initializes the dataloader. NOTE: DatasetWithFlows does not work with current code due to train/test split
        
        parameters: 
            use - the list of integers corrispoding with the class dictionary above that you want to load.
            Unknown Data - if the dataset should only give unknown labels.
            path - unused
            state_worker_loads - debug option that prints data about the different workers ('state' as in 'make a statement')
        
        If you want to make a dataloader that only reads benign data:
          train = Dataload.DatasetWithFlows(use=[0])
        The name of the dataset is taken from Config
        """
        super(DatasetWithFlows).__init__()
        from nids_datasets import Dataset as Data_set_with_flows
        
        self.state_worker_loads = state_worker_loads
        self.unknownData=unknownData
        if Config.parameters["Dataset"][0] == "Payload_data_CICIDS2017":
            self.dataset_name = 'CIC-IDS2017'
        elif Config.parameters["Dataset"][0] == "Payload_data_CICIDS2017":
            self.dataset_name = 'UNSW-NB15'
        else:
            raise ValueError("Invalid name of dataset")
        self.df = Data_set_with_flows(dataset=self.dataset_name,subset=["Payload-Bytes"],files="all")
        
        self.use = [LISTCLASS[x] for x in use]
        self.notuse = [LISTCLASS[x] for x in range(Config.parameters["CLASSES"][0]) if LISTCLASS[x] not in self.use]
        if 'Web Attack – Sql Injection' in self.use:
            self.use[self.use.index('Web Attack – Sql Injection')] = 'Web Attack – SQL Injection'
        if 'Web Attack – Sql Injection' in self.notuse:
            self.notuse[self.notuse.index('Web Attack – Sql Injection')] = 'Web Attack – SQL Injection'

        self.files_to_acces = None

        self.initializeCounts()
        print(self.filecounts)
        self.n_shards = 18
        

    def __iter__(self):
        """
        Returns an iterator over the data
        """
        self.initializeCounts()
        self.filecounts = self.filecounts.sum()
        #https://docs.python.org/3/library/itertools.html#itertools.filterfalse
        return map(self.process_data,filterfalse(self.check_invalid,self.df.read(files=[x+1 for x in self.files_to_acces],stream=True)))
        #For some reason the files are 1 indexed.
    
    def initializeCounts(self):
        """
        Initialize how many of each class the dataloader shoud return. All the rest are skipped.
        """
        from nids_datasets import DatasetInfo
        self.dfInfo = DatasetInfo(self.dataset_name)
        self.dfInfo.drop(columns="total",inplace=True)
        self.listOfCounts = self.dfInfo.sum()
        percentages = self.dfInfo/self.listOfCounts
        # print(percentages)
        #https://stackoverflow.com/a/35125872
        self.filecounts = -((-percentages * Config.parameters["MaxPerClass"][0])//1)
        for name in self.notuse:
            self.filecounts[name] *=0
        # print(self.filecounts.sum())
        # print(self.filecounts.sum(axis=1))
        if self.files_to_acces is None:
            self.files_to_acces = [x for x,a in enumerate(self.filecounts.sum(axis=1)) if a>0]
        self.filecounts = self.filecounts.iloc[self.files_to_acces]
        self.length = self.filecounts.sum().sum()

    def __len__(self):
        """
        Returns the length
        """
        return int(self.length)

    @staticmethod
    def worker_init_fn(id):
        """
        worker_init_fn is the function that is called when creating each worker. 
        This function splits up the iterator data into the different files for each worker as evently as possible.
        It then recalculates the length that it will need to run for.

        parameters:
            id - worker id number
        """
        worker_info = torch.utils.data.get_worker_info()
        self = worker_info.dataset
        if worker_info.id >= len(self.files_to_acces):
                import sys
                print(f"Too many workers, the flows dataset can only have {len(self.files_to_acces)} workers maximum. The data will be repeated",file=sys.stderr)

        #https://stackoverflow.com/a/4260304
        self.files_to_acces = [x for x in self.files_to_acces if x%worker_info.num_workers == worker_info.id%18]
        self.initializeCounts()
        # self.filecounts = self.filecounts.iloc[self.files_to_acces].sum()
        if self.state_worker_loads is True:
            print(f"Worker {worker_info.id}, with id {id}, is handling files {self.files_to_acces} and has {self.filecounts.sum().sum()} items")


    def dictionaryprocess(self,x:dict) -> tuple([torch.Tensor,torch.Tensor]):
        """
        This separates the data from the labels
        
        parameters:
            x - dictionary to turn into tensors.
        
        returns:
            tuple containing:
                data - torch tensor of data values for a label
                label - the true class of the item in integer form.
        """
        true_label = x["attack_label"]
        # index = x["index"]
        flow_id = x["flow_id"]
        data = x.copy()
        data.pop("attack_label")
        data.pop("flow_id")
        data = torch.tensor([int(val[1]) if val[1] is not None else 0 for val in data.items()])

        true_label = torch.tensor(int(true_label),dtype=torch.long)    #The int is because the loss function is expecting ints
        # index = torch.tensor(int(index),dtype=torch.long)
        flow_id = torch.tensor(int(flow_id),dtype=torch.long)
        true_label.unsqueeze_(0)              #This is to allow it to be two dimentional
        # index.unsqueeze_(0)
        flow_id.unsqueeze_(0)
        if self.unknownData:
            obsficated_label = torch.tensor(Config.parameters["CLASSES"][0],dtype=torch.long).unsqueeze_(0)    #unknowns are marked as unknown
        else:
            obsficated_label = groupDoS(true_label.clone())
        true_label = torch.cat([obsficated_label,true_label,flow_id], dim=0)


        return (data,true_label)
    
    def process_data(self,item:dict) -> tuple([torch.Tensor,torch.Tensor]):
        """
        This method removes all non-numeric columns from the dictionary and then runs dictionary process to turn it into tensors.

        parameters:
            item - a dictionary containing all of the values of the data.
        returns:
            tuple containing:
                data - torch tensor of data values for a label
                label - the true class of the item in integer form.

        """
        item["protocol"] = protocalConvert(item["protocol"])
        if item["attack_label"] == 'Web Attack – SQL Injection':
            item["attack_label"] = CLASSLIST['Web Attack – Sql Injection']
        else:
            item["attack_label"] = classConvert(item["attack_label"])
        for a,x in enumerate(item.pop("destination_ip").split(sep='.')):
            item[f"destination_ip_pt{a}"] = x
        for a,x in enumerate(item.pop("source_ip").split(sep='.')):
            item[f"source_ip_pt{a}"] = x
        return self.dictionaryprocess(item)

    def check_invalid(self,item:dict) -> bool:
        """
        Checks if this row should be thrown out based on how many of each class is called for.

        """
        if self.filecounts[item["attack_label"]] <=0:
            if self.filecounts.sum() <= 0:
                print(f"Worker {torch.utils.data.get_worker_info().id} has stopped.")
                raise StopIteration
            return True
        
        self.filecounts[item["attack_label"]]-= 1
        if torch.utils.data.get_worker_info().id == 5:
            print(f"Worker {torch.utils.data.get_worker_info().id} has {self.filecounts.sum()} items left.")
        assert torch.utils.data.get_worker_info().dataset is self
        return False

        



class ClassDivDataset_flows(Dataset):
    def __init__(self, path:str, use:list=None, unknownData=False):
        """
        initializes the dataloader.
        
        parameters: 
            path is the string path that is the main datafile
            use is the list of integers corrispoding with the class dictionary above that you want to load.
            Unknown Data is if the dataset should only give unknown labels.
        
        If you want to make a dataloader that only reads benign data:
          train = Dataload.Dataset("Payload_data_CICIDS2017",use=[0])
        "Payload_data_CICIDS2017" is the main name for where the chunked data folder is
        use = [0] means that we are only using LISTCLASS[0] which is benign
        """
        from nids_datasets import DatasetInfo
        self.unknownData=unknownData
        self.path = path+"_with_flows"
        if "CIC" in path:
            length_name = "CIC-IDS2017"
        elif "UNSW" in path:
            length_name = "UNSW-NB15"
        
        self.data_length = 1514
        
        self.listOfCounts = DatasetInfo(length_name).sum()
        self.listOfCounts.drop(labels="total",inplace=True)
        self.length = None
        self.maxclass = Config.parameters["MaxPerClass"][0]
        if "MaxSamples" in Config.parameters:
            self.totalSamples = Config.parameters["MaxSamples"][0]
        
        
        self.use_numerical = use.copy()
        #This is setting what classes are considered to be knowns.
        self.classlist_with_uppercase = LISTCLASS.copy()
        # print(self.listOfCounts.keys())
        self.classlist_with_uppercase[14] = 'Web Attack – SQL Injection'
        if use is not None:
            self.use_mask = [case in [self.classlist_with_uppercase[x] for x in use] for case in self.listOfCounts.keys()]
            self.usedDict = {count:LISTCLASS[case] for count,case in enumerate(use)}
        else:
            self.use_mask = [case in [self.classlist_with_uppercase[x] for x in self.classlist_with_uppercase.keys()] for case in self.listOfCounts.keys()]
            self.usedDict = LISTCLASS.copy()

        #this will check if the file is chunked and chunk it if it is not
        self.checkIfSplit(path)

    def __len__(self) -> int:
        """
        Finds and saves the length of the dataloader.

        returns an intger of the length of the data.
        
        """
        if self.length is None:
            if "MaxSamples" in Config.parameters:
                #MAX SAMPLES DEFINITION:
                #Max Samples limits the total number of samples to self.totalSamples
                #It then distributes those samples to match with the percentages given in percentages.csv as best as possible.
                maxperclass = pd.read_csv("datasets/percentages.csv", index_col=None)
                maxperclass = maxperclass.iloc[Config.parameters["loopLevel"][0],:len(self.listOfCounts)]
                maxperclass = ((torch.tensor(maxperclass)/100)*self.totalSamples).ceil()
                for x in range(len(self.listOfCounts)):
                    if self.listOfCounts.iloc[x,0] > maxperclass[x].item():
                        self.listOfCounts.iloc[x,0] = maxperclass[x].item()
                #This removes all of the unused classes
                self.listOfCounts = self.listOfCounts.loc[self.use_mask]
                print(f"Items per class: \n{self.listOfCounts}")
            else:
                #add max per class
                if isinstance(self.maxclass,int):
                    #Huh, this only runs if Config MaxPerClass is an integer. 
                    # But it takes x samples of each class where x is MaxPerClass
                    self.listOfCounts.mask(self.listOfCounts>self.maxclass,self.maxclass,inplace=True)
                elif self.maxclass == "File":
                    #Not quite sure how this works, 
                    # I think it is assuming there are 100 samples and just taking the number of samples listed in percentages.csv
                    self.maxclass = pd.read_csv("datasets/percentages.csv", index_col=0)
                    self.listOfCounts.mask(self.listOfCounts>self.maxclass,self.maxclass,inplace=True)
                #This removes all of the unused classes
                self.listOfCounts.loc[self.use_mask] *= 0
            self.length = self.listOfCounts.sum().item()
        return self.length


    def __getitem__(self, index) -> tuple([torch.Tensor,torch.Tensor]):
        """
        Gets the item at integer index. Note: you should not be calling this directly. Torch dataloaders will do that for you.

        parameters:
            index - the index of the item to retrieve.
        
        returns:
            tuple containing two tensors:
                first tensor - data, the data associated with a label
                second tensor is two dimentional:
                    first column - modified label, label with all unknown classes replaced with 15 for unknown.
                    second column - true label, the actual label of the class in integer form.
        
        """
        #For debug
        if index>=self.length:
            print(index)
        
        test = index
        class_possibility = 0
        while index>=self.listOfCounts[self.classlist_with_uppercase[class_possibility]]:
            index -= self.listOfCounts[self.classlist_with_uppercase[class_possibility]]
            class_possibility+=1
        clas = class_possibility

        index_before_offset = index

        t_start = time.time()
        try:
            chunk = pd.read_csv(self.path+f"/{LISTCLASS[clas]}.csv", index_col=False,chunksize=1,skiprows=range(1,index),header=0).get_chunk()
        except:
            print(f"Original index {index_before_offset}, Index with offset {index}, Class number {clas}, Offset {0}",flush=True)
            raise
        # print(chunk)
        t_total = time.time()-t_start
        if t_total>1:
            print(f"load took {t_total:.2f} seconds")

        data, labels = self.seriesprocess(chunk.iloc[0].copy())  
        
        #print(f"index: {index} does not exist in chunk: {chunkNumber} of type: {chunktype} ")

        item = data,labels

        return item

    def seriesprocess(self,x:pd.Series) -> tuple([torch.Tensor,torch.Tensor]):
        """
        This separates the data from the labels with series
        
        parameters:
            x - series to turn into tensors.
        
        returns:
            tuple containing:
                data - torch tensor of data values for a label
                label - the true class of the item in integer form.
        """
        x.fillna(0,inplace=True)
        true_label = x["label"]
        if "index.1" in x.keys():
            x["index"] = x.pop("index.1")
        index = x["index"]
        flow_id = x["flow_id"]
        data = x.copy()
        data.pop("label")
        data.pop("index")
        data.pop("flow_id")
        data = torch.tensor([int(val[1]) for val in data.items()])

        true_label = torch.tensor(int(true_label),dtype=torch.long)    #The int is because the loss function is expecting ints
        index = torch.tensor(int(index),dtype=torch.long)
        flow_id = torch.tensor(int(flow_id),dtype=torch.long)
        true_label.unsqueeze_(0)              #This is to allow it to be two dimentional
        index.unsqueeze_(0)
        flow_id.unsqueeze_(0)
        if self.unknownData:
            obsficated_label = torch.tensor(Config.parameters["CLASSES"][0],dtype=torch.long).unsqueeze_(0)    #unknowns are marked as unknown
        else:
            obsficated_label = groupDoS(true_label.clone())
        true_label = torch.cat([obsficated_label,true_label,flow_id,index], dim=0)


        return (data,true_label)
    
    def checkIfSplit(self, path=None):
        """
        This checks if the data is in the correct format, if it is not in the correct format it will generate the correct format.
        The correct format is clustered by type into chunks with a csv conainging the counts of all of the classes.

        Parameters:
            path - string containing the path to look for the dataset.
        
        """
        if path is None:
            path = self.path
        if False in [os.path.exists(f"datasets/{Config.parameters['Dataset'][0]}_with_flows/{LISTCLASS[clas]}.csv") for clas in self.use_numerical]:
            if False in [os.path.exists(f"datasets/{Config.parameters['Dataset'][0]}_with_flows/file{x+1}") for x in range(18)]:
                files_to_refresh = [x+1 for x in range(18) if not os.path.exists(f"datasets/{Config.parameters['Dataset'][0]}_with_flows/file{x+1}")]
                run_demo(self.split_flows_dataset,len(files_to_refresh),files_to_refresh)
            self.join_split_flows_dataset([x for x in CLASSLIST.keys() if not os.path.exists(f"datasets/{Config.parameters['Dataset'][0]}_with_flows/{x}.csv")])

    @staticmethod
    def split_flows_dataset(worker=None, worldsize=18,file=None):
        """
        Downloads the dataset into csv files. The saved csv files are saved first by original file number and then by class.

        parameters:
            worker - worker ID number
            world size - how many workers exist.
            file - three possibilities:
                -None - loads all files after splitting as evenly as possible with the worker count.
                -integer - loads exactly that file (filenumber +1 because the files are 1 indexed)
                -list - loads the specific files given split as evenly as possible over the workers.
        """
        from nids_datasets import Dataset as Data_set_with_flows
        if Config.parameters["Dataset"][0] == "Payload_data_CICIDS2017":
            dataset_name = 'CIC-IDS2017'
        elif Config.parameters["Dataset"][0] == "Payload_data_CICIDS2017":
            dataset_name = 'UNSW-NB15'
        else:
            raise ValueError("Invalid name of dataset")
        
        if file is None:
            files = [x+1 for x in range(18) if x%worldsize == worker]
        elif isinstance(file, int):
            files = [file+1]
        elif isinstance(file,list):
            files = [x for x in file if (x-1)%worldsize == worker]
        else:
            files = file

        def fixes(item:dict):
            item["protocol"] = protocalConvert(item["protocol"])
            if item["attack_label"] == 'Web Attack – SQL Injection':
                item["label"] = CLASSLIST['Web Attack – Sql Injection']
            else:
                item["label"] = classConvert(item["attack_label"])
            item.pop("attack_label")
            for a,x in enumerate(item.pop("destination_ip").split(sep='.')):
                item[f"destination_ip_pt{a}"] = x
            for a,x in enumerate(item.pop("source_ip").split(sep='.')):
                item[f"source_ip_pt{a}"] = x
            return item

        df = Data_set_with_flows(dataset=dataset_name,subset=["Payload-Bytes"],files="all")

        if not os.path.exists(f"datasets/{Config.parameters['Dataset'][0]}_with_flows"):
            os.mkdir(f"datasets/{Config.parameters['Dataset'][0]}_with_flows")

        for file in files:
            if not os.path.exists(f"datasets/{Config.parameters['Dataset'][0]}_with_flows/file{file}"):
                os.mkdir(f"datasets/{Config.parameters['Dataset'][0]}_with_flows/file{file}")
            for num,item in enumerate(map(fixes,df.read(files=[file],stream=True))):
                item["index"]=num
                #https://stackoverflow.com/a/68206394
                item_df = pd.Series(item).to_frame().T
                if os.path.exists(f"datasets/{Config.parameters['Dataset'][0]}_with_flows/file{file}/{LISTCLASS[item['label']]}.csv"):
                    item_df.to_csv(f"datasets/{Config.parameters['Dataset'][0]}_with_flows/file{file}/{LISTCLASS[item['label']]}.csv",mode='a',header=False,index_label="index")
                else:
                    item_df.to_csv(f"datasets/{Config.parameters['Dataset'][0]}_with_flows/file{file}/{LISTCLASS[item['label']]}.csv",index_label="index")
                if num%10000 == 0:
                    print(f"{num} rows finished in file {file}")
        print(f"{file} finished.")

    @staticmethod
    def join_split_flows_dataset(classes,deleteOld=False):
        """
        Joins the data created by split_flows_dataset() classwise into single csv files per class.

        parameters:
            clssses - list of names of classes to be compiled.
            deleteOld - Deletes the file separated versions of the data as it is being processed.
        """
        for file in range(18):
            for clas in classes:
                if os.path.exists(f"datasets/{Config.parameters['Dataset'][0]}_with_flows/file{file+1}/{clas}.csv"):
                    csv = pd.read_csv(f"datasets/{Config.parameters['Dataset'][0]}_with_flows/file{file+1}/{clas}.csv")
                    if os.path.exists(f"datasets/{Config.parameters['Dataset'][0]}_with_flows/{clas}.csv"):
                        mod = 'a'
                        header = False
                    else:
                        mod = 'w'
                        header = True
                    if deleteOld:
                        os.remove(f"datasets/{Config.parameters['Dataset'][0]}_with_flows/file{file+1}/{clas}.csv")
                    csv.to_csv(f"datasets/{Config.parameters['Dataset'][0]}_with_flows/{clas}.csv",header=header,mode=mod)
            if deleteOld:
                os.remove(f"datasets/{Config.parameters['Dataset'][0]}_with_flows/file{file+1}")
            print(f"File {file} consolidation finished.")


#modified from the torch multiprocessing demo:
import torch.multiprocessing as mp
def run_demo(demo_fn, world_size,other=None):
    mp.spawn(demo_fn,
            args=(world_size,other),
            nprocs=world_size,
            join=True)
        

DataloaderTypes = {"Standard":ClassDivDataset,"Old_Cluster":ClusterDivDataset,"Cluster":ClusterLimitDataset,"Slow_Flows":DatasetWithFlows,"Flows":ClassDivDataset_flows}



if __name__ == "__main__":
    # run_demo(testing,18)
    # print("Testing if this waits for the data to finish.")
    from nids_datasets import DatasetInfo
    print(DatasetInfo('CIC-IDS2017').sum())
    if False:
        t = DataLoader(DatasetWithFlows(use=[0,1,2,3,4,5],unknownData=False,state_worker_loads=True), Config.parameters["batch_size"][0], num_workers=5,pin_memory=True,worker_init_fn=DatasetWithFlows.worker_init_fn)
        startTime = time.time()
        for b,a in tqdm.tqdm(enumerate(t)):
            pass
        print(f"Finished in:{time.time()-startTime}")