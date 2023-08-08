import pandas as pd
import torch
from torch.utils.data import Dataset, IterableDataset
import numpy as np
import os
import time
import Config
from sklearn.cluster import AgglomerativeClustering
from itertools import filterfalse
import tqdm


#List of conversions:
if Config.parameters["Dataset"][0] == "Payload_data_CICIDS2017":
    CLASSLIST = {0: 'BENIGN', 1: 'Infiltration', 2: 'Bot', 3: 'PortScan', 4: 'DDoS', 5: 'FTP-Patator', 6: 'SSH-Patator', 7: 'DoS slowloris', 8: 'DoS Slowhttptest', 9: 'DoS Hulk', 10: 'DoS GoldenEye', 11: 'Heartbleed', 12: 'Web Attack – Brute Force', 13: 'Web Attack – XSS', 14:'Web Attack – Sql Injection'}
elif Config.parameters["Dataset"][0] == "Payload_data_UNSW":
    CLASSLIST = {0:"analysis",1:"backdoor",2:"dos",3:"exploits",4:"fuzzers",5:"generic",6:"normal",7:"reconnaissance",8:"shellcode",9:"worms"}
else:
    print("ERROR, Dataset not implemented")
#PROTOCOLS = {"udp":0,"tcp":1}
PROTOCOLS = {"udp":0,"tcp":1,"others":2,"ospf":3,"sctp":4,"gre":5,"swipe":6,"mobile":7,"sun-nd":8,"sep":9,"unas":10,"pim":11,"secure-vmtp":12,"pipe":13,"etherip":14,"ib":15,"ax.25":16,"ipip":17,"sps":18,"iplt":19,"hmp":20,"ggp":21,"ipv6":22,"rdp":23,"rsvp":24,"sccopmce":25,"egp":26,"vmtp":27,"snp":28,"crtp":29,"emcon":30,"nvp":31,"fire":32,"crudp":33,"gmtp":34,"dgp":35,"micp":36,"leaf-2":37,"arp":38,"fc":39,"icmp":40,"other":41}
LISTCLASS = {CLASSLIST[x]:x for x in range(Config.parameters["CLASSES"][0])}
CHUNKSIZE = 10000

def groupDoS(x):
    if False and Config.parameters["Dataset"][0] == "Payload_data_CICIDS2017":
        x[x>=7 and x<=10] = 7
    return x

def classConvert(x):
    """
    Does a conversion based on the dictionaries
    """
    return LISTCLASS[x]
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
        """
        initializes the dataloader.
        
        parameters: 
            path is the string path that is the main datafile
            use is the list of integers corrispoding with the class dictionary above that you want to load.
            Unknown Data is if the dataset should only give unknown labels.
        
        If you want to make a dataloader that only reads benign data:
          train = Dataload.Dataset("Payload_data_CICIDS2017_Sorted",use=[0])
        "Payload_data_CICIDS2017_Sorted" is the main name for where the chunked data folder is
        use = [0] means that we are only using CLASSLIST[0] which is benign
        """

        self.unknownData=unknownData
        self.path = path
        self.countspath = path+"counts.csv"
        self.length = None
        self.listOfCounts = None
        self.maxclass = Config.parameters["MaxPerClass"][0]
        if "MaxSamples" in Config.parameters:
            self.totalSamples = Config.parameters["MaxSamples"][0]
        
        

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
        """
        Finds and saves the length of the dataloader.

        returns an intger of the length of the data.
        
        """
        if self.listOfCounts is None:
            self.listOfCounts = pd.read_csv(self.countspath, index_col=0)
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
    """
    This version of the dataset is not directly used. 
    It is focused on creating multiple subclasses for each class and giving unique idenification number to each.
    However, the better implemented ClusterLimitDataset does inhearit some of its functionality from this version.
    
    """
    def __init__(self, path:str, use:list=None, unknownData=False):
        super().__init__(path,use,unknownData)
        """
        path is the string path that is the main datafile
        use is the list of integers corrispoding with the class dictionary above that you want to load.
        Unknown Data is if the dataset should only give unknown labels.
        
        If you want to make a dataloader that only reads benign data:
          train = Dataload.Dataset("Payload_data_CICIDS2017_Sorted",use=[0])
        "Payload_data_CICIDS2017_Sorted" is the main name for where the chunked data folder is
        use = [0] means that we are only using CLASSLIST[0] which is benign
        """
        

        self.perclassgroups = None
        self.clusters = 32
        self.minclass = 0
        self.path = path+"_Clustered"
        self.countspath = self.path+"/counts.csv"
            
    def __len__(self) -> int:
        if self.listOfCounts is None:
            self.listOfCounts = pd.read_csv(self.countspath, index_col=0)
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
                    X3.to_csv(self.path+"_Clustered"+f"/chunk{CLASSLIST[x]}-type{i:03d}.csv",index_label=False,index=False)
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
        use = [0] means that we are only using CLASSLIST[0] which is benign
        
        """
        self.perclassgroups = None
        self.clusters = 32
        self.minclass = 0
        super().__init__(path,use,unknownData)
        
        

        
            
    def __len__(self) -> int:
        if self.length is None:
            super().__len__()
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
        

        t_start = time.time()
        #print(f"/chunk{self.usedDict[chunktype]}-type{chunkNumber:03d}.csv")
        #print(f"supposed length:{self.listOfCounts.iat[chunktype,chunkNumber]}, Index:{index}")
        chunk = pd.read_csv(self.path+f"/chunk{self.usedDict[chunktype]}-type{chunkNumber:03d}.csv", index_col=False,chunksize=1,skiprows=index-1).get_chunk()
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
        initializes the dataloader. NOTE: DatasetWithFlows does not work with LOOP == 2
        
        parameters: 
            use is the list of integers corrispoding with the class dictionary above that you want to load.
            Unknown Data is if the dataset should only give unknown labels.
            path is unused
        
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
        
        self.use = [CLASSLIST[x] for x in use]
        self.notuse = [CLASSLIST[x] for x in range(Config.parameters["CLASSES"][0]) if CLASSLIST[x] not in self.use]
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

    @staticmethod
    def worker_init_fn(id):
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
            item["attack_label"] = LISTCLASS['Web Attack – Sql Injection']
        else:
            item["attack_label"] = classConvert(item["attack_label"])
        for a,x in enumerate(item.pop("destination_ip").split(sep='.')):
            item[f"destination_ip_pt{a}"] = x
        for a,x in enumerate(item.pop("source_ip").split(sep='.')):
            item[f"source_ip_pt{a}"] = x
        return self.dictionaryprocess(item)

    def check_invalid(self,item:dict) -> bool:
        """
        Checks if this row should be thrown out.
        #TODO: Modify to keep count of classes to limit the number of samples.


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

        
from torch.utils.data import TensorDataset, DataLoader
import copy
import tqdm
#Try to store all of the data in memory instead?
def recreateDL(dl:torch.utils.data.DataLoader,shuffle=True):
    xList= []
    yList= []
    for xs,ys in tqdm.tqdm(dl,desc="Loading Dataloader into memory"):
        #https://github.com/pytorch/pytorch/issues/11201#issuecomment-486232056
        xList.append(copy.deepcopy(xs))
        del(xs)
        yList.append(copy.deepcopy(ys))
        del(ys)
    xList = torch.cat(xList)
    yList = torch.cat(yList)
    combinedList = TensorDataset(xList,yList)
    return DataLoader(combinedList, Config.parameters["batch_size"][0], shuffle=shuffle, num_workers=Config.parameters["num_workers"][0],pin_memory=False)



if __name__ == "__main__":
    from torch.utils.data import TensorDataset, DataLoader
    import copy
    import tqdm
    t = DataLoader(DatasetWithFlows(use=[0,1,2,3,4,5],unknownData=False,state_worker_loads=True), Config.parameters["batch_size"][0], num_workers=5,pin_memory=True,worker_init_fn=DatasetWithFlows.worker_init_fn)
    startTime = time.time()
    for b,a in tqdm.tqdm(enumerate(t)):
        pass
    print(f"Finished in:{time.time()-startTime}")