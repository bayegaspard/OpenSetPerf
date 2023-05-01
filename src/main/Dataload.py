import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
import os
import time
import Config
from sklearn.cluster import AgglomerativeClustering

#List of conversions:
if Config.parameters["Dataset"][0] == "Payload_data_CICIDS2017":
    CLASSLIST = {0: 'BENIGN', 1: 'Infiltration', 2: 'Bot', 3: 'PortScan', 4: 'DDoS', 5: 'FTP-Patator', 6: 'SSH-Patator', 7: 'DoS slowloris', 8: 'DoS Slowhttptest', 9: 'DoS Hulk', 10: 'DoS GoldenEye', 11: 'Heartbleed', 12: 'Web Attack – Brute Force', 13: 'Web Attack – XSS', 14:'Web Attack – Sql Injection'}
elif Config.parameters["Dataset"][0] == "Payload_data_UNSW":
    CLASSLIST = {0:"analysis",1:"backdoor",2:"dos",3:"exploits",4:"fuzzers",5:"generic",6:"normal",7:"reconnaissance",8:"shellcode",9:"worms"}
else:
    print("ERROR, Dataset not implemented")
#PROTOCOLS = {"udp":0,"tcp":1}
PROTOCOLS = {"udp":0,"tcp":1,"others":2,"ospf":3,"sctp":4,"gre":5,"swipe":6,"mobile":7,"sun-nd":8,"sep":9,"unas":10,"pim":11,"secure-vmtp":12,"pipe":13,"etherip":14,"ib":15,"ax.25":16,"ipip":17,"sps":18,"iplt":19,"hmp":20,"ggp":21,"ipv6":22,"rdp":23,"rsvp":24,"sccopmce":25,"egp":26,"vmtp":27,"snp":28,"crtp":29,"emcon":30,"nvp":31,"fire":32,"crudp":33,"gmtp":34,"dgp":35,"micp":36,"leaf-2":37,"arp":38,"fc":39,"icmp":40}
LISTCLASS = {CLASSLIST[x]:x for x in range(Config.parameters["CLASSES"][0])}
CHUNKSIZE = 10000

def groupDoS(x):
    if Config.parameters["Dataset"][0] == "Payload_data_CICIDS2017" and False:
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
            #add max per class
            if isinstance(self.maxclass,int):
                self.listOfCounts.mask(self.listOfCounts>self.maxclass,self.maxclass,inplace=True)
            elif self.maxclass == "File":
                maxclass = pd.read_csv("datasets/percentages.csv", index_col=0)
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
                maxclass = [self.maxclass]*Config.parameters["CLASSES"][0]
                maxclass = (torch.tensor(maxclass)/100)
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
                X = X.sample(n=200 if 200<len(X) else len(X))
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


        
        
from torch.utils.data import TensorDataset, DataLoader
#Try to store all of the data in memory instead?
def recreateDL(dl:torch.utils.data.DataLoader):
    xList= []
    yList= []
    for xs,ys in dl:
        xList.append(xs)
        yList.append(ys)
    xList = torch.cat(xList)
    yList = torch.cat(yList)
    combinedList = TensorDataset(xList,yList)
    return DataLoader(combinedList, Config.parameters["batch_size"][0], shuffle=True, num_workers=Config.parameters["num_workers"][0],pin_memory=True)
