#note, this is a very modified version of a dataloader found in https://www.youtube.com/watch?v=ZoZHd0Zm3RY
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
import torch.nn as nn

class NetworkDataset(Dataset):
    def __init__(self,csv_files,transforms=None):
        self.transforms = transforms
        self.isOneHot = True
        self.list = []
        classlist = []
        for x,_ in enumerate(csv_files):
            self.list.append(pd.read_csv(csv_files[x],header=0))
            self.list[x].fillna(-1,inplace=True)
            self.list[x].replace(np.inf, 0, inplace=True)
            self.list[x].replace(-np.inf, 0, inplace=True)
            classlist.append(self.list[x][" Label"].unique())

        #find how many classes are in the data 
        classlist = np.concatenate(classlist)
        self.classes = pd.DataFrame(classlist)[0].unique()
        classes = {}
        classes = {b:a for a,b in enumerate(self.classes)}
        self.classes = classes


    def __len__(self):
        total = 0
        for list in self.list:
            total += len(list)
        return total

    def __getitem__(self, index):
        currentlist = 0
        while index >= len(self.list[currentlist]):
            index = index-len(self.list[currentlist])
            currentlist += 1

        currentlist = self.list[currentlist]
        
        data = currentlist.iloc[[index]].to_numpy()
        data = data[0][:len(data[0])-1]
        data = torch.tensor(data.astype(np.float))

        label = currentlist[" Label"][index]
        label = self.classes[label]
        label = torch.tensor(label)

        #add transformations if they exist
        if(self.transforms):
            data = self.transforms(data)
            #image = image/255.0

        #output labels are in single hot encoded vectors
        if self.isOneHot:
            return data, np.eye(len(self.classes))[label]
        else:
            return data, label


def leftOutMask(classes:int,batchsize, itemLeftOut:int):
    if classes<itemLeftOut:
        return torch.zeros((batchsize,classes))
    fullmask = torch.ones((batchsize,classes))
    fullmask[:,itemLeftOut] = 0
    return fullmask


class Network(nn.Module):
    def __init__(self, CLASSES):
        super().__init__()

        self.RL = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.fc1 = nn.Linear(78, 100)
        self.fc2 = nn.Linear(100,50)
        self.fc3 = nn.Linear(50,CLASSES)

        self.fc = nn.Linear(78,CLASSES)
        self.dropout = nn.Dropout(0.3)

        self.soft = nn.Softmax(dim=1)
        self.double()

    def forward(self, input):
        X = input.double()
        X = torch.flatten(X, start_dim=1)

        X = self.tanh(self.fc1(self.dropout(X)))
        X = self.RL(self.fc2(self.dropout(X)))
        return self.fc3(X)
