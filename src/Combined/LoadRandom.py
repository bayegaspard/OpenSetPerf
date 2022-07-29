#note, this is a very modified version of a dataloader found in https://www.youtube.com/watch?v=ZoZHd0Zm3RY
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import torch.nn.functional as F

class RndDataset(Dataset):
    def __init__(self,classes,isonehot=True):
        self.isOneHot = isonehot
        self.classes=classes


    def __len__(self):
        return 1000000

    def __getitem__(self, index):
        #hardcoding in the path is probibly not the best
        #image = Image.open("English/Img/"+self.list.iat[index, 0]+".png")
        #no it was not
        data = torch.rand(78)

        label = torch.tensor(self.classes)

        #output labels are in single hot encoded vectors
        if self.isOneHot:
            return data, F.one_hot(label,self.classes+1)[:self.classes]
        else:
            return data, label

    
