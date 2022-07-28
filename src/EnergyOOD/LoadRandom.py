#note, this is a very modified version of a dataloader found in https://www.youtube.com/watch?v=ZoZHd0Zm3RY
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import torch.nn.functional as F

class RndDataset(Dataset):
    def __init__(self,classes,transforms=None,isonehot=True):
        self.transforms = transforms
        self.isOneHot = isonehot
        self.classes=classes


    def __len__(self):
        return 1000000

    def __getitem__(self, index):
        #hardcoding in the path is probibly not the best
        #image = Image.open("English/Img/"+self.list.iat[index, 0]+".png")
        #no it was not
        image = torch.rand(3,150,150)

        label = torch.tensor(self.classes)

        #add transformations if they exist
        if(self.transforms):
            image = self.transforms(image)
            #image = image/255.0

        #output labels are in single hot encoded vectors
        if self.isOneHot:
            return image, F.one_hot(label,self.classes+1)[:self.classes]
        else:
            return image, label

    
