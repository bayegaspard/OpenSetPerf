import torch
import glob

#four lines from https://xxx-cook-book.gitbooks.io/python-cook-book/content/Import/import-from-parent-folder.html
import sys
import os
root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)

#this seems really messy
from HelperFunctions import LoadPackets


path_to_dataset = "datasets" #put the absolute path to your dataset , type "pwd" within your dataset folder from your teminal to know this path.

def getListOfCSV(path):
    return glob.glob(path+"/*.csv")

data_total = LoadPackets.NetworkDataset(getListOfCSV(path_to_dataset), ignore=[14])

CLASSES = len(data_total.classes)
print(CLASSES)

data_total.isOneHot = False

training =  torch.utils.data.DataLoader(dataset=data_total, batch_size=100000)

sumOfY = torch.zeros(len(data_total.classes))
sumofNanRows = 0
nanColumns = torch.zeros(78)
for X,y in training:
    sumOfY += torch.bincount(y,minlength=CLASSES)
    nanMap = X.isnan()
    #if a row has at least 1 nan it adds one to the column
    sumofNanRows += nanMap.sum(dim=1).greater(0).sum().item()
    #Rotates the array and then totals up the nans
    nanColumns +=nanMap.swapdims(0,1).sum(dim=1)


print(f"total number of classes {sumOfY}")
print(f"Count of each class {data_total.classes}")
print(f"Count rows with nan {sumofNanRows}")
print(f"Count nan in each column {nanColumns}")
print(f"Weights to be applied: {(sumOfY.sum()/len(sumOfY))/sumOfY}")
print("Done")