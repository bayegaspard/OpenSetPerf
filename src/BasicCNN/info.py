import LoadPackets
import torch
import glob



path_to_dataset = "/home/designa/OpenSet-Recognition-for-NIDS/datasets" #put the absolute path to your dataset , type "pwd" within your dataset folder from your teminal to know this path.

def getListOfCSV(path):
    return glob.glob(path+"/*.csv")

data_total = LoadPackets.NetworkDataset(getListOfCSV(path_to_dataset))

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
print(f"Weights to be applied: {(len(data_total)/sumOfY)/data_total.classes}")
print("Done")