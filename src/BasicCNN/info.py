import LoadPackets
import torch

data_total = LoadPackets.NetworkDataset(["MachineLearningCVE/Monday-WorkingHours.pcap_ISCX.csv",
"MachineLearningCVE/Tuesday-WorkingHours.pcap_ISCX.csv",
"MachineLearningCVE/Wednesday-workingHours.pcap_ISCX.csv",
"MachineLearningCVE/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
"MachineLearningCVE/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
"MachineLearningCVE/Friday-WorkingHours-Morning.pcap_ISCX.csv",
"MachineLearningCVE/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
"MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"])

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
print("Done")