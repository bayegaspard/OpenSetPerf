import numpy as np
from LoadRandom import RndDataset
import torch
import torch.utils.data
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import OdinCodeByWetliu
import EnergyCodeByWetliu
import OpenMaxByMaXu
import pandas as pd

#Three lines from https://xxx-cook-book.gitbooks.io/python-cook-book/content/Import/import-from-parent-folder.html
import sys
root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)

#this seems really messy
from HelperFunctions.LoadPackets import NetworkDataset
from HelperFunctions.Evaluation import correctValCounter
from HelperFunctions.ModelLoader import Network

#to know it started
print("Hello, I hope you are having a nice day!")

#pick a device
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")

torch.manual_seed(0)
BATCH = 1000
CUTOFF = 0.85
NAME = "Combined"
ENERGYTRAINED = False
noise = 0.3
temprature = 3

#START IMAGE LOADING
#I looked up how to make a dataset, more information in the LoadImages file
#images are from: http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/
data_total = NetworkDataset(["MachineLearningCVE/Monday-WorkingHours.pcap_ISCX.csv","MachineLearningCVE/Tuesday-WorkingHours.pcap_ISCX.csv"])
unknown_data = NetworkDataset(["MachineLearningCVE/Wednesday-workingHours.pcap_ISCX.csv"])

CLASSES = len(data_total.classes)

random_data = RndDataset(CLASSES,transforms=transforms.Compose([transforms.Grayscale(1),transforms.Resize((100,100)), transforms.Normalize(0.8280,0.351)]))

data_train, data_test = torch.utils.data.random_split(data_total, [len(data_total)-1000,1000])

#create the dataloaders
training =  torch.utils.data.DataLoader(dataset=data_train, batch_size=BATCH, shuffle=True)
testing = torch.utils.data.DataLoader(dataset=data_test, batch_size=BATCH, shuffle=False)
unknowns = torch.utils.data.DataLoader(dataset=unknown_data, batch_size=BATCH, shuffle=False)
rands = torch.utils.data.DataLoader(dataset=random_data, batch_size=BATCH, shuffle=False)


#Loading non one hot data for OpenMax
data_total.isOneHot = False
data_train2, _ = torch.utils.data.random_split(data_total, [len(data_total)-1000,1000])
training2 = torch.utils.data.DataLoader(dataset=data_train2, batch_size=BATCH, shuffle=True)
#END IMAGE LOADING


model = Network(CLASSES).to(device)

soft = correctValCounter(CLASSES)
op = correctValCounter(CLASSES)
eng = correctValCounter(CLASSES, cutoff=5.5)
odin = correctValCounter(CLASSES)

if ENERGYTRAINED:
    chpt = "/checkpointE.pth"
else:
    chpt = "/checkpoint.pth"

if os.path.exists(NAME+chpt):
    model.load_state_dict(torch.load(NAME+chpt))

model.eval()

criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


plotter = torch.zeros((8,25))
plotter[0] += torch.tensor([x for x in range(25)])/2.5
plotter[1] += plotter[0]/10
plotter[2] += -plotter[0]
plotter[3] += plotter[0]/10

#these three lines somehow setup for the openmax thing
scoresOpen, mavs, distances = OpenMaxByMaXu.compute_train_score_and_mavs_and_dists(CLASSES,training2,device,model)
catagories = list(range(CLASSES))
weibullmodel = OpenMaxByMaXu.fit_weibull(mavs,distances,catagories,tailsize=10)


outArray = None
yArray = None
XArray = None

for batch,(X,y) in enumerate(testing):
    print(f"Batch: {batch}")
    X = X.to(device)
    y = y.to("cpu")



    _,output = model(X)
    output = output.to("cpu")
    
    #odin:
    odin.odinSetup(X,model,temprature,noise)
    
    #openmax
    op.setWeibull(weibullmodel)
    
    if outArray is None:
        outArray = output.detatch()
        yArray = y.detatch()
        XArray = X.detatch()
    else:
        outArray = torch.cat((outArray,output))
        yArray = torch.cat((yArray,y))
        XArray = torch.cat((XArray,X))


for a,b in enumerate(plotter[0]):
    soft.cutoff = b
    op.cutoff = plotter[2][a]
    eng.cutoff = plotter[1][a]
    odin.cutoff = plotter[3][a]

    #odin:
    odin.odinSetup(XArray,model,temprature,noise)

    soft.evalN(outArray,yArray)
    odin.evalN(outArray,yArray, type="Odin")
    eng.evalN(outArray, yArray, type="Energy")
    op.evalN(outArray, yArray, type="Open")

    known = soft.fScore()
    unknown = soft.fScoreUnknown()

    plotter[4][a] += 2*(known+unknown)/(known*unknown)
    plotter[6][a] += 2*(known+unknown)/(known*unknown)
    plotter[5][a] += 2*(known+unknown)/(known*unknown)
    plotter[7][a] += 2*(known+unknown)/(known*unknown)
    odin.zero()
    soft.zero()
    op.zero()
    eng.zero()
    optimizer.zero_grad()

    
    
    
    
    
#save data
df = pd.DataFrame(plotter.numpy(), columns=plotter[0].numpy(), index=["SoftVal", "OpenVal", "EnergyVal", "ODINVal", "Soft", "Open", "Energy", "ODIN"])
df = df.transpose()
df.to_csv("Scores for In distribution.csv")




#Everything past here is unknowns
#reset Plotter
plotter = torch.zeros((8,25))
plotter[0] += torch.tensor([x for x in range(25)])/2.5
plotter[1] += plotter[0]/10
plotter[2] += -plotter[0]
plotter[3] += plotter[0]/10





for batch,(X,y) in enumerate(unknowns):
    print(f"Batch: {batch}")
    X = X.to(device)
    y = y.to("cpu")



    _,output = model(X)
    output = output.to("cpu")
    
    #odin:
    odin.odinSetup(X,model,temprature,noise)
    
    #openmax
    op.setWeibull(weibullmodel)
    
    if outArray is None:
        outArray = output.detatch()
        yArray = y.detatch()
        XArray = X.detatch()
    else:
        outArray = torch.cat((outArray,output))
        yArray = torch.cat((yArray,y))
        XArray = torch.cat((XArray,X))


for a,b in enumerate(plotter[0]):
    soft.cutoff = b
    op.cutoff = plotter[2][a]
    eng.cutoff = plotter[1][a]
    odin.cutoff = plotter[3][a]

    #odin:
    odin.odinSetup(XArray,model,temprature,noise)

    soft.evalN(output,y, offset=0, indistribution=False)
    odin.evalN(output,y, offset=0, indistribution=False, type="Odin")
    op.evalN(output,y, offset=0, indistribution=False, type="Open")
    eng.evalN(output,y, offset=0, indistribution=False, type="Energy")

    known = soft.fScore()
    unknown = soft.fScoreUnknown()

    plotter[4][a] += 2*(known+unknown)/(known*unknown)
    plotter[6][a] += 2*(known+unknown)/(known*unknown)
    plotter[5][a] += 2*(known+unknown)/(known*unknown)
    plotter[7][a] += 2*(known+unknown)/(known*unknown)
    odin.zero()
    soft.zero()
    op.zero()
    eng.zero()
    optimizer.zero_grad()


#save data
df = pd.DataFrame(plotter.numpy(),index=["SoftVal", "OpenVal", "EnergyVal", "ODINVal", "Soft", "Open", "Energy", "ODIN"], columns=plotter[0].numpy())
df = df.transpose()
df.to_csv("Scores for Out distribution.csv")



