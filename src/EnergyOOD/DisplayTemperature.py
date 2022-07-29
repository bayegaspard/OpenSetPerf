#https://github.com/wetliu/energy_ood <- associated paper
import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import glob


#three lines from https://xxx-cook-book.gitbooks.io/python-cook-book/content/Import/import-from-parent-folder.html
import sys
root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)

#this seems really messy
from HelperFunctions.LoadPackets import NetworkDataset
from HelperFunctions.ModelLoader import Network
from HelperFunctions.EvaluationDisplay import EvaluationWithPlots as correctValCounter
import CodeFromImplementations.EnergyCodeByWetliu as EnergyCodeByWetliu

torch.manual_seed(0)
CLASSES = 36
BATCH = 500
NAME = os.path.basename(os.path.dirname(__file__))

#pick a device
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")

#I looked up how to make a dataset, more information in the LoadImages file
#images are from: http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/

path_to_dataset = "datasets" #put the absolute path to your dataset , type "pwd" within your dataset folder from your teminal to know this path.

def getListOfCSV(path):
    return glob.glob(path+"/*.csv")

data_total = NetworkDataset(getListOfCSV(path_to_dataset),benign=True)
unknown_data = NetworkDataset(getListOfCSV(path_to_dataset),benign=False)

CLASSES = len(data_total.classes)

data_train, data_test = torch.utils.data.random_split(data_total, [35576,1000])

training =  torch.utils.data.DataLoader(dataset=data_train, batch_size=BATCH, shuffle=True)
testing = torch.utils.data.DataLoader(dataset=data_test, batch_size=BATCH, shuffle=False)
unknowns = torch.utils.data.DataLoader(dataset=unknown_data, batch_size=72, shuffle=True)

fig, ax = plt.subplots()
ax.set_ylabel("Mean Score")
ax.set_xlabel("Temperature")
ax.set_title("Score Vs Temperature")

model = Network().to(device)


soft = correctValCounter(CLASSES,cutoff=0.96)
Eng = correctValCounter(CLASSES, cutoff=3.65)

if os.path.exists(NAME+"/src/checkpoint.pth"):
    model.load_state_dict(torch.load(NAME+"/src/checkpoint.pth"))
    print("Loaded model checkpoint")


criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)


plotting = torch.zeros((2,25))
plotting[0] += (torch.tensor([x+1 for x in range(25)])/12.5)

with torch.no_grad():
    model.eval()

    for batch,(X,y) in enumerate(testing):
        X = X.to(device)
        y = y.to("cpu")

        output = model(X).to("cpu")

        soft.cutoffStorage(output[:len(X)].detach(), "Soft")
        Eng.cutoffStorage(output[:len(X)].detach(), "Energy")

    # soft.autocutoff(0.85)
    # Eng.autocutoff(0.85)


    for batch,(X,y) in enumerate(testing):
        X = X.to(device)
        y = y.to("cpu")

        output = model(X).to("cpu")


        soft.evalN(output,y)
        Eng.evalN(output, y, type="Energy")
        
        for a,b in enumerate(plotting[0]):
            EnergyCodeByWetliu.setTemp(b)
            scores = []
            EnergyCodeByWetliu.energyScoreCalc(scores,output)
            plotting[1][a] += np.array(scores).sum()

        

    print("-----------------------------from savepoint-----------------------------")
    print("SoftMax:")
    soft.PrintEval()
    print("\nEnergy Based OOD:")
    Eng.PrintEval()

    Eng.plotting[0] = Eng.plotting[0]
    ax.plot(plotting[0],plotting[1]/1000, "r-", label="In Distribution")

    soft.zero()
    Eng.zero()
    

    plotting = torch.zeros((2,25))
    plotting[0] += (torch.tensor([x+1 for x in range(25)])/12.5)

#Everything past here is to do with unknowns


    for batch,(X,y) in enumerate(unknowns):
        X = (X).to(device)
        y = y.to("cpu")

        output = model(X).to("cpu")

        soft.evalN(output,y, offset=26)
        Eng.evalN(output, y, offset=26, type="Energy")

        for a,b in enumerate(plotting[0]):
            EnergyCodeByWetliu.setTemp(b)
            scores = []
            EnergyCodeByWetliu.energyScoreCalc(scores,output)
            plotting[1][a] += np.array(scores).sum()
        
    print("SoftMax:")
    soft.PrintUnknownEval()
    print("\nEnergy Based OOD:")
    Eng.PrintUnknownEval()
    
    ax.plot(plotting[0],plotting[1]/26416, "b-", label="Out of Distribution")
    ax.legend()
    plt.show()

    soft.zero()
    Eng.zero()
