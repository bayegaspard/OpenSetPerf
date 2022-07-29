#https://github.com/wetliu/energy_ood <- associated paper
import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import EvaluationDisplay
import os
import EnergyCodeByWetliu
import glob

#four lines from https://xxx-cook-book.gitbooks.io/python-cook-book/content/Import/import-from-parent-folder.html
import os
import sys
root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)

#this seems really messy
from HelperFunctions.LoadPackets import NetworkDataset
from HelperFunctions.ModelLoader import Network

torch.manual_seed(0)
BATCH = 500
NAME = "EnergyOOD"

#pick a device
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")

#I looked up how to make a dataset, more information in the LoadImages file

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


model = Network().to(device)



soft = EvaluationDisplay.correctValCounter(CLASSES,cutoff=0.96, confusionMat=True)
Eng = EvaluationDisplay.correctValCounter(CLASSES, cutoff=3.65, confusionMat=True)

if os.path.exists(NAME+"/checkpoint.pth"):
    model.load_state_dict(torch.load(NAME+"/checkpoint.pth"))
    print("Loaded model checkpoint")

epochs = 1
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)


with torch.no_grad():
    model.eval()
    for batch,(X,y) in enumerate(testing):
        X = X.to(device)
        y = y.to("cpu")

        output = model(X).to("cpu")


        soft.evalN(output,y)
        Eng.evalN(output, y, type="Energy")
        
        Eng.cutoffPlotVals(Eng.energyMod(output))
        soft.cutoffStorage(output[:len(X)].detach(), "Soft")
        Eng.cutoffStorage(output[:len(X)].detach(), "Energy")

    # soft.autocutoff(0.25)
    # Eng.autocutoff(0.25)

        

    print("-----------------------------from savepoint-----------------------------")
    print("SoftMax:")
    soft.PrintEval()
    print("\nEnergy Based OOD:")
    Eng.PrintEval()

    Eng.plotting[0] = Eng.plotting[0]/1000
    plt.plot(Eng.plotting[1],Eng.plotting[0], "r-", label="In Distribution")
    store = Eng.plotting[0]

    soft.zero()
    Eng.zero()
    



#Everything past here is to do with unknowns


    for batch,(X,y) in enumerate(unknowns):
        X = (X).to(device)
        y = y.to("cpu")

        output = model(X).to("cpu")

        soft.evalN(output,y, offset=26)
        Eng.evalN(output, y, offset=26, type="Energy")
        Eng.cutoffPlotVals(Eng.energyMod(output))
        
    print("SoftMax:")
    soft.PrintUnknownEval()
    print("\nEnergy Based OOD:")
    Eng.PrintUnknownEval()
    
    Eng.plotting[0] = Eng.plotting[0]/26416
    plt.plot(Eng.plotting[1],Eng.plotting[0], "b-", label="Out of Distribution")
    plt.plot(Eng.plotting[1],(Eng.plotting[0]-store)**2, "g-", label="Diffrence^2")
    plt.legend()
    plt.show()

    soft.zero()
    Eng.zero()
