#https://github.com/wetliu/energy_ood <- associated paper

#---------------------------------------------Imports------------------------------------------
import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from LoadRandom import RndDataset
import glob

#three lines from https://xxx-cook-book.gitbooks.io/python-cook-book/content/Import/import-from-parent-folder.html
import sys
root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)

#this seems really messy
from HelperFunctions.LoadPackets import NetworkDataset
from HelperFunctions.Evaluation import correctValCounter
from HelperFunctions.ModelLoader import Network
import CodeFromImplementations.EnergyCodeByWetliu as EnergyCodeByWetliu

#------------------------------------------------------------------------------------------------------

#---------------------------------------------Hyperparameters------------------------------------------
torch.manual_seed(0)
BATCH = 500
CUTOFF = 0.9999999
epochs = 10
temperature = 0.001
checkpoint = "checkpoint.pth"
#------------------------------------------------------------------------------------------------------

#---------------------------------------------Model/data set up----------------------------------------
EnergyCodeByWetliu.setTemp(temperature) #I dont think this works and needs to be fixed


NAME = os.path.basename(os.path.dirname(__file__))

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

random_data = RndDataset(CLASSES)

data_train, data_test = torch.utils.data.random_split(data_total, [len(data_total)-1000,1000])

training =  torch.utils.data.DataLoader(dataset=data_train, batch_size=BATCH, shuffle=True)
testing = torch.utils.data.DataLoader(dataset=data_test, batch_size=BATCH, shuffle=False)
unknowns = torch.utils.data.DataLoader(dataset=unknown_data, batch_size=72, shuffle=True)
rands = torch.utils.data.DataLoader(dataset=random_data, batch_size=BATCH, shuffle=False)


model = Network(CLASSES).to(device)


soft = correctValCounter(CLASSES,cutoff=CUTOFF, confusionMat=True)
Eng = correctValCounter(CLASSES, cutoff=CUTOFF, confusionMat=True)

if os.path.exists(NAME+"/src/"+checkpoint):
    model.load_state_dict(torch.load(NAME+"/src/"+checkpoint))
    print("Loaded model checkpoint")


criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

#------------------------------------------------------------------------------------------------------

#---------------------------------------------Training-------------------------------------------------
for e in range(epochs):
    lost_amount = 0
    out_set = iter(rands)
    for batch, (in_set) in enumerate(training):
        X,y = in_set
        X = (X).to(device)
        y = y.to(device)

        _, output = model(torch.cat((X,out_set.next()[0]),0))
        out_set_out = output[len(X):]
        output = output
        lost_points = criterion(output[:len(X)], y)
        EnergyCodeByWetliu.energyLossMod(lost_points,output,in_set)


        optimizer.zero_grad()
        lost_points.backward()

        optimizer.step()

        lost_amount += lost_points.item()


        # soft.cutoffStorage(output[:len(X)].detach(), "Soft")
        # Eng.cutoffStorage(output[:len(X)].detach(), "Energy")

    # soft.autocutoff()
    # Eng.autocutoff()

    #--------------------------------------------------------------------------------

    #--------------------------------------Testing-----------------------------------

    with torch.no_grad():
        model.eval()
        for batch,(X,y) in enumerate(testing):
            X = X.to(device)
            y = y.to("cpu")

            _, output = model(X)
            output = output.to("cpu")


            soft.evalN(output,y)
            Eng.evalN(output, y, type="Energy")

            

        print(f"-----------------------------Epoc: {e+1}-----------------------------")
        print(f"lost: {100*lost_amount/len(data_train)}")
        print("SoftMax:")
        soft.PrintEval()
        print("\nEnergy Based OOD:")
        Eng.PrintEval()

        soft.storeConfusion("CONFUSIONSOFT.CSV")
        soft.zero()
        Eng.storeConfusion("CONFUSION.CSV")
        Eng.zero()
        
        if e%5 == 4:
            torch.save(model.state_dict(), NAME+"/src/"+checkpoint)

        model.train()
    scheduler.step()


#------------------------------------------------------------------------------------------------------

#---------------------------------------------Unknowns-------------------------------------------------


with torch.no_grad():
        model.eval()
        for batch,(X,y) in enumerate(unknowns):
            X = (X).to(device)
            y = y.to("cpu")

            _, output = model(X)
            output = output.to("cpu")

            soft.evalN(output,y, offset=26)
            Eng.evalN(output, y, offset=26, type="Energy")
            
        print("SoftMax:")
        soft.PrintUnknownEval()
        print("\nEnergy Based OOD:")
        Eng.PrintUnknownEval()
        
        soft.storeConfusion("CONFUSIONUSOFT.CSV")
        Eng.storeConfusion("CONFUSIONU.CSV")
        soft.zero()
        Eng.zero()

        model.train()