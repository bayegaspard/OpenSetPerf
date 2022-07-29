#---------------------------------------------Imports------------------------------------------
import numpy as np
import glob
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

#three lines from https://xxx-cook-book.gitbooks.io/python-cook-book/content/Import/import-from-parent-folder.html
import sys
root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))#'/Users/abroggi/Desktop/REU/Ubuntu/Github.nosync/OpenSet-Recognition-for-NIDS/src/HelperFunctions/__init__.py'
sys.path.append(root_folder)

#this seems really messy
from HelperFunctions.LoadPackets import NetworkDataset
from HelperFunctions.Evaluation import correctValCounter
from HelperFunctions.ModelLoader import Network



#pick a device
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")

#------------------------------------------------------------------------------------------------------

#---------------------------------------------Hyperparameters------------------------------------------
torch.manual_seed(0)
BATCH = 100
CUTOFF = 0.85
epochs = 10
checkpoint = "/checkpoint.pth"
#------------------------------------------------------------------------------------------------------

#---------------------------------------------Model/data set up----------------------------------------

NAME = "src/"+os.path.basename(os.path.dirname(__file__))


path_to_dataset = "datasets" #put the absolute path to your dataset , type "pwd" within your dataset folder from your teminal to know this path.

def getListOfCSV(path):
    return glob.glob(path+"/*.csv")

data_total = NetworkDataset(getListOfCSV(path_to_dataset))
unknown_data = NetworkDataset(getListOfCSV(path_to_dataset))

CLASSES = len(data_total.classes)

data_train, data_test = torch.utils.data.random_split(data_total, [len(data_total)-1000,1000])

training =  torch.utils.data.DataLoader(dataset=data_train, batch_size=BATCH, shuffle=True)
testing = torch.utils.data.DataLoader(dataset=data_test, batch_size=BATCH, shuffle=False)
unknowns = torch.utils.data.DataLoader(dataset=unknown_data, batch_size=BATCH, shuffle=False)


model = Network(CLASSES).to(device)
evaluative = correctValCounter(CLASSES,cutoff=CUTOFF,confusionMat=True)

if os.path.exists(NAME+checkpoint):
    model.load_state_dict(torch.load(NAME+checkpoint))

criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

#------------------------------------------------------------------------------------------------------

#---------------------------------------------Training-------------------------------------------------

for e in range(epochs):
    lost_amount = 0

    for batch, (X, y) in enumerate(training):
        X = X.to(device)
        y = y.to(device)

        _, output = model(X)
        lost_points = criterion(output, y)
        optimizer.zero_grad()
        lost_points.backward()


        optimizer.step()

        lost_amount += lost_points.item()

    
    #--------------------------------------------------------------------------------

    #--------------------------------------Testing-----------------------------------

    with torch.no_grad():
        model.eval()
        for batch,(X,y) in enumerate(testing):
            X = X.to(device)
            y = y.to("cpu")

            _, output = model(X).to("cpu")
            evaluative.evalN(output,y)
            

        print(f"-----------------------------Epoc: {e+1}-----------------------------")
        print(f"lost: {100*lost_amount/len(data_train)}")
        evaluative.PrintEval()
        evaluative.zero()
        
        if e%5 == 4:
            torch.save(model.state_dict(), NAME+checkpoint)

        model.train()
    scheduler.step()


#------------------------------------------------------------------------------------------------------

#---------------------------------------------Unknowns-------------------------------------------------

with torch.no_grad():
        model.eval()
        for batch,(X,y) in enumerate(unknowns):
            X = X.to(device)
            y = y.to("cpu")

            output = model(X).to("cpu")
            evaluative.evalN(output,y, indistribution=False)

        evaluative.PrintUnknownEval()
        evaluative.storeConfusion("CONFUSION.CSV")
        evaluative.zero()
        
        model.train()