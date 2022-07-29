import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import EvaluationDisplay
import os
import OdinCodeByWetliu
import glob

#three lines from https://xxx-cook-book.gitbooks.io/python-cook-book/content/Import/import-from-parent-folder.html
import sys
root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)

#this seems really messy
from HelperFunctions.LoadPackets import NetworkDataset
from HelperFunctions.Evaluation import correctValCounter
from HelperFunctions.ModelLoader import Network


#pick a device
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")

torch.manual_seed(0)
CLASSES = 36
BATCH = 100
CUTOFF = 0.85
NAME = "ODIN"
noise = 0.3
temprature = 12

#I looked up how to make a dataset, more information in the LoadImages file
path_to_dataset = "datasets" #put the absolute path to your dataset , type "pwd" within your dataset folder from your teminal to know this path.

def getListOfCSV(path):
    return glob.glob(path+"/*.csv")

data_total = NetworkDataset(getListOfCSV(path_to_dataset),benign=True)
unknown_data = NetworkDataset(getListOfCSV(path_to_dataset),benign=False)

CLASSES = len(data_total.classes)

data_train, data_test = torch.utils.data.random_split(data_total, [len(data_total)-1000,1000])

training =  torch.utils.data.DataLoader(dataset=data_train, batch_size=BATCH, shuffle=True)
testing = torch.utils.data.DataLoader(dataset=data_test, batch_size=BATCH, shuffle=False)
unknowns = torch.utils.data.DataLoader(dataset=unknown_data, batch_size=BATCH, shuffle=False)


model = Network(CLASSES).to(device)
soft = correctValCounter(CLASSES)
odin = EvaluationDisplay.correctValCounter(CLASSES,cutoff=0.5)

if os.path.exists("ODIN/checkpoint.pth"):
    model.load_state_dict(torch.load("ODIN/checkpoint.pth"))

epochs = 1
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

#For graphing
plotter = torch.zeros((2,15))
plotter[1] += ((torch.tensor([x+1 for x in range(15)])/5))
fig, ax = plt.subplots()

for e in range(epochs):
    lost_amount = 0

    for batch, (X, y) in enumerate(training):

        X = (X).to(device)
        y = y.to(device)

        output = model(X)
        lost_points = criterion(output, y)
        optimizer.zero_grad()
        lost_points.backward()

        #printing paramiters to check if they are moving
        #for para in model.parameters():
            #print(para.grad)

        optimizer.step()
        optimizer.zero_grad()

        lost_amount += lost_points.item()

    for a,b in enumerate(plotter[1]):
        scores = OdinCodeByWetliu.get_ood_scores_odin(testing,model,BATCH,0,T=temprature,noise=b,in_dist=True)
        plotter[0][a] += scores[0].sum()

    scores = OdinCodeByWetliu.get_ood_scores_odin(testing,model,BATCH,0,T=temprature,noise=noise,in_dist=True)
    
    model.eval()
    for batch,(X,y) in enumerate(testing):
        X = X.to(device)
        y = y.to("cpu")

        output = model(X).to("cpu")
        if (batch+1)*BATCH <= len(scores[0]):
            odin.odinUnknownSave(scores[0][BATCH*batch:BATCH*(batch+1)])
            #odin.cutoffPlotVals(scores[0][BATCH*batch:BATCH*(batch+1)])
        else:
            odin.odinUnknownSave(scores[0][BATCH*batch:])
            #odin.cutoffPlotVals(scores[0][BATCH*batch:])
        soft.evalN(output.detach(),y)
        odin.evalN(output.detach(),y, type="Odin")
        
    
    
    print(f"-----------------------------Epoc: {e+1}-----------------------------")
    print(f"lost: {100*lost_amount/len(data_train)}")
    soft.PrintEval()
    odin.PrintEval()

    plotter[0] = plotter[0]/1000
    ax.plot(plotter[1],plotter[0], "r-", label="In Distribution")
    store = plotter[0]

    odin.zero()
    soft.zero()
    
    if e%5 == 4:
        torch.save(model.state_dict(), "ODIN/checkpoint.pth")

    model.train()
    scheduler.step()


#Everything past here is unknowns
#For graphing
plotter = torch.zeros((2,15))
plotter[1] += ((torch.tensor([x+1 for x in range(15)])/5))

model.eval()

for a,b in enumerate(plotter[1]):
    scores = OdinCodeByWetliu.get_ood_scores_odin(unknowns,model,BATCH,BATCH*len(unknowns),T=temprature,noise=b,in_dist=False)
    plotter[0][a] += scores.sum()

scores = OdinCodeByWetliu.get_ood_scores_odin(unknowns,model,BATCH,BATCH*len(unknowns),T=temprature,noise=noise,in_dist=False)
for batch,(X,y) in enumerate(unknowns):
    X = (X).to(device)
    y = y.to("cpu")

    output = model(X).to("cpu")

    if (batch+1)*BATCH < len(scores):
        odin.odinUnknownSave(scores[BATCH*batch:BATCH*(batch+1)])
        #odin.cutoffPlotVals(scores[BATCH*batch:BATCH*(batch+1)])
    else:
        odin.odinUnknownSave(scores[BATCH*batch:])
        #odin.cutoffPlotVals(scores[BATCH*batch:])
    

    soft.evalN(output,y, offset=26)
    odin.evalN(output,y, offset=26, type="Odin")
    

soft.PrintUnknownEval()
odin.PrintUnknownEval()

plotter[0] = plotter[0]/odin.count
ax.plot(plotter[1],plotter[0], "b-", label="Out of Distribution")
ax.plot(plotter[1],(plotter[0]-store)**2, "g-", label="Diffrence squared")

ax.set_ylabel("ODIN probabilities")
ax.set_xlabel("Noise")
ax.set_title("ODIN probabilities vs noise")
ax.legend()
plt.show()

soft.zero()
odin.zero()

model.train()