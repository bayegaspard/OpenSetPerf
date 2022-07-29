import numpy as np
from LoadRandom import RndDataset
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import EnergyCodeByWetliu
import OpenMaxByMaXu
import pandas as pd
import glob

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

BATCH = 100
CUTOFF = 0.85
NAME = os.path.basename(os.path.dirname(__file__))
ENERGYTRAINED = False
AUTOCUTOFF = True
noise = 0.3
temprature = 9

#START IMAGE LOADING
#I looked up how to make a dataset, more information in the LoadImages file

path_to_dataset = "datasets" #put the absolute path to your dataset , type "pwd" within your dataset folder from your teminal to know this path.

def getListOfCSV(path):
    return glob.glob(path+"/*.csv")

data_total = NetworkDataset(getListOfCSV(path_to_dataset),benign=True)
unknown_data = NetworkDataset(getListOfCSV(path_to_dataset),benign=False)

CLASSES = len(data_total.classes)
CLASSES = 1

random_data = RndDataset(CLASSES)

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

if os.path.exists("src/"+NAME+chpt):
    model.load_state_dict(torch.load("src/"+NAME+chpt))

epochs = 10
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


plotter = torch.zeros((8,25))
plotter[0] += torch.tensor([x for x in range(25)])/2.5
plotter[1] += plotter[0]/10
plotter[2] += -plotter[0]
plotter[3] += plotter[0]/10

#making the items that appear less frequently have a higher penalty so that they are not missed.
# magnification = torch.tensor([1.0000e+00, 2.8636e+02, 3.8547e+02, 3.9218e+02, 4.1337e+02, 9.8373e+00,
#         2.2084e+02, 2.0665e+05, 1.5084e+03, 3.4863e+03, 1.0824e+05, 6.3142e+04,
#         1.1562e+03, 1.4303e+01, 1.7754e+01])[:CLASSES]

for e in range(epochs):
    lost_amount = 0
    out_set = iter(rands)
    for batch, (X, y) in enumerate(training):
        X = (X).to(device)
        y = y.to(device)


        if ENERGYTRAINED:
            test = torch.cat((X,out_set.next()[0]),0)
            _, outputE = model(test)
            output = outputE[:len(X)]
        else:
            _, output = model(X)

        lost_points = criterion(output, torch.nn.functional.one_hot(y,CLASSES).to(torch.float))
        optimizer.zero_grad()
        if ENERGYTRAINED:
            EnergyCodeByWetliu.energyLossMod(lost_points,outputE,(X,y))

        lost_points.backward()

        optimizer.step()
        optimizer.zero_grad()

        lost_amount += lost_points.item()

        

    model.eval()

    

    #these three lines somehow setup for the openmax thing
    scoresOpen, mavs, distances = OpenMaxByMaXu.compute_train_score_and_mavs_and_dists(CLASSES,training2,device,model)
    catagories = list(range(CLASSES))
    weibullmodel = OpenMaxByMaXu.fit_weibull(mavs,distances,catagories,tailsize=3)

    #make a call about where the cutoff is
    if AUTOCUTOFF:
        for batch, (X, y) in enumerate(training):

            #odin:
            odin.odinSetup(X,model,temprature,noise)
        
            #openmax
            op.setWeibull(weibullmodel)

            _, output = model(X)

            soft.cutoffStorage(output.detach(), "Soft")
            eng.cutoffStorage(output.detach(), "Energy")
            op.cutoffStorage(output.detach(), "Open")
            odin.cutoffStorage(output.detach(), "Odin")
        soft.autocutoff(0.73)
        eng.autocutoff(0.76)    
        op.autocutoff(0.85)
        odin.autocutoff(0.67)


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
        
        #Plotting
        if e == epochs-1:
            #things that don't need to be recalculated for plotting
            outmax = output.max(dim=1)[0]
            energy = []
            EnergyCodeByWetliu.energyScoreCalc(energy,output)
            energy = torch.tensor(np.array(energy))
            openoutmax = op.openMaxMod(output).max(dim=1)
            odinoutmax =odin.odinMod(output).max(dim=1)[0]

            for a,b in enumerate(plotter[0]):
                plotter[4][a] += outmax.greater_equal(b).sum()/1000
                plotter[6][a] += energy.less_equal(plotter[2][a]).sum()/1000
                plotter[5][a] += (openoutmax[0].greater_equal(plotter[1][a])*(openoutmax[1]!=CLASSES)).sum()/1000
                plotter[7][a] += odinoutmax.greater_equal(plotter[3][a]).sum()/1000

    
        soft.evalN(output,y)
        odin.evalN(output,y, type="Odin")
        eng.evalN(output, y, type="Energy")
        op.evalN(output, y, type="Open")
        optimizer.zero_grad()
        
    
    
    print(f"-----------------------------Epoc: {e+1}-----------------------------")
    print(f"lost: {100*lost_amount/len(data_train)}")
    print("SoftMax:")
    soft.PrintEval()
    print("OpenMax:")
    op.PrintEval()
    print("Energy:")
    eng.PrintEval()
    print("ODIN:")
    odin.PrintEval()

    #save data
    df = pd.DataFrame(plotter.numpy(), columns=plotter[0].numpy(), index=["SoftVal", "OpenVal", "EnergyVal", "ODINVal", "Soft", "Open", "Energy", "ODIN"])
    df = df.transpose()
    df.to_csv("Scores for In distribution.csv")

    odin.zero()
    soft.zero()
    op.zero()
    eng.zero()
    
    if e%5 == 4:
        torch.save(model.state_dict(), "src/"+NAME+chpt)

    model.train()
    scheduler.step()


#Everything past here is unknowns
#reset Plotter
plotter = torch.zeros((8,25))
plotter[0] += torch.tensor([x for x in range(25)])/2.5
plotter[1] += plotter[0]/10
plotter[2] += plotter[0]
plotter[3] += plotter[0]/10


model.eval()




#these three lines somehow setup for the openmax thing
scoresOpen, mavs, distances = OpenMaxByMaXu.compute_train_score_and_mavs_and_dists(CLASSES,training2,device,model)
catagories = list(range(CLASSES))
weibullmodel = OpenMaxByMaXu.fit_weibull(mavs,distances,catagories,tailsize=10)


for batch,(X,y) in enumerate(unknowns):
    print(f"Batch: {batch}")
    optimizer.zero_grad()
    X = (X).to(device)
    y = y.to("cpu")



    _, output = model(X)
    output = output.to("cpu")

    #odin:
    odin.odinSetup(X,model,temprature,noise)
    
    #openmax
    op.setWeibull(weibullmodel)

    #things that don't need to be recalculated for plotting
    outmax = output.max(dim=1)[0]
    energy = []
    EnergyCodeByWetliu.energyScoreCalc(energy,output)
    energy = torch.tensor(np.array(energy))
    openoutmax = op.openMaxMod(output).max(dim=1)
    odinoutmax =odin.odinMod(output).max(dim=1)[0]

    #Plotting
    for a,b in enumerate(plotter[0]):
        plotter[4][a] += outmax.greater_equal(b).sum()/26416
        plotter[6][a] += energy.less_equal(plotter[2][a]).sum()/26416
        plotter[5][a] += (openoutmax[0].greater_equal(plotter[1][a])*(openoutmax[1]!=CLASSES)).sum()/26416
        plotter[7][a] += odinoutmax.greater_equal(plotter[3][a]).sum()/26416

    soft.evalN(output,y, indistribution=False)
    odin.evalN(output,y, indistribution=False, type="Odin")
    op.evalN(output,y, indistribution=False, type="Open")
    eng.evalN(output,y, indistribution=False, type="Energy")
    optimizer.zero_grad()

print("SoftMax:")
soft.PrintEval()
print("OpenMax:")
op.PrintEval()
print("Energy:")
eng.PrintEval()
print("ODIN:")
odin.PrintEval()


#save data
df = pd.DataFrame(plotter.numpy(),index=["SoftVal", "OpenVal", "EnergyVal", "ODINVal", "Soft", "Open", "Energy", "ODIN"], columns=plotter[0].numpy())
df = df.transpose()
df.to_csv("Scores for Out distribution.csv")


odin.zero()
soft.zero()
op.zero()
eng.zero()
model.train()


rands = torch.utils.data.DataLoader(dataset=random_data, batch_size=500, shuffle=False)
it = iter(rands)
for batch in range(50):
    (X,y) = it.next()
    print(f"Batch: {batch}")
    optimizer.zero_grad()
    X = (X).to(device)
    y = y.to("cpu")



    _, output = model(X)
    output = output.to("cpu")

    #odin:
    odin.odinSetup(X,model,temprature,noise)
    
    #openmax
    op.setWeibull(weibullmodel)
    soft.evalN(output,y)
    odin.evalN(output,y, type="Odin")
    op.evalN(output,y, type="Open")
    eng.evalN(output,y, type="Energy")
    optimizer.zero_grad()


print("SoftMax:")
soft.PrintEval()
print("OpenMax:")
op.PrintEval()
print("Energy:")
eng.PrintEval()
print("ODIN:")
odin.PrintEval()


odin.zero()
soft.zero()
op.zero()
eng.zero()
model.train()

