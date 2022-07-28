#https://github.com/wetliu/energy_ood <- associated paper
import numpy as np
from LoadPackets import NetworkDataset
import torch
import torch.utils.data
from torchvision import transforms
from ModelLoader import Network
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import Evaluation
import os
import EnergyCodeByWetliu
from LoadRandom import RndDataset


torch.manual_seed(0)
CLASSES = 36
BATCH = 500
NAME = "EnergyOOD"

#pick a device
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")

#I looked up how to make a dataset, more information in the LoadImages file
#images are from: http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/
data_total = NetworkDataset(["MachineLearningCVE/Monday-WorkingHours.pcap_ISCX.csv","MachineLearningCVE/Tuesday-WorkingHours.pcap_ISCX.csv"])
unknown_data = NetworkDataset(["MachineLearningCVE/Wednesday-workingHours.pcap_ISCX.csv"])

CLASSES = len(data_total.classes)

random_data = RndDataset(CLASSES,transforms=transforms.Compose([transforms.Grayscale(1),transforms.Resize((100,100)), transforms.Normalize(0.8280,0.351)]))

data_train, data_test = torch.utils.data.random_split(data_total, [len(data_total)-1000,1000])

training =  torch.utils.data.DataLoader(dataset=data_train, batch_size=BATCH, shuffle=True)
testing = torch.utils.data.DataLoader(dataset=data_test, batch_size=BATCH, shuffle=False)
unknowns = torch.utils.data.DataLoader(dataset=unknown_data, batch_size=72, shuffle=True)
rands = torch.utils.data.DataLoader(dataset=random_data, batch_size=BATCH, shuffle=False)


model = Network(CLASSES).to(device)


soft = Evaluation.correctValCounter(CLASSES,cutoff=0.005, confusionMat=True)
Eng = Evaluation.correctValCounter(CLASSES, cutoff=0.5, confusionMat=True)

if os.path.exists(NAME+"/checkpointR.pth"):
    model.load_state_dict(torch.load(NAME+"/checkpointR.pth"))
    print("Loaded model checkpoint")

epochs = 10
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


for e in range(epochs):
    lost_amount = 0
    out_set = iter(rands)
    for batch, (in_set) in enumerate(training):
        X,y = in_set
        X = (X).to(device)
        y = y.to(device)

        output = model(torch.cat((X,out_set.next()[0]),0))
        out_set_out = output[len(X):]
        output = output
        lost_points = criterion(output[:len(X)], y)
        EnergyCodeByWetliu.energyLossMod(lost_points,output,in_set)


        optimizer.zero_grad()
        lost_points.backward()

        optimizer.step()

        lost_amount += lost_points.item()


        soft.cutoffStorage(output[:len(X)].detach(), "Soft")
        Eng.cutoffStorage(output[:len(X)].detach(), "Energy")

    soft.autocutoff()
    Eng.autocutoff()

    with torch.no_grad():
        model.eval()
        for batch,(X,y) in enumerate(testing):
            X = X.to(device)
            y = y.to("cpu")

            output = model(X).to("cpu")


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
            torch.save(model.state_dict(), NAME+"/checkpointR.pth")

        model.train()
    scheduler.step()



#Everything past here is to do with unknowns

with torch.no_grad():
        model.eval()
        for batch,(X,y) in enumerate(unknowns):
            X = (X).to(device)
            y = y.to("cpu")

            output = model(X).to("cpu")

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