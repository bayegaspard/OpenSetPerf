import numpy as np
from LoadPackets import NetworkDataset
import torch
import torch.utils.data
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import Evaluation
import os
from ModelLoader import Network


#pick a device
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")

torch.manual_seed(0)

BATCH = 100
CUTOFF = 0.85
NAME = "BasicUnknowns"

#I looked up how to make a dataset, more information in the LoadImages file
#images are from: http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/
data_total = NetworkDataset(["MachineLearningCVE/Monday-WorkingHours.pcap_ISCX.csv","MachineLearningCVE/Tuesday-WorkingHours.pcap_ISCX.csv"])
unknown_data = NetworkDataset(["MachineLearningCVE/Wednesday-workingHours.pcap_ISCX.csv"])

CLASSES = len(data_total.classes)

data_train, data_test = torch.utils.data.random_split(data_total, [len(data_total)-1000,1000])

training =  torch.utils.data.DataLoader(dataset=data_train, batch_size=BATCH, shuffle=True)
testing = torch.utils.data.DataLoader(dataset=data_test, batch_size=BATCH, shuffle=False)
unknowns = torch.utils.data.DataLoader(dataset=unknown_data, batch_size=BATCH, shuffle=False)


model = Network(CLASSES).to(device)
evaluative = Evaluation.correctValCounter(CLASSES,confusionMat=True)

if os.path.exists(NAME+"/checkpointCheckingRand.pth") and False:
    model.load_state_dict(torch.load(NAME+"/checkpointCheckingRand.pth"))

epochs = 5
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)


for e in range(epochs):
    lost_amount = 0

    for batch, (X, y) in enumerate(training):
        X = X.to(device)
        y = y.to(device)

        output = model(X)
        lost_points = criterion(output, y)
        optimizer.zero_grad()
        lost_points.backward()

        #printing paramiters to check if they are moving
        #for para in model.parameters():
            #print(para.grad)

        optimizer.step()

        lost_amount += lost_points.item()

    
    with torch.no_grad():
        model.eval()
        for batch,(X,y) in enumerate(testing):
            X = X.to(device)
            y = y.to("cpu")

            output = model(X).to("cpu")
            evaluative.evalN(output,y)
            

        print(f"-----------------------------Epoc: {e+1}-----------------------------")
        print(f"lost: {100*lost_amount/len(data_train)}")
        evaluative.PrintEval()
        evaluative.zero()
        
        if e%5 == 4:
            torch.save(model.state_dict(), NAME+"/checkpointCheckingRand.pth")

        model.train()
    scheduler.step()


#Everything past here is unknowns

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