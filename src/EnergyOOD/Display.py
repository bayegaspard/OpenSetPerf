#https://github.com/wetliu/energy_ood <- associated paper
import numpy as np
from LoadPackets import NetworkDataset
import torch
import torch.utils.data
from torchvision import transforms
import torchvision
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import EvaluationDisplay
import os
import EnergyCodeByWetliu

torch.manual_seed(0)
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

data_train, data_test = torch.utils.data.random_split(data_total, [35576,1000])

training =  torch.utils.data.DataLoader(dataset=data_train, batch_size=BATCH, shuffle=True)
testing = torch.utils.data.DataLoader(dataset=data_test, batch_size=BATCH, shuffle=False)
unknowns = torch.utils.data.DataLoader(dataset=unknown_data, batch_size=72, shuffle=True)


class Network(nn.Module):
    def __init__(self):
        super().__init__()

        self.cv1 = nn.Conv2d(1,4,5)
        self.pool1 = nn.MaxPool2d((2,2),(2,2))
        self.RL = nn.LeakyReLU()
        self.cv2 = nn.Conv2d(4,16,5)
        self.pool2 = nn.MaxPool2d(2,2)
        self.cv3 = nn.Conv2d(16,64,5)
        self.pool3 = nn.MaxPool2d(2,2)

        self.fc1 = nn.Linear(5184, 1028)
        self.fc2 = nn.Linear(1028,240)
        self.fc3 = nn.Linear(240,CLASSES)

        self.fc = nn.Linear(4800,CLASSES)
        self.dropout = nn.Dropout(0.3)

        self.soft = nn.Softmax(dim=1)
        self.double()

    def forward(self, input):
        X = input.double()
        X = self.pool1(self.RL(self.cv1(self.dropout(X))))
        X = self.pool2(self.RL(self.cv2(self.dropout(X))))
        X = self.pool3(self.RL(self.cv3(self.dropout(X))))

        X = torch.flatten(X, start_dim=1)

        X = self.RL(self.fc1(self.dropout(X)))
        X = self.RL(self.fc2(self.dropout(X)))
        return self.fc3(X)

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
