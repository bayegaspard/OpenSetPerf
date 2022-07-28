import numpy as np
from LoadPackets import NetworkDataset
import torch
import torch.utils.data
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import Evaluation
import OpenMaxByMaXu
import os
import matplotlib.pyplot as plt
from ModelLoader import Network



#pick a device
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")

torch.manual_seed(0)
BATCH = 500
NAME = "OpenMax"
CUTOFF = 0.5

#I looked up how to make a dataset, more information in the LoadImages file
#images are from: http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/
data_total = NetworkDataset(["MachineLearningCVE/Monday-WorkingHours.pcap_ISCX.csv","MachineLearningCVE/Tuesday-WorkingHours.pcap_ISCX.csv"])
unknown_data = NetworkDataset(["MachineLearningCVE/Wednesday-workingHours.pcap_ISCX.csv"])

CLASSES = len(data_total.classes)

data_train, data_test = torch.utils.data.random_split(data_total, [len(data_total)-1000,1000])
testing = torch.utils.data.DataLoader(dataset=data_test, batch_size=BATCH, shuffle=False)
training =  torch.utils.data.DataLoader(dataset=data_train, batch_size=BATCH, shuffle=True)

#this needs to be improved 
data_total.isOneHot = False
data_train2, _ = torch.utils.data.random_split(data_total, [len(data_total)-1000,1000])
training2 = torch.utils.data.DataLoader(dataset=data_train2, batch_size=BATCH, shuffle=True)

#load the unknown data
unknowns = torch.utils.data.DataLoader(dataset=unknown_data, batch_size=BATCH, shuffle=False)

#For graphing
plotter = torch.zeros((2,18))
plotter[1] += torch.tensor([0,0.01,0.02,0.03,0.04,0.05,0.1,0.15,0.2,0.25,0.5,0.75,1,2,4,8,16,32])


model = Network(CLASSES).to(device)
#initialize the counters, op for open because open is a keyword
soft = Evaluation.correctValCounter(CLASSES, cutoff=CUTOFF)
op = Evaluation.correctValCounter(CLASSES, cutoff=CUTOFF)

if os.path.exists(NAME+"/checkpoint 100 epochs.pth"):
    model.load_state_dict(torch.load(NAME+"/checkpoint 100 epochs.pth",map_location=device))

criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

weibullmodel = []


unknownscore = 0
with torch.no_grad():
    model.eval()

    #these three lines somehow setup for the openmax thing
    scores, mavs, distances = OpenMaxByMaXu.compute_train_score_and_mavs_and_dists(CLASSES,training2,device,model)
    catagories = list(range(CLASSES))
    weibullmodel = OpenMaxByMaXu.fit_weibull(mavs,distances,catagories,tailsize=10)

    for batch,(X,y) in enumerate(testing):
        X = X.to(device)
        y = y.to("cpu")

        _, output_true = model(X)

        output_true = output_true.to("cpu")

        output_soft = []
        output_open = []
        for logits in output_true:
            #this is where the openmax is run, I did not create the openmax
            output_open_new, output_soft_new = OpenMaxByMaXu.openmax(weibullmodel, catagories,logits.numpy()[np.newaxis,:], 0.4)
            output_soft.append(torch.tensor(output_soft_new).unsqueeze(dim=0))
            output_open.append(torch.tensor(output_open_new).unsqueeze(dim=0))

        

        output_soft = torch.cat(output_soft, dim=0)
        output_open = torch.cat(output_open, dim=0)

        for a,b in enumerate(plotter[1]):
            plotter[0][a] += output_open.max(dim=1)[0].greater_equal(b).sum()


        unknownscore += (output_open[:,CLASSES]).sum().item()

        soft.evalN(output_soft,y,needSoft=False)
        op.evalN(output_open,y, type="Open",needSoft=False)



    
    print("-----------------------------From savepoint-----------------------------")
    print("SoftMax:")
    soft.PrintEval()
    print("OpenMax:")
    op.PrintEval()
    print(f"Unknown Score: {unknownscore}")
    plt.plot(plotter[1],plotter[0]/1000,"r-")
    soft.zero()
    op.zero() 

    model.train()


#For graphing
plotter = torch.zeros((2,18))
plotter[1] += torch.tensor([0,0.01,0.02,0.03,0.04,0.05,0.1,0.15,0.2,0.25,0.5,0.75,1,2,4,8,16,32])


#Everything past here is unknowns


with torch.no_grad():
    unknownscore = 0
    model.eval()
    #these three lines somehow setup for the openmax thing
    scores, mavs, distances = OpenMaxByMaXu.compute_train_score_and_mavs_and_dists(CLASSES,training2,device,model)
    catagories = list(range(CLASSES))
    weibullmodel = OpenMaxByMaXu.fit_weibull(mavs,distances,catagories,tailsize=10)

    for batch,(X,y) in enumerate(unknowns):
        X = X.to(device)
        y = y.to("cpu")

        _, output_true = model(X)

        output_true = output_true.to("cpu")


        output_soft = []
        output_open = []
        for logits in output_true:
            #this is where the openmax is run, I did not create the openmax
            output_open_new, output_soft_new = OpenMaxByMaXu.openmax(weibullmodel, catagories,logits.numpy()[np.newaxis,:], 0.4)
            output_soft.append(torch.tensor(output_soft_new).unsqueeze(dim=0))
            output_open.append(torch.tensor(output_open_new).unsqueeze(dim=0))


        output_soft = torch.cat(output_soft, dim=0)
        output_open = torch.cat(output_open, dim=0)

        for a,b in enumerate(plotter[1]):
            plotter[0][a] += output_open.max(dim=1)[0].greater_equal(b).sum()

        unknownscore += (output_open[:,CLASSES]).sum().item()

        soft.evalN(output_soft,y,offset=26,needSoft=False)
        op.evalN(output_open,y,offset=26,type="Open",needSoft=False)

    print("SoftMax:")
    soft.PrintUnknownEval()
    print("OpenMax:")
    op.PrintUnknownEval()
    print(f"Unknown Score: {unknownscore}")
    plt.plot(plotter[1],plotter[0]/26416,"b-")
    soft.zero()
    op.zero()
    
    model.train()

plt.show()
print("Done")