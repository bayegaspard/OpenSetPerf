import numpy as np
import torch
import torch.utils.data
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import OpenMaxByMaXu
import os
import matplotlib.pyplot as plt
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
BATCH = 500
NAME = os.path.basename(os.path.dirname(__file__))
CUTOFF = 0.5

#I looked up how to make a dataset, more information in the LoadImages file
#images are from: http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/

path_to_dataset = "datasets" #put the absolute path to your dataset , type "pwd" within your dataset folder from your teminal to know this path.

def getListOfCSV(path):
    return glob.glob(path+"/*.csv")

data_total = NetworkDataset(getListOfCSV(path_to_dataset),benign=True)
unknown_data = NetworkDataset(getListOfCSV(path_to_dataset),benign=False)

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
plotter = torch.zeros((2,25))
plotter[1] += torch.tensor([x for x in range(25)])/25


model = Network(CLASSES).to(device)

#initialize the counters, op for open because open is a keyword
soft = correctValCounter(CLASSES, cutoff=CUTOFF)
op = correctValCounter(CLASSES, cutoff=CUTOFF)

if os.path.exists(NAME+"/src/checkpoint.pth"):
    model.load_state_dict(torch.load(NAME+"/src/checkpoint.pth",map_location=device))
    epochs = 5
else:
    epochs = 5

epochs = 0
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)



#this is really awful looking but I want the code to be able to work even with only saved epochs
if epochs == 0:
    unknownscore = 0
    with torch.no_grad():
        model.eval()

        #these three lines somehow setup for the openmax thing
        scores, mavs, distances = OpenMaxByMaXu.compute_train_score_and_mavs_and_dists(CLASSES,training2,device,model)
        catagories = list(range(CLASSES))
        weibullmodel = OpenMaxByMaXu.fit_weibull(mavs,distances,catagories,tailsize=10)

        op.setWeibull(weibullmodel)

        for batch,(X,y) in enumerate(testing):
            X = X.to(device)
            y = y.to("cpu")

            _, output = model(X)

            output_true = output.to("cpu")

            for x in plotter[1]:
                for logits in output_true:
                    #this is where the openmax is run, I did not create the openmax
                    output_open_new, output_soft_new = OpenMaxByMaXu.openmax(weibullmodel, catagories,logits.numpy()[np.newaxis,:], x)
                    plotter[0][int(x*25)] += (torch.argmax(torch.tensor(output_open_new), dim=0) != CLASSES).sum()/1000
            

            soft.evalN(output,y)
            op.evalN(output,y, type="Open")
            soft.cutoffStorage(output[:len(X)].detach(), "Soft")
            op.cutoffStorage(output[:len(X)].detach(), "Open")
        soft.autocutoff()
        op.autocutoff()


        
        print("-----------------------------From savepoint-----------------------------")
        print("SoftMax:")
        soft.PrintEval()
        print("OpenMax:")
        op.PrintEval()
        plt.plot(plotter[1],plotter[0],"r-")
        soft.zero()
        op.zero() 

        model.train()

#For graphing
plotter = torch.zeros((2,25))
plotter[1] += torch.tensor([x for x in range(25)])/25

#Everything past here is unknowns

with torch.no_grad():
    unknownscore = 0
    model.eval()
    #these three lines somehow setup for the openmax thing
    scores, mavs, distances = OpenMaxByMaXu.compute_train_score_and_mavs_and_dists(CLASSES,training2,device,model)
    catagories = list(range(CLASSES))
    weibullmodel = OpenMaxByMaXu.fit_weibull(mavs,distances,catagories,tailsize=10)

    op.setWeibull(weibullmodel)

    for batch,(X,y) in enumerate(unknowns):
        X = X.to(device)
        y = y.to("cpu")

        _, output = model(X)

        output_true = output.to("cpu")


        for x in plotter[1]:
            for logits in output_true:
                #this is where the openmax is run, I did not create the openmax
                output_open_new, output_soft_new = OpenMaxByMaXu.openmax(weibullmodel, catagories,logits.numpy()[np.newaxis,:], x)
                plotter[0][int(x*25)] += (torch.argmax(torch.tensor(output_open_new), dim=0) != CLASSES).sum()/26416

        soft.evalN(output,y,offset=26)
        op.evalN(output,y,offset=26,type="Open")

    print("SoftMax:")
    soft.PrintUnknownEval()
    print("OpenMax:")
    op.PrintUnknownEval()
    print(f"Unknown Score: {unknownscore}")
    plt.plot(plotter[1],plotter[0],"b-")
    soft.zero()
    op.zero()
    
    model.train()

plt.show()
print("Done")