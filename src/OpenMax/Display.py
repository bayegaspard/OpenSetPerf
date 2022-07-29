import sys
import numpy as np
import torch
import torch.utils.data
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import glob

#three lines from https://xxx-cook-book.gitbooks.io/python-cook-book/content/Import/import-from-parent-folder.html
import sys
root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)

#this seems really messy
from HelperFunctions.LoadPackets import NetworkDataset
from HelperFunctions.Evaluation import correctValCounter
from HelperFunctions.ModelLoader import Network
import CodeFromImplementations.OpenMaxByMaXu as OpenMaxByMaXu

#this file shows graphs for the openmax model. The graphs use the same axies as the openmax paper (https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Bendale_Towards_Open_Set_CVPR_2016_paper.pdf)
#This can be run in the same way as a 'main.py' file.

device = torch.device("cpu")

torch.manual_seed(0)
NAME = os.path.basename(os.path.dirname(__file__))
BATCH = 100

#I looked up how to make a dataset, more information in the LoadImages file
#images are from: http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/

path_to_dataset = "datasets" #put the absolute path to your dataset , type "pwd" within your dataset folder from your teminal to know this path.

def getListOfCSV(path):
    return glob.glob(path+"/*.csv")

data_total = NetworkDataset(getListOfCSV(path_to_dataset),benign=True)
unknown_data = NetworkDataset(getListOfCSV(path_to_dataset),benign=False)

CLASSES = len(data_total.classes)

#This does not have a direct comparison.
random_data = NetworkDataset(getListOfCSV(path_to_dataset))

data_train, data_test = torch.utils.data.random_split(data_total, [len(data_total)-1000,1000])
testing = torch.utils.data.DataLoader(dataset=data_test, batch_size=BATCH, shuffle=False)
training =  torch.utils.data.DataLoader(dataset=data_train, batch_size=BATCH, shuffle=True)

#this needs to be improved 
data_total.isOneHot = False
data_train2, _ = torch.utils.data.random_split(data_total, [len(data_total)-1000,1000])
training2 = torch.utils.data.DataLoader(dataset=data_train2, batch_size=BATCH, shuffle=True)

#load the unknown data
unknowns = torch.utils.data.DataLoader(dataset=unknown_data, batch_size=BATCH, shuffle=False)

#load the randomly distributed data
randoms = torch.utils.data.DataLoader(dataset=random_data, batch_size=25, shuffle=True)


model = Network(CLASSES).to(device)

#initialize the counters, op for open because open is a keyword
soft = correctValCounter(CLASSES)
op = correctValCounter(CLASSES, cutoff=0.95)

fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()
fig4, ax4 = plt.subplots()

ax2.set_ylabel("OpenMax Probibities")
ax2.set_xlabel("SoftMax Probibities")
ax2.set_title("Open Vs Soft Probibilities")

ax3.set_title("F-Measures vs cutoff threasholds")
ax3.set_ylabel("F-Measures")
ax3.set_xlabel("Thresholds")

ax4.set_title("Accuracy vs cutoff threasholds")
ax4.set_ylabel("Accuracy")
ax4.set_xlabel("Thresholds")


if os.path.exists(NAME+"/src/checkpoint.pth"):
    model.load_state_dict(torch.load(NAME+"/src/checkpoint.pth",map_location=torch.device("cpu")))
else:
    sys.exit("you need a model")

#not training it here
model.eval()



with torch.no_grad():
    unknownscore = 0
    #these three lines somehow setup for the openmax thing
    scores, mavs, distances = OpenMaxByMaXu.compute_train_score_and_mavs_and_dists(CLASSES,training2,device,model)
    catagories = list(range(CLASSES))
    weibullmodel = OpenMaxByMaXu.fit_weibull(mavs,distances,catagories)

    op.setWeibull(weibullmodel)

    for batch,(X,y) in enumerate(testing):
        X = X
        y = y

        _, output = model(X)


        soft.evalN(output,y)
        op.evalN(output,y, type="Open")


    for batch,(X,y) in enumerate(training):
        X = X
        y = y

        _, output = model(X)


        
        soft.evalN(output,y)
        op.evalN(output,y, type="Open")

    
    print("-----------------------------From savepoint-----------------------------")
    print("SoftMax:")
    soft.PrintEval()
    print("OpenMax:")
    op.PrintEval()
    print(f"Unknown score: {unknownscore}")

    ax2.plot(soft.plotPercentageList,op.plotPercentageList, "r.",label="In distribution")

    soft.zero()
    op.zero()


#Everything past here is unknowns

with torch.no_grad():
    unknownscore = 0
    for batch,(X,y) in enumerate(unknowns):

        _, output = model(X)

        
        soft.evalN(output,y, offset=26)
        op.evalN(output,y, type="Open", offset=26)

    print("SoftMax:")
    soft.PrintUnknownEval()
    print("OpenMax:")
    op.PrintUnknownEval()

    ax2.plot(soft.plotPercentageList,op.plotPercentageList, "b.", label="Out distribution")

    soft.zero()
    op.zero()


soft.cutoff = 0
op.cutoff = 0

plots = [[[],[]],[[],[]]]
distance = []


times = 25

for x in range(times):
    with torch.no_grad():
        correctlyDecided = [0,0]
        for batch,(X,y) in enumerate(randoms):
            _, output = model(X)

            i = torch.zeros(25, CLASSES+1)
            i[:,:CLASSES] += y
            i[:,CLASSES] += 1
            y = i

            trueval = torch.argmax(y,dim=1)
            totalfory = torch.bincount(trueval, minlength=CLASSES+1)
            
            
            #SOFTMAX
            values = soft.softMaxMod(output)
            valmax = values.max(dim=1)[0]

            selected = torch.argmax(output, dim=1)

            #in distribution
            sumInDist = (selected==trueval) * valmax.greater(x/25)
            #out distribution
            sumOutDist = (trueval==CLASSES) * (valmax.less_equal(x/25))
            #totals
            correctlyDecided[0] += (sumInDist+sumOutDist).sum().item()

            save = (valmax.less_equal(x/25))

            #OPENMAX
            values = op.openMaxMod(output)
            valmax = values.max(dim=1)[0]

            selected = torch.argmax(values, dim=1)

            #in distribution
            sumInDist = (selected==trueval) * valmax.greater(x/25)
            #out distribution
            sumOutDist = (trueval==CLASSES) * (((selected==CLASSES) + valmax.less_equal(x/25))>0)
            #totals
            correctlyDecided[1] += (sumInDist+sumOutDist).sum().item()


            if (save * (((selected==CLASSES) + valmax.less_equal(x/25))>0)).sum().item() != 0:
                print(f"{x},{batch}")

        #openMax
        precision = correctlyDecided[0]/10000
        #True Positive Rate
        recall = correctlyDecided[0]/totalfory.sum()
        openF = 2*(precision*recall)/(precision+recall)    
        openF = torch.nan_to_num(openF, 0).numpy()

        #SoftMax
        precision = correctlyDecided[1]/10000
        #True Positive Rate
        recall = correctlyDecided[1]/totalfory.sum()
        softF = 2*(precision*recall)/(precision+recall)
        softF = torch.nan_to_num(softF.sum(), 0).numpy()    


        print(f"[{x+1}/{times}]")

        
        plots[0][0].append(softF)
        plots[0][1].append(openF)
        
        plots[1][0].append(correctlyDecided[0]/10000)
        plots[1][1].append(correctlyDecided[1]/10000)

        soft.zero()
        op.zero()
        soft.cutoff += 0.036
        op.cutoff=soft.cutoff
        distance.append(op.cutoff)

ax3.plot(distance,plots[0][0], "bs-", label="SoftMax")
ax3.plot(distance,plots[0][1], "ro-", label="OpenMax")
ax4.plot(distance,plots[1][0], "bs-", label="SoftMax")
ax4.plot(distance,plots[1][1], "ro-", label="OpenMax")

ax2.legend()
ax3.legend()
ax4.legend()


plt.show()
print("Done")