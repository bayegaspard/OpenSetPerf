#These are all the functions that dont fit elsewhere
import Config as Config
import os
import pandas as pd
import torch
import GPU
import FileHandling
import Dataload
import plots
from sklearn.metrics import (precision_score, recall_score, average_precision_score)

#Translation dictionaries for algorithms that cannot have gaps in their numbers.
relabel = {15:15}
rerelabel = {15:15}
def setrelabel():
    global relabel,rerelabel
    relabel = {15:15}
    rerelabel = {15:15}
    temp = 0
    for x in range(15):
        if temp < len(Config.helper_variables["unknowns_clss"]) and x == Config.helper_variables["unknowns_clss"][temp]:
            temp = temp+1
        else:
            relabel[x] = x-temp
            rerelabel[x-temp] = x
    temp = None
setrelabel()

def deleteSaves():
    i = 0
    while os.path.exists(f"Saves/Epoch{i:03d}.pth"):
        os.remove(f"Saves/Epoch{i:03d}.pth")
        i = i+1






#Handels running the loop
def testRotate(notes=(0,0,0)):
    global relabel,rerelabel
    stage = notes[0]
    step = notes[1]
    al = notes[2]

    deleteSaves()
    if step+1 < len(Config.loops[stage]):
        step = step+1

        if stage == 2:
            Config.parameters[Config.loops2[stage]] = Config.loops[stage][step]
        elif stage == 4:
            Config.helper_variables["unknowns_clss"] = Config.loops[stage][step]
            Config.parameters["Unknowns"] = f"{len(Config.loops[stage][step])} Unknowns"
            Config.helper_variables["knowns_clss"] = Config.loopOverUnknowns(Config.helper_variables["unknowns_clss"])
            setrelabel()
        else:
            Config.parameters[Config.loops2[stage]][0] = Config.loops[stage][step]

        return (stage,step,al)

    #reset this stage
    step = 0

    if stage == 2:
        Config.parameters[Config.loops2[stage]] = Config.loops[stage][step]
    elif stage == 4:
        Config.helper_variables["unknowns_clss"] = Config.loops[stage][step]
        Config.parameters["Unknowns"] = f"{len(Config.loops[stage][step])} Unknowns"
        Config.helper_variables["knowns_clss"] = Config.loopOverUnknowns(Config.helper_variables["unknowns_clss"])
        setrelabel()
    else:
        Config.parameters[Config.loops2[stage]][0] = Config.loops[stage][step]

    #Go to next stage
    if stage+1 < len(Config.loops):
        stage = stage+1
        #Skip the next rotate algorithm step and just go to rotate step
        return testRotate((stage,step,al))

    #Reset stage
    stage = 0

    if al+1 < len(Config.alg):
        al = al+1
        Config.parameters["OOD Type"][0] = Config.alg[al]
        return (stage,step,al)
    

    #Done with looping
    Config.parameters["LOOP"][0] = False
    return False

def getcurrentlychanged(notes):
    algorithm = Config.alg[notes[2]]
    currentlyChanging = Config.loops2[notes[0]]
    currentSetting = Config.loops[notes[0]][notes[1]]
    return str(algorithm)+" "+str(currentlyChanging)+" "+str(currentSetting)


def looptest():
    out = pd.DataFrame(())
    out2 = pd.DataFrame(())

    count = 0
    notes = (0,0,0)
    while notes:
        current = pd.DataFrame(Config.parameters)
        current2 = pd.DataFrame(Config.helper_variables["unknowns_clss"])
        out = pd.concat([out,current.iloc[0]],axis=1)
        out2 = pd.concat([out2,current2],axis=1)
        print(getcurrentlychanged(notes))
        notes = testRotate(notes)
        count = count+1

    out = pd.concat([current.iloc[0],out],axis=1)

    out.to_csv("Testing.csv")
    out2.to_csv("Testing2.csv")
    print(f"That means the model will have to run {count} times")


#This loops through all the thresholds without resetting the model.
def thresholdTest(net,val_loader):
        net.end.type = Config.parameters["OOD Type"][0]
        net.loadPoint("Saves")
        thresh = Config.thresholds
        for y in range(len(thresh)):
            x = thresh[y]
            #reset
            net.end.resetvals()
            net.store = GPU.to_device(torch.tensor([]), net.device), GPU.to_device(torch.tensor([]), net.device), GPU.to_device(torch.tensor([]), net.device)
            net.end.cutoff = x
            
            #evaluate
            net.evaluate(val_loader)

            #get the data
            y_pred,y_true,y_tested_against = net.store

            #evaluate the data
            y_true = y_true.to(torch.int).tolist()
            y_pred = y_pred.to(torch.int).tolist()
            y_tested_against = y_tested_against.to(torch.int).tolist()
            recall = recall_score(y_tested_against,y_pred,average='weighted',zero_division=0)
            precision = precision_score(y_tested_against,y_pred,average='weighted',zero_division=0)
            f1 = 2 * (precision * recall) / (precision + recall)
            #save the f1 score
            FileHandling.create_params_Fscore("",f1,x)

            #Generate and save confusion matrix
            # if plots.name_override:
            #     if y > 0:
            #         plots.name_override = plots.name_override.replace(f" Threshold:{thresh[y-1]}",f" Threshold:{x}")
            #     else:
            #         plots.name_override = plots.name_override+f" Threshold:{x}"
            # else:
            #     plots.name_override = f"Default_setting Threshold:{x}"

            # class_names = Dataload.get_class_names(range(15))
            # for x in Config.helper_variables["unknowns_clss"]["unknowns"]:
            #     class_names[x] = class_names[x]+"*"
            # class_names.append("*Unknowns")
            #cnf_matrix = plots.confusionMatrix(y_true.copy(), y_pred.copy(), y_tested_against.copy()) 

            #plots.plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
            #                title='Confusion matrix', knowns = Config.helper_variables["knowns_clss"])


class LossPerEpoch():
    def __init__(self,name):
        self.loss = 0
        self.name = name

    def addloss(self,predicted:torch.Tensor, target:torch.Tensor):
        locations = predicted!=target
        self.loss += locations.sum().item()

    def collect(self):
        if os.path.exists(os.path.join("Saves",self.name)):
            hist = pd.read_csv(os.path.join("Saves",self.name),index_col=0)
        else:
            hist = pd.DataFrame([])
        param = pd.DataFrame(Config.parameters).iloc[0]
        #current = pd.DataFrame({"Number of failures":[self.loss]})
        #current = pd.concat([param,current])
        param["Number Of Failures"] = self.loss

        hist = pd.concat([hist,param],axis=1)
        hist.to_csv(os.path.join("Saves",self.name))
        self.loss = 0


if __name__ == "__main__":
    looptest()