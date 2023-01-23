#These are all the functions that dont fit elsewhere
import Config as Config
import os
import pandas as pd

#Translation dictionaries for algorithms that cannot have gaps in their numbers.
relabel = {15:15}
rerelabel = {15:15}
temp = 0
for x in range(15):
    if temp < len(Config.helper_variables["unknowns_clss"]["unknowns"]) and x == Config.helper_variables["unknowns_clss"]["unknowns"][temp]:
        temp = temp+1
    else:
        relabel[x] = x-temp
        rerelabel[x-temp] = x
temp = None


def deleteSaves():
    i = 0
    while os.path.exists(f"Saves/Epoch{i:03d}.pth"):
        os.remove(f"Saves/Epoch{i:03d}.pth")
        i = i+1



#This is to test all of the algorithms one after the other
alg = ["Soft","Open","Energy","COOL","DOC"]
learning_rates = [0.001,0.01,1]
epochs = [10,50,100]
optim = [Config.opt_func["Adam"], Config.opt_func["SGD"], Config.opt_func["RMSprop"]]
activation = ["ReLU", "Tanh", "Sigmoid"]

learning_rates.insert(0,Config.parameters["learningRate"][0])
epochs.insert(0,Config.parameters["num_epochs"][0])

loops = [learning_rates,epochs,optim,activation]
loops2 = ["learningRate","num_epochs","optimizer","Activation"]

def testRotate(notes=(0,0,0)):
    stage = notes[0]
    step = notes[1]
    al = notes[2]
    if al+1 < len(alg):
        al = al+1
        Config.parameters["OOD Type"][0] = alg[al]
        return (stage,step,al)
    al = 0
    Config.parameters["OOD Type"][0] = alg[al]
    if step+1 < len(loops[stage]):
        step = step+1
        if stage != 2:
            Config.parameters[loops2[stage]][0] = loops[stage][step]
        else:
            Config.parameters[loops2[stage]] = loops[stage][step]
        return (stage,step,al)
    step = 0
    if stage != 2:
        Config.parameters[loops2[stage]][0] = loops[stage][step]
    else:
        Config.parameters[loops2[stage]] = loops[stage][step]
    if stage+1 < len(loops):
        stage = stage+1
        return testRotate((stage,step,al))
    return False


def looptest():
    out = pd.DataFrame(())

    notes = (0,0,0)
    while notes:
        notes = testRotate(notes)
        current = pd.DataFrame(Config.parameters)
        out = pd.concat([out,current.iloc[0]],axis=1)

    out = pd.concat([current.iloc[0],out],axis=1)

    out.to_csv("Testing.csv")


if __name__ == "__main__":
    looptest()