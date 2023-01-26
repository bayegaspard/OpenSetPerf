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
#alg.remove("Soft")
#alg.remove("Open")
#alg.remove("Energy")
#epochs= []
epochs = [10,50,100]
optim = [Config.opt_func["Adam"], Config.opt_func["SGD"], Config.opt_func["RMSprop"]]
activation = ["ReLU", "Tanh", "Sigmoid"]
groups = [[2],[2,3,6],[2,3,4,5,6],[2,3,4,5,6,7,11],[2,3,4,5,6,7,11,12,14],[2,3,4,5,6,7,8,9,11,12,14],[2,3,4,5,6,7,8,9,10,11,12,13,14],[1,2,3,4,5,6,7,8,9,10,11,12,13,14]]

learning_rates.insert(0,Config.parameters["learningRate"][0])
epochs.insert(0,Config.parameters["num_epochs"][0])
groups.insert(0,Config.helper_variables["unknowns_clss"]["unknowns"])

loops = [learning_rates,epochs,optim,activation,groups]
loops2 = ["learningRate","num_epochs","optimizer","Activation","Unknowns"]

def testRotate(notes=(0,0,0)):
    stage = notes[0]
    step = notes[1]
    al = notes[2]
    if al+1 < len(alg):
        al = al+1
        Config.parameters["OOD Type"][0] = alg[al]
        return (stage,step,al)
    deleteSaves()
    al = 0
    Config.parameters["OOD Type"][0] = alg[al]
    if step+1 < len(loops[stage]):
        step = step+1

        if stage == 2:
            Config.parameters[loops2[stage]] = loops[stage][step]
        elif stage == 4:
            Config.helper_variables["unknowns_clss"]["unknowns"] = loops[stage][step]
            Config.parameters["Unknowns"] = f"{len(loops[stage][step])} Unknowns"
        else:
            Config.parameters[loops2[stage]][0] = loops[stage][step]

        return (stage,step,al)
    step = 0

    if stage == 2:
        Config.parameters[loops2[stage]] = loops[stage][step]
    elif stage == 4:
        Config.helper_variables["unknowns_clss"]["unknowns"] = loops[stage][step]
        Config.parameters["Unknowns"] = f"{len(loops[stage][step])} Unknowns"
    else:
        Config.parameters[loops2[stage]][0] = loops[stage][step]

    if stage+1 < len(loops):
        stage = stage+1
        #Skip the next rotate algorithm step and just go to rotate step
        return testRotate((stage,step,len(alg)))

    #Done with looping
    Config.parameters["LOOP"][0] = False
    return False

def getcurrentlychanged(notes):
    algorithm = alg[notes[2]]
    currentlyChanging = loops2[notes[0]]
    currentSetting = loops[notes[0]][notes[1]]
    return str(algorithm)+" "+str(currentlyChanging)+" "+str(currentSetting)


def looptest():
    out = pd.DataFrame(())
    out2 = pd.DataFrame(())

    notes = (0,0,0)
    while notes:
        current = pd.DataFrame(Config.parameters)
        current2 = pd.DataFrame(Config.helper_variables["unknowns_clss"]["unknowns"])
        out = pd.concat([out,current.iloc[0]],axis=1)
        out2 = pd.concat([out2,current2],axis=1)
        print(getcurrentlychanged(notes))
        notes = testRotate(notes)

    out = pd.concat([current.iloc[0],out],axis=1)

    out.to_csv("Testing.csv")
    out2.to_csv("Testing2.csv")


if __name__ == "__main__":
    looptest()