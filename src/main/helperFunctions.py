#These are all the functions that dont fit elsewhere
import Config as Config
import os

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
#THIS IS REALLY BADLY WRITTEN
AlgstartedOn = Config.parameters["OOD Type"][0]
LrstartedOn = Config.parameters["learningRate"][0]
EPstartedOn = Config.parameters["num_epochs"][0]
OPstartedOn = Config.parameters["optimizer"]
ACstartedOn = Config.parameters["Activation"][0]

startedEpochs = Config.parameters["num_epochs"][0]
def testRotateLayerFinal():
    current = Config.parameters["OOD Type"][0]
    if current == "Soft":
        Config.parameters["OOD Type"][0] = "Open"
        #Config.parameters["num_epochs"][0] = 0
    elif current == "Open":
        Config.parameters["OOD Type"][0] = "Energy"
        #Config.parameters["num_epochs"][0] = 0
    elif current == "Energy":
        Config.parameters["OOD Type"][0] = "COOL"
        #Config.parameters["num_epochs"][0] = startedEpochs
    elif current == "COOL":
        Config.parameters["OOD Type"][0] = "DOC"
        #Config.parameters["num_epochs"][0] = startedEpochs
    elif current == "DOC":
        Config.parameters["OOD Type"][0] = "Soft"
        #Config.parameters["num_epochs"][0] = startedEpochs
    if Config.parameters["OOD Type"][0] == AlgstartedOn:
        return False
    return True

def testRotateLayer4():
    if Config.parameters["Activation"][0] == "ReLU":
        Config.parameters["Activation"][0] = "Tanh"
    elif Config.parameters["Activation"][0] == "Tanh":
        Config.parameters["Activation"][0] = "Sigmoid"
    elif Config.parameters["Activation"][0] == "Sigmoid":
        Config.parameters["Activation"][0] = ACstartedOn
        return False
    else:
        Config.parameters["Activation"][0] = "ReLU"
    return True

def testRotateLayer3():
    if Config.parameters["optimizer"] is optim[0]:
        Config.parameters["optimizer"] = optim[1]
    elif Config.parameters["optimizer"] is optim[1]:
        Config.parameters["optimizer"] = optim[2]
    elif Config.parameters["optimizer"] is optim[2]:
        Config.parameters["optimizer"] = OPstartedOn
        if not testRotateLayer4():
            return False
    else:
        Config.parameters["optimizer"] = optim[0]
    return True

def testRotateLayer2():
    if Config.parameters["num_epochs"][0] == 10:
        Config.parameters["num_epochs"][0] = 50
    elif Config.parameters["num_epochs"][0] == 50:
        Config.parameters["num_epochs"][0] = 100
    elif Config.parameters["num_epochs"][0] == 100:
        Config.parameters["num_epochs"][0] = EPstartedOn
        if not testRotateLayer3():
            return False
    else:
        Config.parameters["num_epochs"][0] = 10
    return True

def testRotate():
    deleteSaves()
    if Config.parameters["learningRate"][0] == 0.001:
        Config.parameters["learningRate"][0] = 0.01
    elif Config.parameters["learningRate"][0] == 0.01:
        Config.parameters["learningRate"][0]= 1
    elif Config.parameters["learningRate"][0] == 1:
        Config.parameters["learningRate"][0] = LrstartedOn
        if not testRotateLayer2():
            return testRotateLayerFinal()
    else:
        Config.parameters["learningRate"][0] = 0.001
    return True

learning_rates = [0.001,0.01,1]
epochs = [10,50,100]
optim = [Config.opt_func["Adam"], Config.opt_func["SGD"], Config.opt_func["RMSprop"]]
activation = ["ReLU", "Tanh", "Sigmoid"]
