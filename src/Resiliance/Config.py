import torch
import pandas as pd
import os

#This config file is mainly used as global variables for the rest of the program.
#It should only be modified by the two loop commands in helperfunctions


def loopOverUnknowns(unknownlist):
    """
    Given a list of unknowns (integers 0-14) this will create a list of knowns (the inverted list).
    """
    knownVals = list(range(parameters["CLASSES"][0]))
    for un in unknownlist:
        knownVals.remove(un)
    return knownVals

#This is the diffrent optimization functions
opt_func = {"Adam":torch.optim.Adam,"SGD":torch.optim.SGD, "RMSprop":torch.optim.RMSprop}

#I do not know why this is diffrent than the parameters dictionary
helper_variables = {
    "phase" : -1,
    "startphase" : 0,

    #This is the only important value in this dictionary and it lists the diffrent values to consider unkowns.
    #Mappings are at the top of Dataload.py
    "unknowns_clss": [10,11,12,13,14], #Overriden if loop=2

    "e": 0
}


#Here are all of the paremeters for the model.
parameters = {
    #These parameters are orginized like this:
    #"ParamName":[Value,"Description"]
    #for a parameter called "ParamName" with a value of Value
    "batch_size":[1000, "Number of items per batch"],
    "num_workers":[0, "Number of threads working on building batches"],
    "attemptLoad":[0, "0: do not use saves\n1:use saves"],
    "testlength":[1/4, "[0,1) percentage of training to test with"],
    "Mix unknowns and validation": [0,"0 or 1, 0 means that the test set is purely unknowns and 1 means that the testset is the validation set plus unknowns (for testing)"],
    "MaxSamples": [5000, "Maximum number of samples"],
    "num_epochs":[10,"Number of times it trains on the whole trainset"],
    "learningRate":[0.01, "a modifier for training"],
    "threshold":[0.05,"When to declare something to be unknown"],
    "model":["Convolutional","Model type [Fully_Connected,Convolutional]"],
    "OOD Type":["Soft","type of out of distribution detection [Soft,Open,Energy,COOL,DOC]"],
    "Dropout":[0.01,"percent of nodes that are skipped per run, larger numbers for more complex models [0,1)"],
    "Datagrouping":["Dendrogramlimit","Datagroup type [ClassChunk,Dendrogramlimit]"],
    "optimizer":opt_func["Adam"],
    "Unknowns":"refer to unknowns.CSV",
    "CLASSES":[15,"Number of classes, do not change"],
    "Temperature":[1,"Energy OOD scaling parameter"],
    "Degree of Overcompleteness": [3,"Parameter for Fitted Learning"],
    "Number of Layers": [2,"Number of layers to add to the base model"],
    "Nodes": [512,"The number of nodes per added layer"],
    "Activation": ["ReLU","The type of activation function to use"],
    "LOOP": [0,"This is a parameter that detumines if we want to loop over the algorithms.\n "\
    "0: no loop, 1:loop through variations of algorithms,thresholds,learning rates, groups and numbers of epochs, \n"\
    "2: Loop while adding more unknowns into the training data (making them knowns) without resetting the model"],
    "Dataset": ["Payload_data_CICIDS2017", "This is what dataset we are using, [Payload_data_CICIDS2017,Payload_data_UNSW]"],
    "loopLevel": [0,"What percentages the model is on"]
}

DOC_kernels = [3,4,5]

#Set Number of classes:
if parameters["Dataset"][0] == "Payload_data_UNSW":
    parameters["CLASSES"][0] = 10


#Dendrogram chunk uses a slightly diffrent output on the model structure. 
# (Also, dendrogram chunk is not working, so don't use it. Possibly related.)
if parameters["Datagrouping"][0] == "DendrogramChunk":
    parameters["CLASSES"][0] = parameters["CLASSES"][0] *32


#Add a value to the dictionary that is the inverse of the unknowns
helper_variables["knowns_clss"] = loopOverUnknowns(helper_variables["unknowns_clss"])


#This is for saving the original number of epochs
num_epochs = parameters["num_epochs"][0]


#This is to test all of the algorithms one after the other. (Loop 1 values)
alg = ["Soft","Open","Energy","COOL","DOC"]
batch = [10,100,1000]
datapoints_per_class = []
thresholds = [0.1,1,10]
learning_rates = [0.1,0.01,0.0001]
activation = ["ReLU", "Tanh", "Sigmoid"]
groups = [[2],[2,3,4,5,6],[1,2,3,4,5,6,7,8]]
if parameters["Dataset"][0] == "Payload_data_CICIDS2017":
    incGroups = [[2,3,4,5,6,7,8,9,10,11,12,13,14],[3,4,5,6,7,8,9,10,11,12,13,14],[4,5,6,7,8,9,10,11,12,13,14],[5,6,7,8,9,10,11,12,13,14],[6,7,8,9,10,11,12,13,14],[7,8,9,10,11,12,13,14],[8,9,10,11,12,13,14],[9,10,11,12,13,14],[10,11,12,13,14],[11,12,13,14],[12,13,14],[13,14],[14]] #This one list is for loop 2. Note: array size should be decreasing.
else:
    incGroups = [[2,3,4,5,6,7,8,9],[3,4,5,6,7,8,9],[4,5,6,7,8,9],[5,6,7,8,9],[6,7,8,9],[7,8,9],[8,9],[9]]
epochs= []
epochs = []


# groups = [list(range(2,parameters["CLASSES"][0]))]
# #Little bit of code that generates incremental numbers of unknowns.
# while len(groups[0])>2:
#     new = groups[0].copy()
#     new.pop(0)
#     new.pop(0)
#     groups.insert(0,new)
# #Little bit of code that generates decrementing numbers of unknowns.
# incGroups = [list(range(2,parameters["CLASSES"][0]))]
# while len(incGroups[-1])>1:
#     new = incGroups[-1].copy()
#     new.pop(0)
#     incGroups.append(new)

#Here is where we remove some of the algorithms if we want to skip them. We could also just remove them from the list above.
#alg.remove("Soft")
alg.remove("Open")
#alg.remove("Energy")
#alg.remove("COOL")
#alg.remove("DOC")


#Optimizer has been removed from the list of things we are changing
optim = [opt_func["Adam"], opt_func["SGD"], opt_func["RMSprop"]]
optim = [opt_func["Adam"]]


#Adds in everything in config:

# #learning_rates.remove(Config.parameters["learningRate"][0])
# learning_rates.insert(0,parameters["learningRate"][0])
# #epochs.remove(Config.parameters["num_epochs"][0])
# epochs.insert(0,parameters["num_epochs"][0])
# groups.insert(0,helper_variables["unknowns_clss"])

# #Always starts with the configured activation type
# alg.remove(parameters["OOD Type"][0])
# alg.insert(0,parameters["OOD Type"][0])

#This is an array to eaiser loop through everything.
loops = [batch,datapoints_per_class,thresholds,learning_rates,epochs,optim,activation,groups]
#loops = [batch,thresholds,datapoints_per_class]
loops2 = ["batch_size","MaxSamples","threshold","learningRate","num_epochs","optimizer","Activation","Unknowns"]
#loops2 = ["batch_size","MaxPerClass","threshold"]
for i in range(len(loops)):
    if loops2[i] == "Unknowns":
        loops[i].insert(0,helper_variables["unknowns_clss"])
    elif loops2[i] == "optimizer":
        loops[i].insert(0,parameters[loops2[i]])
    else:
        loops[i].insert(0,parameters[loops2[i]][0])

#Override the unknowns because model is kept
if parameters["LOOP"][0] == 2:
    helper_variables["unknowns_clss"] = incGroups[0]
    parameters["Unknowns"] = f"{incGroups[0]} Unknowns"
    helper_variables["knowns_clss"] = loopOverUnknowns(helper_variables["unknowns_clss"])
