import torch
import pandas as pd
import os

def loopOverUnknowns(unknownlist):
    knownVals = list(range(15))
    for un in unknownlist:
        knownVals.remove(un)
    return knownVals

opt_func = {"Adam":torch.optim.Adam,"SGD":torch.optim.SGD, "RMSprop":torch.optim.RMSprop}
helper_variables = {
    "phase" : -1,
    "startphase" : 0,
    "unknowns_clss": [0,1,2,3,4,5,6,11,12,13,14],

    "e": 0
}

helper_variables["knowns_clss"] = loopOverUnknowns(helper_variables["unknowns_clss"])
parameters = {
    #These parameters are orginized like this:
    #"ParamName":[Value,"Description"]
    #for a parameter called "ParamName" with a value of Value
    "batch_size":[100, "Number of items per batch"],
    "num_workers":[0, "Number of threads working on building batches"],
    "attemptLoad":[0, "0: do not use saves\n1:use saves"],
    "testlength":[1/4, "[0,1) percentage of training to test with"],
    "MaxPerClass": [200, "Maximum number of samples per class"],
    "num_epochs":[1,"Number of times it trains on the whole trainset"],
    "learningRate":[0.0001, "a modifier for training"],
    "threshold":[0.5,"When to declare something to be unknown"],
    "model":["Convolutional","Model type [Fully_Connected,Convolutional]"],
    "OOD Type":["DOC","type of out of distribution detection [Soft,Open,Energy,COOL,DOC]"],
    "Dropout":[0.01,"percent of nodes that are skipped per run, larger numbers for more complex models [0,1)"],
    "Datagrouping":["ClassChunk","Datagroup type [ClassChunk,Dendrogramlimit]"],
    "optimizer":opt_func["Adam"],
    "Unknowns":"refer to unknowns.CSV",
    "CLASSES":[15,"Number of classes, do not change"],
    "Temperature":[1,"Energy OOD scaling parameter"],
    "Degree of Overcompleteness": [3,"Parameter for Fitted Learning"],
    "Number of Layers": [1,"Number of layers to add to the base model"],
    "Nodes": [256,"The number of nodes per added layer"],
    "Activation": ["ReLU","The type of activation function to use"],
    "LOOP": [1,"This is a parameter that detumines if we want to loop over the algorithms."]
}

if parameters["Datagrouping"][0] == "DendrogramChunk":
    parameters["CLASSES"][0] = parameters["CLASSES"][0] *32


#This is for saving the original number of epochs
num_epochs = parameters["num_epochs"][0]


#This is to test all of the algorithms one after the other
alg = ["Soft","Open","Energy","COOL","DOC"]
thresholds = [0.1,0.5,0.75,0.99,1.1,2,5,10]
learning_rates = [1,0.1,0.001,0.0001,0.00001,0.000001,0.0000001,0.00000001]
activation = ["ReLU", "Tanh", "Sigmoid","Leaky","Elu","PRElu","Softplus","Softmax"]
groups = [[2],[2,3,6],[2,3,4,5,6],[2,3,4,5,6,7,11],[2,3,4,5,6,7,11,12,14],[2,3,4,5,6,7,8,9,11,12,14],[2,3,4,5,6,7,8,9,10,11,12,13,14],[1,2,3,4,5,6,7,8,9,10,11,12,13,14]]
incGroups = [[10,11,12,13,14],[11,12,13,14],[12,13,14],[13,14],[14],[]]
epochs= []
epochs = [1,2,5,10,25,50,100,200]


#alg.remove("Soft")
alg.remove("Open")
alg.remove("Energy")
#alg.remove("COOL")


#Optimizer has been removed from the list of things we are changing
optim = [opt_func["Adam"], opt_func["SGD"], opt_func["RMSprop"]]
optim = [opt_func["Adam"]]


#Add in everything in config.

#learning_rates.remove(Config.parameters["learningRate"][0])
learning_rates.insert(0,parameters["learningRate"][0])
#epochs.remove(Config.parameters["num_epochs"][0])
epochs.insert(0,parameters["num_epochs"][0])
groups.insert(0,helper_variables["unknowns_clss"])

#Always starts with the configured activation type
alg.remove(parameters["OOD Type"][0])
alg.insert(0,parameters["OOD Type"][0])

loops = [learning_rates,epochs,optim,activation,groups]
loops2 = ["learningRate","num_epochs","optimizer","Activation","Unknowns"]