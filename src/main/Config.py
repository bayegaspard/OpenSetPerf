import torch
import pandas as pd
import os

def loopOverUnknowns(unknownlist):
    knownVals = list(range(15))
    for un in unknownlist["unknowns"]:
        knownVals.remove(un)
    return knownVals

opt_func = {"Adam":torch.optim.Adam,"SGD":torch.optim.SGD}
helper_variables = {
    "phase" : -1,
    "startphase" : 0,
    "unknowns_clss": {"unknowns":[2,3,4,5,6]},

    "e": 0
}

helper_variables["knowns_clss"] = loopOverUnknowns(helper_variables["unknowns_clss"])
parameters = {
    #These parameters are orginized like this:
    #"ParamName":[Value,"Description"]
    #for a parameter called "ParamName" with a value of Value
    "batch_size":[100, "Number of items per batch"],
    "num_workers":[3, "Number of threads working on building batches"],
    "attemptLoad":[0, "0: do not use saves\n1:use saves"],
    "testlength":[1/4, "[0,1) percentage of training to test with"],
    "MaxPerClass": [10, "Maximum number of samples per class"],
    "num_epochs":[7,"Number of times it trains on the whole trainset"],
    "learningRate":[0.0001, "a modifier for training"],
    "threshold":[0.5,"When to declare something to be unknown"],
    "model":["Convolutional","Model type [Fully_Connected,Convolutional]"],
    "OOD Type":["Soft","type of out of distribution detection [Soft,Open,Energy,COOL,DOC]"],
    "Dropout":[0.01,"percent of nodes that are skipped per run, larger numbers for more complex models [0,1)"],
    "Datagrouping":["Dendrogramlimit","Datagroup type [ClassChunk,Dendrogramlimit]"],
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
