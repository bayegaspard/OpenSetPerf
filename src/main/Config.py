import torch
import pandas as pd
import os

def loopOverUnknowns(unknownlist):
    print(torch.__version__)
    knownVals = list(range(15))
    for un in unknownlist["unknowns"]:
        knownVals.remove(un)
    return knownVals

opt_func = {"Adam":torch.optim.Adam,"SGD":torch.optim.SGD}
helper_variables = {
    "phase" : -1,
    "startphase" : 0,
    "unknowns_clss": {"unknowns":[2,3]},

    "e": 0
}

helper_variables["knowns_clss"] = loopOverUnknowns(helper_variables["unknowns_clss"])
parameters = {
    "batch_size":[100, "Number of items per batch"],
    "num_workers":[0, "Number of threads working on building batches"],
    "attemptLoad":[0, "0: do not use saves\n1:use saves"],
    "testlength":[1/4, "[0,1) percentage of training to test with"],
    "num_epochs":[3,"Number of times it trains on the whole trainset"],
    "learningRate":[0.001, "a modifier for training"],
    "threshold":[0.5,"When to declare something to be unknown"],
    "model":["Convolutional","Model type [Fully_Connected,Convolutional]"],
    "OOD Type":["DOC","type of out of distribution detection [Soft,Open,Energy,COOL]"],
    "Dropout":[0.01,"percent of nodes that are skipped per run, larger numbers for more complex models [0,1)"],
    "Datagrouping":["ClassChunk","Datagroup type [ClassChunk,DendrogramChunk]"],
    "optimizer":opt_func["Adam"],
    "Unknowns":"refer to unknowns.CSV",
    "CLASSES":[15,"Number of classes, do not change"],
    "Temperature":[1,"Energy OOD scaling parameter"],
    "Degree of Overcompleteness": [3,"Parameter for Fitted Learning"]
}

if parameters["Datagrouping"][0] == "DendrogramChunk":
    parameters["CLASSES"][0] = parameters["CLASSES"][0] *32


