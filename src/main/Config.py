import torch
import pandas as pd
import os
import sys
import argparse


if __name__ != "Config":
    # if "Config" in sys.modules:
    #     class doubleImport(ImportError):
    #         """
    #         Config was imported using a different path after it has already been imported.
    #         This causes problems when Config is modified.
    #         """
    #         pass
    #     raise doubleImport
    print(f"A POSSIBLE PROBLEM HAS OCCURED, Config was loaded improperly, from {__name__} instead of directly\
    this might break some global variables by having two copies",file=sys.stderr)

#TODO: Rework config so that it is less janky and uses less bad practices of global variables. 
# Possibly by moving HelperFunctions Loop functions to outside of the program 
# and just using the command line parser for the individual sections.

#This config file is mainly used as global variables for the rest of the program.
#It should only be modified by the loop commands in helperfunctions


def loopOverUnknowns(unknownlist=False):
    """
    Given a list of unknowns (integers 0-14) this will create a list of knowns (the inverted list).
    """
    if unknownlist == False:
        unknownlist = parameters["Unknowns_clss"][0]
    knownVals = list(range(parameters["CLASSES"][0]))
    notused = unknownlist + UnusedClasses
    notused.sort()
    for un in notused:
        if un in knownVals:
            knownVals.remove(un)
    
    if len(unknownlist) > parameters["CLASSES"][0] -3:
        print("Too many unknowns, some algorithms might not work")
    if len(knownVals)<2:
        print("Too few knowns, things might break")
    parameters["Unknowns"] = f"{len(unknownlist)} Unknowns"
    parameters["Unknowns_clss"] = [unknownlist,"Values used for testing"]
    parameters["Knowns_clss"] = [knownVals,"Values used for training"]
    return knownVals

#This is the diffrent optimization functions
opt_func = {"Adam":torch.optim.Adam,"SGD":torch.optim.SGD, "RMSprop":torch.optim.RMSprop}


#Here are all of the paremeters for the model.
parameters = {
    #These parameters are orginized like this:
    #"ParamName":[Value,"Description",[possible values]]
    #for a parameter called "ParamName" with a value of Value
    "batch_size":[100000, "Number of items per batch"],
    "num_workers":[14, "Number of threads working on building batches"],
    "attemptLoad":[0, "0: do not use saves\n1:use saves"],
    "testlength":[1/4, "[0,1) percentage of training to test with"],
    "Mix unknowns and validation": [1,"0 or 1, 0 means that the test set is purely unknowns and 1 means that the testset is the validation set plus unknowns (for testing)"],
    "MaxPerClass": [1000, "Maximum number of samples per class\n if Dataloader_Variation is Cluster and this value is a float it interprets it as the maximum percentage of the class instead."],
    "num_epochs":[150,"Number of times it trains on the whole trainset"],
    "learningRate":[0.001, "a modifier for training"],
    "threshold":[0.5,"When to declare something to be unknown"],
    "model":["Convolutional","Model type",["Fully_Connected","Convolutional"]],
    "OOD Type":["Soft","type of out of distribution detection", ["Soft","Open","Energy","COOL","DOC","iiMod"]],
    "Dropout":[0.1,"percent of nodes that are skipped per run, larger numbers for more complex models [0,1)"],
    "Dataloader_Variation":["Standard","Defines the style of Dataloader used. This affects sampling from the dataset", ["Standard","Cluster","Flows"]],
    "optimizer":opt_func["Adam"],
    "Unknowns":["UNUSED"],
    "Unknowns_clss": [[7,8,9],"Class indexes used as unknowns."],
    "CLASSES":[15,"Number of classes, do not change"],
    "Temperature":[1,"Energy OOD scaling parameter"],
    "Degree of Overcompleteness": [3,"Parameter for Fitted Learning"],
    "Number of Layers": [3,"Number of layers to add to the base model"],
    "Nodes": [512,"The number of nodes per added layer"],
    "Activation": ["Leaky","The type of activation function to use",["ReLU", "Tanh", "Sigmoid","Leaky"]],
    "LOOP": [0,"This is a parameter that determines if we want to loop over the algorithms.\n "\
    "0: no loop, 1:loop through variations of algorithms,thresholds,learning rates, groups and numbers of epochs, \n"\
    "2: Loop while adding more unknowns into the training data (making them knowns) without resetting the model, \n"\
    "3: Loop through different data distributions without training the model.\n"\
    "4: Loop through predefined hyperparameters found in datasets/hyperparamList.csv"],
    "Dataset": ["Payload_data_CICIDS2017", "This is what dataset we are using,", ["Payload_data_CICIDS2017","Payload_data_UNSW"]],
    "SchedulerStepSize": [10, "This is how often the scheduler takes a step, 3 means every third epoch"],
    "SchedulerStep": [0.9,"This is how big a step the scheduler takes, leave 0 for no step"]
}


#Argparse tutorial: https://docs.python.org/3/howto/argparse.html 
parser = argparse.ArgumentParser()
for x in parameters.keys():
    if x in ["batch_size","num_workers","MaxPerClass","num_epochs","Degree of Overcompleteness","Number of Layers","Nodes","SchedulerStepSize"]:
        parser.add_argument(f"--{x}",type=int,default=parameters[x][0],help=parameters[x][1],required=False)
    if x in ["testlength","learningRate","threshold","Dropout","Temperature","SchedulerStep"]:
        parser.add_argument(f"--{x}",type=float,default=parameters[x][0],help=parameters[x][1],required=False)
    if x in ["attemptLoad","Mix unknowns and validation"]:
        parser.add_argument(f"--{x}",type=int,choices=[1,0],default=parameters[x][0],help=parameters[x][1],required=False)
    if x in ["LOOP"]:
        parser.add_argument(f"--{x}",type=int,choices=[0,1,2,3,4],default=parameters[x][0],help=parameters[x][1],required=False)
    if x in ["model","OOD Type","Dataloader_Variation","Activation","Dataset"]:
        parser.add_argument(f"--{x}",choices=parameters[x].pop(),default=parameters[x][0],help=parameters[x][1],required=False)
    if x in ["Unknowns_clss"]:
        parser.add_argument(f"--{x}",default=f"{parameters[x][0]}",help=parameters[x][1],required=False)
if "pytest" not in sys.modules: #The argument parser appears to have issues with the pytest tests. I have no idea why.
    args = parser.parse_args()
    for x in args._get_kwargs():
        parameters[x[0]][0] = x[1]

if isinstance(parameters["Unknowns_clss"][0],str):
    if len(parameters["Unknowns_clss"][0])>0 and len(parameters["Unknowns_clss"][0])!=2: #Not sure why I need this specifier but it breaks if the default is []
        # print(len(parameters["Unknowns_clss"][0]))
        parameters["Unknowns_clss"][0] = [int(y) for y in parameters["Unknowns_clss"][0].removesuffix("]").removeprefix("[").split(sep=",")]
    else:
        parameters["Unknowns_clss"][0] = []


DOC_kernels = [3,4,5]

#Set Number of classes:
if parameters["Dataset"][0] == "Payload_data_UNSW":
    parameters["CLASSES"][0] = 10
    UnusedClasses = []
else:
    UnusedClasses = [8,9,10]
UnusedClasses = []

#Dendrogram chunk uses a slightly diffrent output on the model structure.
# (Also, dendrogram chunk is not working, so don't use it. Possibly related.)
if parameters["Dataloader_Variation"][0] == "Old_Cluster":
    parameters["CLASSES"][0] = parameters["CLASSES"][0] *32


#Add a value to the dictionary that is the inverse of the unknowns
loopOverUnknowns()


#This is for saving the original number of epochs
num_epochs = parameters["num_epochs"][0]


#This is to test all of the algorithms one after the other. (Loop 1 values)
alg = ["Soft","Open","Energy","COOL","DOC","iiMod"]
batch = [100,1000,10000,100000]
datapoints_per_class = [10,100,1000]
thresholds = [0.1,1,10]
thresholds = [30,20,15,5]
thresholds = [parameters["threshold"][0]]
learning_rates = [0.1,0.01,0.001,0.0001]
activation = ["ReLU", "Tanh", "Sigmoid","Leaky"]
groups = [[],[2],[2,3],[2,3,4],[2,3,4,5],[2,3,4,5,6],[2,3,4,5,6,7],[1,2,3,4,5,6,7],[1,2,3,4,5,6,7,8]]
#groups = [[7,8,9]]
if parameters["Dataset"][0] == "Payload_data_CICIDS2017":
    incGroups = [[2,3,4,5,6,7,8,9,10,11,12,13,14],[3,4,5,6,7,8,9,10,11,12,13,14],[4,5,6,7,8,9,10,11,12,13,14],[5,6,7,8,9,10,11,12,13,14],[6,7,8,9,10,11,12,13,14],[7,8,9,10,11,12,13,14],[8,9,10,11,12,13,14],[9,10,11,12,13,14],[10,11,12,13,14],[11,12,13,14],[12,13,14],[13,14],[14]] 
#This one list is for loop 2. Note: array size should be decreasing.
else:
    incGroups = [[2,3,4,5,6,7,8,9],[3,4,5,6,7,8,9],[4,5,6,7,8,9],[5,6,7,8,9],[6,7,8,9],[7,8,9],[8,9],[9]]
epochs= []
epochs = [1,10,100,150]


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
# alg.remove("Soft")
# alg.remove("Open")
# alg.remove("Energy")
# alg.remove("COOL")
# alg.remove("DOC")
# alg.remove("iiMod")

#it causes problems if you dont start at the start of the loop.
if parameters["LOOP"][0] == 1:
    parameters["OOD Type"][0] = alg[0]


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
loops = [batch,learning_rates,activation,["Standard","Cluster"],groups]
# loops = [groups]
loops2 = ["batch_size","learningRate","Activation","Dataloader_Variation","Unknowns"]
# loops2 = ["Unknowns"]
for i in range(len(loops)):
    if loops2[i] == "Unknowns":
        loops[i].insert(0,parameters["Unknowns_clss"][0])
    elif loops2[i] == "optimizer":
        loops[i].insert(0,parameters[loops2[i]])
    elif loops2[i] == "None":
        pass
    else:
        loops[i].insert(0,parameters[loops2[i]][0])

#Override the unknowns because model is kept
if parameters["LOOP"][0] == 2:
    parameters["Unknowns_clss"][0] = incGroups[0]
    parameters["Unknowns"] = f"{incGroups[0]} Unknowns"
    parameters["Knowns_clss"][0] = loopOverUnknowns()


#This controls all of the save data (so that we can run small tests without changing the nice files.)
unit_test_mode = False

use_alg_thesholds =False
def algorithmSpecificSettings(alg="None"):
    if use_alg_thesholds == False:
        return
    if alg == "None":
        alg = parameters["OOD Type"][0]
    
    
    # match alg:
    if alg == "Soft":
        pass
    if alg == "Open":
        parameters["threshold"][0] = 0.8
    if alg == "Energy":
        parameters["threshold"][0] = 0.474
    if alg == "COOL":
        parameters["threshold"][0] = 0.516034961
    if alg == "DOC":
        parameters["threshold"][0] = 0.06449493
    if alg == "iiMod":
        parameters["threshold"][0] = 102064.4453
    
if parameters["LOOP"][0] == 3:
    # parameters["num_epochs"][0] = 0
    parameters["loopLevel"] = [0,"What percentages the model is on"]
    parameters["MaxSamples"] = [parameters["MaxPerClass"][0], "Max number of samples total"]


#Getting version number
#https://gist.github.com/sg-s/2ddd0fe91f6037ffb1bce28be0e74d4e
f = open("build_number.txt","r")
parameters["Version"] = [f.read(),"The version number"]

save_as_tensorboard = True
datasetRandomOffset =True
