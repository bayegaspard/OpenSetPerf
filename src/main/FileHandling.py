import os
import Config
import pandas as pd
import torch


# hyperpath= r"C:\Users\bgaspard\Desktop\OpenSetPerf\src\main\hyperparam\\"
# unknownpath = r"C:\Users\bgaspard\Desktop\OpenSetPerf\src\main\unknown\\"
# modelsavespath = r"C:\Users\bgaspard\Desktop\OpenSetPerf\src\main\Saves\\"

def generateHyperparameters(hyperpath,unknownpath):
    if os.path.exists(hyperpath+"hyperParam.csv") and os.path.exists(unknownpath+"unknowns.csv"):
        print("hyperparam.csv and unknown.csv files exist")
    else:
        print("One or all the files does not exist, generating one based on config file settings ....")
        parameters = Config.parameters
        print(parameters)
        param = pd.DataFrame.from_dict(parameters, orient="columns")
        param.to_csv(hyperpath+"hyperParam.csv")
        parameters = {"unknowns": [2, 3, 13, 14]}
        param = pd.DataFrame.from_dict(parameters)
        param.to_csv(unknownpath+"unknowns.csv")
        print("Files created successfully !")



def readCSVs(hyperpath,unknownpath):
        param = pd.read_csv(hyperpath+"hyperParam.csv")
        batch_size = int(param["batch_size"][0])
        num_workers = int(param["num_workers"][0])
        attemptLoad = int(param["attemptLoad"][0])
        testlen = float(param["testlength"][0])
        num_epochs = int(param["num_epochs"][0])
        lr = float(param["learningRate"][0])
        threshold = float(param["threshold"][0])
        param = pd.read_csv(unknownpath+"unknowns.csv")
        unknownVals = param["unknowns"].to_list()

        return batch_size,num_workers,attemptLoad,testlen,num_epochs,lr,threshold,unknownVals


def loopOverUnknowns(unknownlist):
    print(torch.__version__)
    knownVals = list(range(15))
    for un in unknownlist:
        knownVals.remove(un)
    return knownVals

# generateHyperparameters(hyperpath,unknownpath)

# def readFromFiles(path):


# def savePoint(net:AttackClassification, path:str, epoch=0, phase=None):
#     if not os.path.exists(path):
#         os.mkdir(path)
#     torch.save(net.state_dict(),path+f"/Epoch{epoch:03d}.pth")
#     if phase is not None:
#         file = open("Saves/phase","w")
#         file.write(str(phase))
#         file.close()

# def loadPoint(net:AttackClassification, path:str):
#     if not os.path.exists(path):
#         os.mkdir(path)
#     i = 999
#     epochFound = 0
#     while i >= 0:
#         if os.path.exists(path+f"/Epoch{i:03d}.pth"):
#             net.load_state_dict(torch.load(path+f"/Epoch{i:03d}.pth"))
#             print(f"Loaded  model /Epoch{i:03d}.pth")
#             epochFound = i
#             i = -1
#         i = i-1
#     if i != -2:
#         print("No model to load found.")
#     elif os.path.exists(modelsavespath+"phase"):
#         file = open(modelsavespath+"phase","r")
#         phase = file.read()
#         file.close()
#         return int(phase),epochFound
#     return -1, -1
#
#
# def attemptedLoadcheck(model):
#     if attemptLoad and os.path.exists(modelsavespath+"phase"):
#         file = open(modelsavespath+"phase","r")
#         try:
#             startphase = int(file.read())
#         except:
#             startphase = 0
#         file.close()
#
#         model = cnn.Conv1DClassifier()
#         # model = nn.DataParallel(model)
#         model.to(device)
#         _,e = loadPoint(model, modelsavespath+"Saves")
#         e = e
#     for x in ["Soft","Energy","Open"]:
#         phase += 1
#         if phase<startphase:
#             continue
#         elif e==0:
#             model = Net()
#             model.to(device)
#         model.end.type=x
#         if x == "Open":
#             model.end.prepWeibull(train_loader,device,model)
#
#         Y_test = []
#         y_pred =[]
#         history_finaltyped = []
#         history_finaltyped += fit(num_epochs-e, lr, model, train_loader, val_loader, opt_func)
#         plots.store_values(history_finaltyped, y_pred, Y_test, num_epochs, x)
#         e=0
#         del model
#     model = Net()
#     model.to(device)
#     if attemptLoad:
#         loadPoint(model,modelsavespath+"Saves")
#     phase += 1
#
# def phasecheck(modelsavespath)
#     if attemptLoad and os.path.exists(modelsavespath+"phase"):
#         file = open(modelsavespath+"phase","w")
#         startphase = "0"
#         file.close()