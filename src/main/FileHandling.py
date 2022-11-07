import os
import pandas as pd

hyperpath= "/Users/bayegaspard/Downloads/OpenSetPerf/src/Sofmax/hyperparam/"
unknowpath = "/Users/bayegaspard/Downloads/OpenSetPerf/src/Sofmax/unknown/"
modelsavespath = "/Users/bayegaspard/Downloads/OpenSetPerf/src/Sofmax/Saves/"
path = ""

def generateHyperparameters(hyperpath,unknowpath):
    if os.path.exists(hyperpath+"hyperParam.csv") and os.path.exists(unknowpath+"unknowns.csv"):
        return
    parameters = {"batch_size":[100, "Number of items per batch"],"num_workers":[6, "Number of threads working on building batches"],"attemptLoad":[0, "0: do not use saves\n1:use saves"],
    "testlength":[1/4, "[0,1) percentage of training to test with"],"num_epochs":[1,"Number of times it trains on the whole trainset"],"learningRate":[0.01, "a modifier for training"],
    "threshold":[0.25,"When to declare something to be unknown"], "optimizer":"Adam", "Unknowns":"refer to unknowns.CSV","CLASSES":[15,"Number of classes, do not change"], 
    "Temperature":[1,"Energy OOD scaling parameter"]}
    param = pd.DataFrame.from_dict(parameters,orient="columns")

    param.to_csv(hyperpath+"hyperParam.csv")
    parameters = {unknowpath+"unknowns":[2,3,13,14]}
    param = pd.DataFrame.from_dict(parameters)
    param.to_csv(unknowpath+"unknowns.csv")


def savePoint(net:AttackClassification, path:str, epoch=0, phase=None):
    if not os.path.exists(path):
        os.mkdir(path)
    torch.save(net.state_dict(),path+f"/Epoch{epoch:03d}.pth")
    if phase is not None:
        file = open("Saves/phase","w")
        file.write(str(phase))
        file.close()

def loadPoint(net:AttackClassification, path:str):
    if not os.path.exists(path):
        os.mkdir(path)
    i = 999
    epochFound = 0
    while i >= 0:
        if os.path.exists(path+f"/Epoch{i:03d}.pth"):
            net.load_state_dict(torch.load(path+f"/Epoch{i:03d}.pth"))
            print(f"Loaded  model /Epoch{i:03d}.pth")
            epochFound = i
            i = -1
        i = i-1
    if i != -2:
        print("No model to load found.")
    elif os.path.exists(modelsavespath+"phase"):
        file = open(modelsavespath+"phase","r")
        phase = file.read()
        file.close()
        return int(phase),epochFound
    return -1, -1


def attemptedLoadcheck(model):
    if attemptLoad and os.path.exists(modelsavespath+"phase"):
        file = open(modelsavespath+"phase","r")
        try:
            startphase = int(file.read())
        except:
            startphase = 0
        file.close()

        model = cnn.Conv1DClassifier()
        # model = nn.DataParallel(model)
        model.to(device)
        _,e = loadPoint(model, modelsavespath+"Saves")
        e = e
    for x in ["Soft","Energy","Open"]:
        phase += 1
        if phase<startphase:
            continue
        elif e==0:
            model = Net()
            model.to(device)
        model.end.type=x
        if x == "Open":
            model.end.prepWeibull(train_loader,device,model)

        Y_test = []
        y_pred =[]
        history_finaltyped = []
        history_finaltyped += fit(num_epochs-e, lr, model, train_loader, val_loader, opt_func)
        plots.store_values(history_finaltyped, y_pred, Y_test, num_epochs, x)
        e=0
        del model
    model = Net()
    model.to(device)
    if attemptLoad:
        loadPoint(model,modelsavespath+"Saves")
    phase += 1
            
def phasecheck(modelsavespath)
    if attemptLoad and os.path.exists(modelsavespath+"phase"):
        file = open(modelsavespath+"phase","w")
        startphase = "0"
        file.close()