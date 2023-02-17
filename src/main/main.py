import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import (precision_score, recall_score, average_precision_score)
import numpy as np
import torch


# user defined modules
import GPU, FileHandling
from EndLayer import EndLayers
import plots
import Dataload
import ModelStruct
import Config
import os
import helperFunctions

root_path = os.getcwd()

if __name__ == "__main__":
    print(torch.__version__)


#useful variables
opt_func = Config.parameters["optimizer"]
device = GPU.get_default_device() # selects a device, cpu or gpu

def run_model():
    #Delete me
    FileHandling.refreshFiles(root_path)

    FileHandling.generateHyperparameters(root_path) # generate hyper parameters if not present.
    batch_size,num_workers,attemptLoad,testlen,num_epochs,lr,threshold,model_type,datagroup,unknownVals = FileHandling.readCSVs(root_path)
    knownVals = Config.helper_variables["knowns_clss"]
    # print(knownVals)
    # print(unknownVals)
    model_list = {"Convolutional":ModelStruct.Conv1DClassifier,"Fully_Connected":ModelStruct.FullyConnected}
    model = model_list[model_type]() # change index to select a specific architecture. 0=conv1d ad 1=fully connected
    model = ModelStruct.ModdedParallel(model)
    #model.to(device)
    #model.device = device
    model.end.type = Config.parameters["OOD Type"][0]
    model.end.cutoff = threshold

    train, test, val = FileHandling.checkAttempLoad(root_path)


    trainset = DataLoader(train, batch_size, num_workers=num_workers,shuffle=True,
            pin_memory=False)  # for faster processing enable pin memory to true and num_workers=4
    validationset = DataLoader(val, batch_size, shuffle=True, num_workers=num_workers,pin_memory=False)
    testset = DataLoader(test, batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)

    print("length of train",len(train),"\nlength of test",len(test))

    #train_loader = trainset
    #val_loader = validationset
    train_loader =  GPU.DeviceDataLoader(trainset, device)
    val_loader = GPU.DeviceDataLoader(validationset, device)
    test_loader = GPU.DeviceDataLoader(testset, device)

    #print("Test1")

    history_final = []
    model.end.prepWeibull(train_loader,device,model)
    history_final += model.fit(num_epochs, lr, train_loader, test_loader,val_loader, opt_func=opt_func)

    #Validation values
    f1, recall, precision, accuracy = helperFunctions.getFscore(model.store)
    FileHandling.addMeasurement("Val_F1",f1)
    FileHandling.addMeasurement("Val_Recall",recall)
    FileHandling.addMeasurement("Val_Precision",precision)
    FileHandling.addMeasurement("Val_Accuracy",accuracy)


    model.storeReset()
    model.eval()
    model.evaluate(test_loader)
    # epochs, lr, model, train_loader, val_loader, opt_func

    #print("Test2")

    if not Config.parameters["LOOP"][0]:
        plots.plot_all_losses(history_final)
        plots.plot_losses(history_final)
        plots.plot_accuracies(history_final)

    
    # print("y len and pred",len(y_pred),y_pred)
    # print("y len and test", len(y_test),y_test)
#plots.plot_confusion_matrix(y_test,y_pred)

    #print("Test3")

    np.set_printoptions(precision=1)
    #class_names = Dataload.get_class_names(knownVals) #+ Dataload.get_class_names(unknownVals)
    #class_names.append("Unknown")
    # class_names = Dataload.get_class_names(range(15))
    # for x in unknownVals:
    #     class_names[x] = class_names[x]+"*"
    # class_names.append("*Unknowns")
    # print("class names", class_names)
    #cnf_matrix = plots.confusionMatrix(y_true.copy(), y_pred.copy(), y_tested_against.copy()) 

    #print("Test4")

    
    f1, recall, precision, accuracy = helperFunctions.getFscore(model.store)
    FileHandling.addMeasurement("Test_F1",f1)
    FileHandling.addMeasurement("Test_Recall",recall)
    FileHandling.addMeasurement("Test_Precision",precision)
    FileHandling.addMeasurement("Test_Accuracy",accuracy)
    FileHandling.create_params_Fscore(root_path,f1)

    #plots.plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
    #                title='Confusion matrix', knowns = knownVals)
    if not Config.parameters["LOOP"][0]:
        plt.show()

    
    
    #auprc = average_precision_score(y_compaire, y_pred, average='weighted')
    score_list = [recall,precision,f1]
    FileHandling.write_hist_to_file(history_final,num_epochs,model.end.type)
    FileHandling.write_scores_to_file(score_list,num_epochs,model.end.type)
    print("Type : ",model.end.type)
    print(f"Now changing : {plots.name_override}")
    print(f"F-Score : {f1*100:.2f}%")
    print(f"Precision : {precision*100:.2f}%")
    print(f"Recall : {recall*100:.2f}%")

    if Config.parameters["LOOP"][0]:
        net = model_list[model_type]()
        model.thresholdTest(val_loader)
    # print("AUPRC : ", auprc * 100)

def main():
    global root_path
    while (os.path.basename(root_path) == "main.py" or os.path.basename(root_path) == "main" or os.path.basename(root_path) == "src"):
        #checking that you are running from the right folder.
        print(f"Please run this from the source of the repository not from {os.path.basename(root_path)}. <---- Look at this!!!!")
        os.chdir("..")
        root_path=os.getcwd()

    plots.name_override = "Config File settings"

    run_model()

    step = (0,0,0) #keeps track of what is being updated.
    while Config.parameters["LOOP"][0]:
        step = helperFunctions.testRotate(step)
        if step:
            plt.clf()
            plots.name_override = helperFunctions.getcurrentlychanged(step)
            plt.figure(figsize=(4,4))
            print(f"Now changing: {plots.name_override}")
            run_model()


if __name__ == '__main__':
    main()




