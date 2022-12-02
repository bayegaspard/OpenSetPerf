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
import cnn
import Config
import os



root_path = os.getcwd()




#useful variables
opt_func = Config.parameters["optimizer"]
device = GPU.get_default_device() # selects a device, cpu or gpu

def main():
        #Delete me
        FileHandling.refreshFiles(root_path)

        FileHandling.generateHyperparameters(root_path) # generate hyper parameters if not present.
        batch_size,num_workers,attemptLoad,testlen,num_epochs,lr,threshold,model_type,unknownVals = FileHandling.readCSVs(root_path)
        knownVals = FileHandling.loopOverUnknowns(unknownVals)
        # print(knownVals)
        # print(unknownVals)
        model_conv1d = cnn.Conv1DClassifier()
        model_fully_connected = cnn.FullyConnected()
        model_list = {"Convolutional":model_conv1d,"Fully_Connected":model_fully_connected}
        model = model_list[model_type] # change index to select a specific architecture. 0=conv1d ad 1=fully connected
        model = cnn.ModdedParallel(model)
        model.to(device)
        model.device = device
        model.end.type = "COOL"
        model.end.cutoff = threshold

        train, test = FileHandling.checkAttempLoad(root_path)


        trainset = DataLoader(train, batch_size, num_workers=num_workers,shuffle=True,
                pin_memory=False)  # for faster processing enable pin memory to true and num_workers=4
        validationset = DataLoader(test, batch_size, shuffle=True, num_workers=num_workers,pin_memory=False)
        testset = DataLoader(test, batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)

        print("length of train",len(train),"\nlength of test",len(test))

        train_loader =  GPU.DeviceDataLoader(trainset, device)
        val_loader = GPU.DeviceDataLoader(validationset, device)
        test_loader = testset

        history_final = []
        model.end.prepWeibull(train_loader,device,model)
        history_final += model.fit(num_epochs, lr, train_loader, val_loader, opt_func=opt_func)
        # epochs, lr, model, train_loader, val_loader, opt_func

        plots.plot_all_losses(history_final)
        plots.plot_losses(history_final)
        plots.plot_accuracies(history_final)

        y_pred,y_test,y_compaire = model.store
        y_test = y_test.to(torch.int).tolist()
        y_pred = y_pred.to(torch.int).tolist()
        y_compaire = y_compaire.to(torch.int).tolist()
        print("y len and pred",len(y_pred),y_pred)
        print("y len and test", len(y_test),y_test)
    #plots.plot_confusion_matrix(y_test,y_pred)

        
        np.set_printoptions(precision=1)
        #class_names = Dataload.get_class_names(knownVals) #+ Dataload.get_class_names(unknownVals)
        #class_names.append("Unknown")
        class_names = Dataload.get_class_names(range(15))
        for x in unknownVals:
            class_names[x] = class_names[x]+"*"
        class_names.append("*Unknowns")
        print("class names", class_names)
        cnf_matrix = plots.confusionMatrix(y_test, y_pred)
        plots.plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='Confusion matrix', knowns = knownVals)
        plt.show()

        recall = recall_score(y_compaire,y_pred,average='weighted',zero_division=0)
        precision = precision_score(y_compaire,y_pred,average='weighted',zero_division=0)
        f1 = 2 * (precision * recall) / (precision + recall)
        FileHandling.create_params_Fscore(root_path,f1)
        #auprc = average_precision_score(y_compaire, y_pred, average='weighted')
        score_list = [recall,precision,f1]
        FileHandling.write_hist_to_file(history_final,num_epochs,model.end.type)
        FileHandling.write_scores_to_file(score_list,num_epochs,model.end.type)
        print("F-Score : ", f1*100)
        print("Precision : " ,precision*100)
        print("Recall : ", recall*100)
    # print("AUPRC : ", auprc * 100)



if __name__ == '__main__':
    while (os.path.basename(root_path) == "main.py" or os.path.basename(root_path) == "main" or os.path.basename(root_path) == "src"):
        #checking that you are running from the right folder.
        print("Please run this from the source of the repository.")
        os.chdir("..")
        root_path=os.getcwd()

    main()




