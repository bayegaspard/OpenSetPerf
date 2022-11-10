import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import (precision_score, recall_score)
import numpy as np


# user defined modules
import GPU, FileHandling
from EndLayer import EndLayers
import plots
import Dataload
import cnn
import Config


# Uncomment this if you are on Unix system
root_path= ""


#uncomment this and change your root directory if you are using windows
#root_path = r"C:\\Users\\bgaspard\\Desktop\\OpenSetPerf\\"

#useful variables
opt_func = Config.parameters["optimizer"]
device = GPU.get_default_device() # selects a device, cpu or gpu

def main():

        FileHandling.generateHyperparameters(root_path) # generate hyper parameters if not present.
        batch_size,num_workers,attemptLoad,testlen,num_epochs,lr,threshold,unknownVals = FileHandling.readCSVs(root_path)
        knownVals = FileHandling.loopOverUnknowns(unknownVals)
        # print(knownVals)
        # print(unknownVals)
        model_conv1d = cnn.AttackTrainingClassification()
        model_fully_connected = cnn.FullyConnected()
        model_list = [model_conv1d,model_fully_connected]
        model = model_list[0] # change index to select a specific architecture. 0=conv1d ad 1=fully connected
        #model = nn.DataParallel(model)
        model.to(device)
        model.device = device

        train, test = FileHandling.checkAttempLoad(root_path)

        trainset = DataLoader(train, batch_size, num_workers=Config.parameters["num_workers"],shuffle=True,
                pin_memory=False)  # for faster processing enable pin memory to true and num_workers=4
        validationset = DataLoader(test, batch_size, shuffle=True, num_workers=Config.parameters["num_workers"],pin_memory=False)
        testset = DataLoader(test, batch_size, shuffle=True, num_workers=Config.parameters["num_workers"], pin_memory=False)

        print("length of train",len(train),"\nlength of test",len(test))

        Y_test = []
        y_pred = []


        train_loader =  GPU.DeviceDataLoader(trainset, device)
        val_loader = GPU.DeviceDataLoader(validationset, device)
        test_loader = testset


        Y_test = []
        y_pred =[]
        history_final = []
        history_final += model.fit(num_epochs, lr, train_loader, val_loader, opt_func=opt_func)
        # epochs, lr, model, train_loader, val_loader, opt_func

        plots.plot_all_losses(history_final)
        plots.plot_losses(history_final)
        plots.plot_accuracies(history_final)

        y_test, y_pred = plots.convert_to_1d(Y_test,y_pred)
    #plots.plot_confusion_matrix(y_test,y_pred)

        cnf_matrix = plots.confusionMatrix(y_test, y_pred)
        np.set_printoptions(precision=2)
        class_names = Dataload.get_class_names(knownVals)
        class_names.append("unknown")
        plots.plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=False,
                          title='Confusion matrix')
        plt.show()

        recall = recall_score(y_test,y_pred,average='weighted',zero_division=0)
        precision = precision_score(y_test,y_pred,average='weighted',zero_division=0)
        f1 = 2 * (precision * recall) / (precision + recall)
    # auprc = average_precision_score(y_test, y_pred, average='samples')
        score_list = [recall,precision,f1]
        plots.write_hist_to_file(history_final,num_epochs,model.end.type)
        plots.write_scores_to_file(score_list,num_epochs,model.end.type)
        print("F-Score : ", f1*100)
        print("Precision : " ,precision*100)
        print("Recall : ", recall*100)
    # print("AUPRC : ", auprc * 100)



if __name__ == '__main__':
    main()




