import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader
import os
from sklearn.metrics import (precision_score, recall_score)
import warnings


# user defined modules
import GPU, FileHandling
from EndLayer import EndLayers
import plots
import Dataload
import cnn

# uncomment this if you are on windows machine.
# hyperpath= r"C:\Users\bgaspard\Desktop\OpenSetPerf\src\main\hyperparam\\"
# unknownpath = r"C:\Users\bgaspard\Desktop\OpenSetPerf\src\main\unknown\\"
# modelsavespath = r"C:\Users\bgaspard\Desktop\OpenSetPerf\src\main\Saves\\"

# Uncomment this if you are on Unix system
hyperpath= "/media/designa/New Volume/OpenSetPerf/src/main/hyperparam/"
unknownpath = "/media/designa/New Volume/OpenSetPerf/src/main/unknown/"
modelsavespath = "/media/designa/New Volume/OpenSetPerf/src/main/Saves/"


def main():
        FileHandling.generateHyperparameters(hyperpath,unknownpath) # generate hyper parameters if not present.
        batch_size,num_workers,attemptLoad,testlen,num_epochs,lr,threshold,unknownVals = FileHandling.readCSVs(hyperpath,unknownpath)
        knownVals = FileHandling.loopOverUnknowns(unknownVals)
        print(knownVals)
        

        class AttackClassification():
            def training_step(self, batch):
                data, labels = batch
                out = cnn.model(data)  # Generate predictions
                if self.end.type == "COOL":
                    labels = self.end.COOL_Label_Mod(labels)
                    out = torch.split(out.unsqueeze(dim=1),15, dim=2)
                    out = torch.cat(out,dim=1)
                #out = DeviceDataLoader(out, device)
                loss = F.cross_entropy(out, labels)  # Calculate loss
                torch.cuda.empty_cache()
                print("loss from training ... " ,loss)
                return loss








    #
    # Y_test = []
    # y_pred =[]


    # device = GPU.get_default_device()
    #
    #
    # train_loader = trainset
    # val_loader = GPU.DeviceDataLoader(validationset, device)
    # test_loader = testset
    #
    #
    # Y_test = []
    # y_pred =[]
    # history_final = []
    # history_final += fit(num_epochs, lr, model, train_loader, val_loader, opt_func)


    # print("all history", history_final)
    # print("y test outside",Y_test)
    # print("y pred outside",y_pred)
    #
    # plots.plot_all_losses(history_final)
    # plots.plot_losses(history_final)
    # plots.plot_accuracies(history_final)
    #
    # y_test, y_pred = plots.convert_to_1d(Y_test,y_pred)
    # #plots.plot_confusion_matrix(y_test,y_pred)
    #
    # cnf_matrix = confusion_matrix(y_test, y_pred)
    # np.set_printoptions(precision=2)
    # class_names = Dataload.get_class_names(knownVals)
    # class_names.append("unknown")
    # plots.plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=False,
    #                       title='Confusion matrix')
    # plt.show()
    #
    # recall = recall_score(y_test,y_pred,average='weighted',zero_division=0)
    # precision = precision_score(y_test,y_pred,average='weighted',zero_division=0)
    # f1 = 2 * (precision * recall) / (precision + recall)
    # # auprc = average_precision_score(y_test, y_pred, average='samples')
    # score_list = [recall,precision,f1]
    # plots.write_hist_to_file(history_final,num_epochs,model.end.type)
    # plots.write_scores_to_file(score_list,num_epochs,model.end.type)
    # print("F-Score : ", f1*100)
    # print("Precision : " ,precision*100)
    # print("Recall : ", recall*100)
    # # print("AUPRC : ", auprc * 100)



if __name__ == '__main__':
    main()




