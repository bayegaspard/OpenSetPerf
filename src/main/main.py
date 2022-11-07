import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import Dataload
import pandas as pd
from torch.utils.data import DataLoader
import plots
from EndLayer import EndLayers
import os
from sklearn.metrics import (precision_score, recall_score)
import warnings
import cnn
import GPU


def main():

    trainset = DataLoader(train, batch_size, num_workers=num_workers,shuffle=True,
                        pin_memory=False)  # for faster processing enable pin memory to true and num_workers=4
    validationset = DataLoader(test, batch_size, shuffle=True, num_workers=num_workers,pin_memory=False)
    testset = DataLoader(test, batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)


    print(len(train))
    print(len(test))

    # print(next(iter(testset)))
    # test_features, testset_labels = next(iter(testset))
    # print(f"Feature batch shape: {test_features.size()}")
    # print(f"Labels batch shape: {testset_labels.size()}")
    # img = test_features[0].squeeze()
    # label = testset_labels[:]
    # print("label sss", label)

    Y_test = []
    y_pred =[]


    device = GPU.get_default_device()


    train_loader = trainset
    val_loader = GPU.DeviceDataLoader(validationset, device)
    test_loader = testset


    Y_test = []
    y_pred =[]
    history_final = []
    history_final += fit(num_epochs, lr, model, train_loader, val_loader, opt_func)


    print("all history", history_final)
    print("y test outside",Y_test)
    print("y pred outside",y_pred)

    plots.plot_all_losses(history_final)
    plots.plot_losses(history_final)
    plots.plot_accuracies(history_final)

    y_test, y_pred = plots.convert_to_1d(Y_test,y_pred)
    #plots.plot_confusion_matrix(y_test,y_pred)

    cnf_matrix = confusion_matrix(y_test, y_pred)
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




