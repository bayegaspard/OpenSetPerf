import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import Dataload
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import (precision_score, recall_score,
                             f1_score, confusion_matrix, average_precision_score)
import itertools
import numpy as np
import itertools
import pandas as pd
import Config

# epoch = 20 history_final [{'val_loss': 3.6822474002838135, 'val_acc': 0.29999998211860657, 'train_loss': 11.54269027709961}, {'val_loss': 1.771144151687622, 'val_acc': 0.459090918302536, 'train_loss': 2.622060537338257}, {'val_loss': 1.49733304977417, 'val_acc': 0.47272729873657227, 'train_loss': 1.668800711631775}, {'val_loss': 1.4760304689407349, 'val_acc': 0.5136363506317139, 'train_loss': 1.4828541278839111}, {'val_loss': 1.4107227325439453, 'val_acc': 0.5590909719467163, 'train_loss': 1.355711579322815}, {'val_loss': 1.3187934160232544, 'val_acc': 0.5363636612892151, 'train_loss': 1.436032772064209}, {'val_loss': 1.171736717224121, 'val_acc': 0.550000011920929, 'train_loss': 1.3234833478927612}, {'val_loss': 1.09578275680542, 'val_acc': 0.5954546332359314, 'train_loss': 1.1878279447555542}, {'val_loss': 0.9965360164642334, 'val_acc': 0.663636326789856, 'train_loss': 1.134917140007019}, {'val_loss': 1.0754733085632324, 'val_acc': 0.6136363744735718, 'train_loss': 1.0401008129119873}, {'val_loss': 1.015233039855957, 'val_acc': 0.668181836605072, 'train_loss': 1.0170620679855347}, {'val_loss': 0.9538360834121704, 'val_acc': 0.6318181753158569, 'train_loss': 0.9501714706420898}, {'val_loss': 0.9685512185096741, 'val_acc': 0.6590909361839294, 'train_loss': 0.9937074780464172}, {'val_loss': 0.9329931139945984, 'val_acc': 0.7090909481048584, 'train_loss': 0.9353311061859131}, {'val_loss': 0.8152506351470947, 'val_acc': 0.7272727489471436, 'train_loss': 1.0522938966751099}, {'val_loss': 0.7926329970359802, 'val_acc': 0.7409090399742126, 'train_loss': 0.9276139736175537}, {'val_loss': 0.790667712688446, 'val_acc': 0.7000000476837158, 'train_loss': 0.8848629593849182}, {'val_loss': 0.7107415795326233, 'val_acc': 0.7363636493682861, 'train_loss': 0.860095202922821}, {'val_loss': 0.7106726765632629, 'val_acc': 0.75, 'train_loss': 0.810321569442749}, {'val_loss': 0.7572275400161743, 'val_acc': 0.7545454502105713, 'train_loss': 0.699836015701294}]

# num_epochs = 5
# history_final =  [{'val_loss': 3.6822474002838135, 'val_acc': 0.29999998211860657, 'train_loss': 11.54269027709961}, {'val_loss': 1.771144151687622, 'val_acc': 0.459090918302536, 'train_loss': 2.622060537338257}, {'val_loss': 1.49733304977417, 'val_acc': 0.47272729873657227, 'train_loss': 1.668800711631775}, {'val_loss': 1.4760304689407349, 'val_acc': 0.5136363506317139, 'train_loss': 1.4828541278839111}, {'val_loss': 1.4107227325439453, 'val_acc': 0.5590909719467163, 'train_loss': 1.355711579322815},{'val_loss': 1.3187934160232544, 'val_acc': 0.5363636612892151, 'train_loss': 1.436032772064209},{'val_loss': 1.171736717224121, 'val_acc': 0.550000011920929, 'train_loss': 1.3234833478927612},]
# Y_test = [[12, 11, 0, 7, 1, 4, 8, 6, 5, 5], [4, 0, 5, 10, 7, 6, 9, 0, 11, 0], [0, 12, 0, 11, 6, 5, 0, 7, 11, 8], [6, 4, 0, 9, 5, 7, 1, 10, 4, 5]]
# Y_pred = [[9, 0, 0, 8, 7, 5, 9, 0, 0, 6], [6, 0, 5, 0, 0, 0, 9, 0, 0, 0], [7, 7, 0, 12, 5, 4, 7, 7, 7, 9], [7, 0, 7, 9, 8, 7, 4, 7, 5, 7]]




# y_test, y_pred = convert_to_1d(Y_test,Y_pred)
name_override = False

def plot_losses(history):
    losses = [x['val_loss'] for x in history]
    print("losses", losses)
    plt.plot(losses, '-x')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['losses'])
    plt.title('Loss vs. No. of epochs')
    plt.savefig("Saves/plot_losses.png", dpi=600)
    plt.show()


def plot_accuracies(history):
    accuracies = [x['val_acc'] for x in history]
    print("accuracy", accuracies)
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['accuracy'])
    plt.title('Accuracy vs. No. of epochs')
    plt.ylim([0, 1])
    plt.savefig("Saves/plot_accuracies.png", dpi=600)
    plt.show()


def plot_all_losses(history):
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training_loss', 'val_loss'])
    plt.title('train & val Loss vs. No. of epochs')
    plt.savefig("Saves/plot_all_losses.png", dpi=600)
    plt.show()


# plot_all_losses(history_final)
# plot_losses(history_final)
# plot_accuracies(history_final)

def confusionMatrix(dat):
    #This block of code was due to confusion between thoese of us working on it.
    #As an explanation, I was told to assume the model got something correct if it was unknown and the model predicted unknown.
    #This created misleading confusion matrixes so it was removed.
    #for x in range(len(y_pred)):
    #    if y_pred[x] == 15 and y_tested[x] == 15:
    #        y_pred[x] = y_test[x]
    #    #else:
    #    #    print(f"{y_pred[x]},{y_test[x]}")

    y_pred,y_true,y_tested_against = dat
    y_pred = y_pred / (Config.parameters["CLASSES"][0]/Config.parameters["CLASSES"][0]) #The whole config thing is if we are splitting the classes further
    y_true = y_true / (Config.parameters["CLASSES"][0]/Config.parameters["CLASSES"][0])
    y_true = y_true.to(torch.int).tolist()
    y_pred = y_pred.to(torch.int).tolist()
    y_tested_against = y_tested_against.to(torch.int).tolist()

    return confusion_matrix(y_tested_against, y_pred, labels=list(range(Config.parameters["CLASSES"][0]+1)))


# def plot_confusion_matrix(y_test,y_pred):
#     cm = confusion_matrix(y_test, y_pred)
#     cm_display = ConfusionMatrixDisplay(cm).plot()
#     plt.savefig("./confusion_matrix.png", dpi=600)
#     plt.show()

# def compute_recall(y_test,y_pred):
#     return recall_score(y_test,y_pred,average='micro')
#
# def compute_precision(y_test,y_pred):
#     return precision_score(y_test,y_pred,average='micro')
#
# def compute_f1score(y_test,y_pred):
#     return f1_score(y_test,y_pred,average='micro')

import Dataload


def plot_confusion_matrix(cm:np.ndarray, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues, knowns=list(range(Config.parameters["CLASSES"][0]))):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    plt.figure(figsize=(12,12))

    #plt.xkcd()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    xtick = plt.xticks()

    #from https://stackoverflow.com/questions/24617429/matplotlib-different-colors-for-each-axis-label
    for number, text in zip(xtick[0], xtick[1]):
        if number in knowns:
            text.set_color("black")
        else:
            text.set_color("red")

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis].clip(0.000001,1)
        # cm = "{:.2f}".format(float)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print("cm", cm.shape)
    cm = cm.astype("int")

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        # print("cm ij ",str(round(cm[i, j], 2)))
        if j in knowns:
            plt.text(j, i, str(round(cm[i, j], 2)),
                    horizontalalignment="center",verticalalignment="center_baseline",
                    color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, str(round(cm[i, j], 2)),
                    horizontalalignment="center",verticalalignment="center_baseline",
                    color="red")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if not name_override:
        specific = pd.read_csv("Saves/fscore.csv").tail(1).to_string(header=False,index=False,).replace(" ","").replace(".","")
    else:
        specific = name_override
    #plt.savefig(f"Saves/conf/{title}{specific}.png", dpi=1600)
    plt.savefig(f"Saves/{title}.png", dpi=1600)



# Compute confusion matrix
# cnf_matrix = confusion_matrix(y_test, y_pred)
# np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
# plt.figure()

# Plot normalized confusion matri
#
# class_names = Dataload.get_class_names([0,1,4,5,6,7,8,9,10,11,12])
# plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
#                       title='Normalized confusion matrix')
# plt.show()


if __name__ == "__main__":
    import plotly
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    def findX(currently):
        if pd.isna(currently):
            return "Error"
        else:
            #Got how to split the space from https://stackoverflow.com/a/16313889
            return " ".join(currently.split()[1:])

    def findType(X):
        if pd.isna(X):
            return "Error"
        else:
            return X.split()[0]

    def variations(data):
        fig = px.histogram(data[data["OOD Type"]=="Soft"],y="Test_F1",x="x",barmode="group")
        for alg in data["OOD Type"].unique():
            fig.add_bar(x=data[data["OOD Type"]==alg]["x"],y=data[data["OOD Type"]==alg]["Test_F1"],name=alg)
        fig.show()

    def allPRF(data:pd.DataFrame):
        fig = px.bar(data,y=["Test_F1","Test_Recall","Test_Precision"],x="OOD Type",barmode="group",text_auto=True,orientation="v")
        #fig = px.bar(data,x="Test_F1",y=data["OOD Type"].unique(),barmode="group",text_auto=True,orientation="v")
        #fig.update()
        #fig.add_bar(x=data[data["OOD Type"]=="Energy"]["x"],y=data[data["OOD Type"]=="Energy"]["Test_Recall"],name="Recall")
        
        fig.show()

    def allPRF2(data,alg:str):
        fig = go.Figure()
        fig.add_trace(go.Bar(name="F1",x=data[data["OOD Type"]==alg]["x"],y=data[data["OOD Type"]==alg]["Test_F1"]))
        fig.add_trace(go.Bar(name="Recall",x=data[data["OOD Type"]==alg]["x"],y=data[data["OOD Type"]==alg]["Test_Recall"]))
        fig.add_trace(go.Bar(name="Precision",x=data[data["OOD Type"]==alg]["x"],y=data[data["OOD Type"]==alg]["Test_Precision"]))
        fig.update_layout(title=alg)
        fig.show()

    def reformatData(data):
        if os.path.exists("Saves/ScoresAll.xlsx"):
            os.remove("Saves/ScoresAll.xlsx")
        for x in data["x"].unique():
            df = data[data["x"]==x]
            #print(df[["OOD Type","Test_F1","Test_Recall","Test_Precision"]])
            x = x.replace(" ", "")
            x = x.replace("[","")
            x = x.replace("]","")[0:31]
            if os.path.exists("Saves/ScoresAll.xlsx"):
                #Gotten from https://stackoverflow.com/a/63692307
                with pd.ExcelWriter('Saves/ScoresAll.xlsx', engine='openpyxl', mode='a') as writer: 
                    df[["OOD Type","Test_F1","Test_Recall","Test_Precision","Dataset","model"]].to_excel(writer,x)
            else:
                df[["OOD Type","Test_F1","Test_Recall","Test_Precision","Dataset","model"]].to_excel("Saves/ScoresAll.xlsx",x)

    if __name__ == "__main__":
        data = pd.read_csv("Saves/Scoresall.csv")
        #data.drop_duplicates(subset=["Currently Modifying"],inplace=True,keep="last")
        data["x"] = data["Currently Modifying"].apply(findX)
        #Remove rows that dont match and print error
        matching = data["Currently Modifying"].apply(findType) == data["OOD Type"]
        if matching.sum()!=len(matching):
            print(matching)
        data = data[matching]
        data["Type of change"] = data["x"].apply(findType)

        allPRF(data[data["x"]=="num_epochs 100"])
        variations(data)
        for alg in data["OOD Type"].unique():
            allPRF2(data,alg)
        reformatData(data)
