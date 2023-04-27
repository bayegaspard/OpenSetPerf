# -*- coding: utf-8 -*-

# Load the top modules that are used in multiple places
import numpy as np
import pandas as pd

# Some global variables to drive the script
# data_url is the location of the data
# Data is not loaded from a local file
# Data is loaded from a prepocessed dataset
data_url="/home/bgaspard/Desktop/softmax/updated_CIC.csv"


# label names (YY) in the data and their
# mapping to numerical values
label_map = {
 'BENIGN' : 0,
 'Bot' : 1,
 'DDoS' : 2,
 'DoS GoldenEye': 3,
 'DoS Hulk' : 4,
 'DoS Slowhttptest' : 5,
 'DoS slowloris' : 6,
 'FTP-Patator' : 7,
 'Heartbleed' : 8,
 'Infiltration' : 9,
 'PortScan' : 10,
 'SSH-Patator' : 11,
 'Web Attack – Brute Force' : 12,
 'Web Attack – Sql Injection' : 13,
 'Web Attack – XSS':14

}

num_ids_features = 1503
num_ids_classes = 15
ids_classes = ['BENIGN', 'Bot', 'DDoS', 'DoS GoldenEye', 'DoS Hulk', 'DoS Slowhttptest', 'DoS slowloris', 'FTP-Patator', 'Heartbleed', 'Infiltration', 'PortScan', 'SSH-Patator', 'Web Attack – Brute Force', 'Web Attack – Sql Injection', 'Web Attack – XSS']
# Utility functions used by classifiers
# In particular to load and split data and output results
def ids_load_df_from_csv():
    """
    Load dataframe from csv file
    Input:
        None
    Returns:
        None
    """

    df = pd.read_csv(data_url)

    print ("load Dataframe shape", df.shape)

    return df

def ids_split(df):
    """
    Input:
        Dataframe that has columns of covariates followed by a column of labels
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test as numpy arrays
    """

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    numcols = len(df.columns)
    print("df.shape", df.shape)

    X = df.iloc[:, 0:numcols-1]
    y = df.loc[:, 'label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)
    print ("X_train.shape", X_train.shape, "y_train.shape", y_train.shape)
    print ("X_val.shape", X_val.shape, "y_val.shape", y_val.shape)
    print ("X_test.shape", X_test.shape, "y_test.shape", y_test.shape)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    y_train = y_train.values
 #   y_train = y_train.astype('str')
#    y_train = np.nan_to_num(y_train, nan=0) # added
    print(y_train)
    y_train = y_train.astype(np.float64)
    y_val = y_val.values
    y_test = y_test.values
    y_train = y_train.astype('str')
    return X_train, X_val, X_test, y_train, y_val, y_test

def ids_accuracy (y_actual, y_pred):
    """
    Input:
        Numpy arrays with actual and predicted labels
    Returns:
        multiclass accuracy and f1 scores; two class accuracy and f1 scores
    """

    from sklearn.metrics import accuracy_score
    from sklearn.metrics import f1_score

    # modify labels to get results for two class classification
    y_actual_2 = (y_actual > 0).astype(int)
    y_pred_2 = (y_pred > 0).astype(int)

    acc = accuracy_score (y_actual, y_pred)
    f1 = f1_score(y_actual, y_pred, average='macro')
    acc_2 = accuracy_score (y_actual_2, y_pred_2)
    f1_2 = f1_score(y_actual_2, y_pred_2)

    return acc, f1, acc_2, f1_2


def ids_metrics(y_actual, y_pred):
    """
    Input:
        Numpy arrays with actual and predicted labels
    Returns:
        None
    Print: various classification metrics
    """

    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix (y_actual, y_pred)
    print (cm)

    acc, f1, acc_2, f1_2 = ids_accuracy (y_actual, y_pred)
    print('Classifier accuracy : {:.4f}'.format(acc), 'F1 score: {:.4f}'.format(f1))
    print('Two class classifier accuracy : {:.4f}'.format(acc_2), 'F1 score: {:.4f}'.format(f1_2))


'''
Fully connected three layer neural network using PyTorch
With utility functions based on deeplizard tutorial
https://deeplizard.com/learn/video/v5cngxo4mIg
'''

# Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import TensorDataset, DataLoader
from IPython.display import display, clear_output
import pandas as pd
import time

from itertools import product
from collections import namedtuple
from collections import OrderedDict

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

class RunBuilder():
    @staticmethod
    def get_runs(params):
        Run = namedtuple('Run', params.keys())
        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))
        return runs

class RunManager():
    def __init__(self):
        self.epoch_count = 0
        self.epoch_loss = 0
        self.epoch_num_correct = 0
        self.epoch_start_time = None

        self.run_params = None
        self.run_count = 0
        self.run_data = []
        self.run_start_time = None

        self.network = None

    def begin_run(self, run, network, loss_fn, train_inputs, train_targets, X_val, y_val):
        self.run_start_time = time.time()
        self.run_params = run
        self.run_count += 1

        self.network = network
        self.loss_fn = loss_fn
        self.train_inputs = train_inputs
        self.train_targets = train_targets
        self.X_val = X_val
        self.y_val = y_val

    def end_run(self):
        self.epoch_count = 0

    def begin_epoch(self):
        self.epoch_start_time = time.time()
        self.epoch_count += 1
        self.epoch_loss = 0
        self.epoch_num_correct = 0

    def end_epoch(self):
        epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time() - self.run_start_time

        with torch.no_grad():
            loss = (self.loss_fn(self.network(self.train_inputs), self.train_targets)).item()

            val_inputs = torch.from_numpy(self.X_val).float()
            val_preds = self.network(val_inputs)
            y_pred = val_preds.argmax(dim=1)
            accuracy = accuracy_score (self.y_val, y_pred)

        results = OrderedDict()
        results["run"] = self.run_count
        results["epoch"] = self.epoch_count
        results['loss'] = loss
        results["accuracy"] = accuracy
        results['epoch duration'] = epoch_duration
        results['run duration'] = run_duration
        for k,v in self.run_params._asdict().items(): results[k] = v
        self.run_data.append(results)

def ids_nn():
    params = OrderedDict(
        lr = [.008]
        ,batch_size = [256]
        ,num_epochs = [10]
        ,step_size = [5]
        ,gamma = [0.50]
    )

    rm = RunManager()

    df = ids_load_df_from_csv ()
    X_train, X_val, X_test, y_train, y_val, y_test = ids_split(df)
#    y_train = y_train.nan_to_num(arr, nan=0)
    print("ytrain in place",y_train)
    y_train = y_train.astype(np.float64)
    train_inputs = torch.from_numpy(X_train).float()
    train_targets = torch.from_numpy(y_train).long()
    train_ds = TensorDataset(train_inputs, train_targets)

    # Run for each combination of params
    for run in RunBuilder.get_runs(params):
        torch.manual_seed(42)
        print (run)

        network = nn.Sequential(
            nn.Linear(num_ids_features, num_ids_features)
            ,nn.ReLU()
            ,nn.Linear(num_ids_features, num_ids_classes)
        )

        train_dl = DataLoader(train_ds, run.batch_size, shuffle=True)
        opt = torch.optim.Adam(network.parameters(), run.lr)
        sch = torch.optim.lr_scheduler.StepLR(opt, run.step_size, run.gamma)
        loss_fn = F.cross_entropy

        rm.begin_run(run, network, loss_fn, train_inputs, train_targets, X_val, y_val)
        # Training loop
        for epoch in range(run.num_epochs):
            rm.begin_epoch()

            for xb,yb in train_dl:
                pred = network(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()
                opt.zero_grad()

            rm.end_epoch()
            sch.step()
        rm.end_run()

    print(pd.DataFrame.from_dict(rm.run_data))

    val_inputs = torch.from_numpy(X_val).float()
    val_pred = network(val_inputs)

    # Since the model returns values for all num_ids_classes
    # The ids_class with the maximim value is picked as the label
    val_pred = val_pred.argmax(dim=1)

    # A numpy array is needed to evaluate metrics
    y_pred = val_pred.detach().to('gpu').numpy()
    ids_metrics(y_val, y_pred)

ids_nn()
