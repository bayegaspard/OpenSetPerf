import os
import Config
import pandas as pd
import torch

import Dataload
import plots

def generateHyperparameters(root_path=""):
    """
    Saves current versions of the hyperparameters in Saves/hyperparam/hyperparam.csv

    This file is no longer being used for things.
    """
    if not (os.path.exists(os.path.join(root_path,"Saves","hyperparam")) and os.path.exists(os.path.join(root_path,"Saves","unknown"))):
        os.mkdir(os.path.join(root_path,"Saves","hyperparam"))
        os.mkdir(os.path.join(root_path,"Saves","unknown"))
    if os.path.exists(os.path.join(root_path,"Saves","hyperparam","hyperParam.csv")) and os.path.exists(os.path.join(root_path,"src","main","unknown","unknowns.csv")):
        print("Hyperparam.csv and unknown.csv files exist")
    else:
        print("Either hyperparam.csv or unknown.csv does not exist , generating one based on config file settings ....")
        parameters = Config.parameters
        # print(parameters)
        param = pd.DataFrame.from_dict(parameters, orient="columns")
        if param["num_epochs"][0] == 0:
            param["num_epochs"][0] = Config.num_epochs
        param.to_csv(os.path.join(root_path,"Saves","hyperparam","hyperParam.csv"))
        unknown_classes = Config.class_split["unknowns_clss"]
        param = pd.DataFrame(unknown_classes,columns=["Unknowns"])
        param.to_csv(os.path.join(root_path,"Saves","unknown","unknowns.csv"))
        print("Files created successfully !")
    
    if not os.path.exists("Saves/roc"):
        os.mkdir("Saves/roc")
    if not os.path.exists("Saves/models"):
        os.mkdir("Saves/models")
    if not os.path.exists("Saves/conf"):
        os.mkdir("Saves/conf")



# generateHyperparameters(hyperpath,unknownpath)

def getDatagroup():
    """
    This gets the datagroups as described in the config.

    returns:
        tuple containing:
            knowns set - torch dataloader using the classes in config knowns_clss
            unknown set - torch dataloader using the classes in congfig unknowns_clss
    """
    groupType = Config.parameters["Datagrouping"][0]
    if groupType == "ClassChunk":
        train = Dataload.ClassDivDataset(os.path.join("datasets",Config.parameters["Dataset"][0]), use=Config.class_split["knowns_clss"])
        unknowns = Dataload.ClassDivDataset(os.path.join("datasets",Config.parameters["Dataset"][0]), use=Config.class_split["unknowns_clss"], unknownData=True)
    elif groupType == "Dendrogramlimit":
        train = Dataload.ClusterLimitDataset(os.path.join("datasets",Config.parameters["Dataset"][0]), use=Config.class_split["knowns_clss"])
        unknowns = Dataload.ClusterLimitDataset(os.path.join("datasets",Config.parameters["Dataset"][0]), use=Config.class_split["unknowns_clss"], unknownData=True)
    elif groupType == "DendrogramChunk":
        train = Dataload.ClusterDivDataset(os.path.join("datasets",Config.parameters["Dataset"][0]), use=Config.class_split["knowns_clss"])
        unknowns = Dataload.ClusterDivDataset(os.path.join("datasets",Config.parameters["Dataset"][0]), use=Config.class_split["unknowns_clss"], unknownData=True)
    else:
        raise ValueError("Invalid Dataloader type")
    return train,unknowns

# def readFromFiles(path):
def checkAttempLoad(root_path=""):
    """
    Creates the training, testing, and validaton datasets and saves them in Saves/Data.pt, Saves/DataTest.pt, and Saves/DataVal.pt.
    if Config's "attemptLoad" is true it instead loads the datasets from the files and does not create them. 
    This is so that the validation and testing data does not get mixed up which would invalidate the validation data.
    """
    # get the data and create a test set and train set
    
    print("Reading datasets to create test and train sets")
    
    train, unknowns = getDatagroup()
    train, val = torch.utils.data.random_split(train,[len(train) - int(len(train) * Config.parameters["testlength"][0]),int(len(train) * Config.parameters["testlength"][0])]) 
    
    if len(Config.class_split["unknowns_clss"])>0:
        if (Config.parameters["Mix unknowns and validation"][0]):
            test = torch.utils.data.ConcatDataset([val, unknowns])
        else:
            test = unknowns
    else:
        test=val
    
    if Config.unit_test_mode:
        return train, test, val

    if Config.parameters["attemptLoad"][0] and os.path.exists(os.path.join(root_path,"Saves","Data.pt")):
        try:
            train = torch.load(os.path.join(root_path,"Saves","Data.pt"))
            test = torch.load(os.path.join(root_path,"Saves","DataTest.pt"))
            val = torch.load(os.path.join(root_path,"Saves","DataVal.pt"))
            print("Loading from data and test checkpoint ...")
        except ModuleNotFoundError:
            print("Dataset outdated and failed to load.")

    else:
        #test = unknowns
        torch.save(train,os.path.join(root_path,"Saves","Data.pt"))
        torch.save(test,os.path.join(root_path,"Saves","DataTest.pt"))
        torch.save(val,os.path.join(root_path,"Saves","DataVal.pt"))
        if Config.parameters["attemptLoad"][0]:
            print("No model train and test checkpoint was found, saving datacheckpoints ...")
    return train, test, val

def incrementLoopModData(changed:list):
    """
    Adds classes into the train and validation datasets.

    parameters:
        changed - list of integers corrisponding to the classes to add to the known classes
    
    returns:
        Nothing, the changed datasets are saved in the Saves/Data*.pt files
    """
    if Config.parameters["Datagrouping"][0] == "ClassChunk":
        known = Dataload.ClassDivDataset(os.path.join("datasets",Config.parameters["Dataset"][0]), use=changed)
        unknowns = Dataload.ClassDivDataset(os.path.join("datasets",Config.parameters["Dataset"][0]), use=Config.class_split["unknowns_clss"], unknownData=True)
    elif Config.parameters["Datagrouping"][0] == "Dendrogramlimit":
        known = Dataload.ClusterLimitDataset(os.path.join("datasets",Config.parameters["Dataset"][0]), use=changed)
        unknowns = Dataload.ClusterLimitDataset(os.path.join("datasets",Config.parameters["Dataset"][0]), use=Config.class_split["unknowns_clss"], unknownData=True)
    elif Config.parameters["Datagrouping"][0] == "DendrogramChunk":
        known = Dataload.ClusterDivDataset(os.path.join("datasets",Config.parameters["Dataset"][0]), use=changed)
        unknowns = Dataload.ClusterDivDataset(os.path.join("datasets",Config.parameters["Dataset"][0]), use=Config.class_split["unknowns_clss"], unknownData=True)
    else:
        raise ValueError("Invalid Dataloader type")
    
    trainGroup, testGroup = torch.utils.data.random_split(known,[len(known) - int(len(known) * Config.parameters["testlength"][0]),int(len(known) * Config.parameters["testlength"][0])]) 

    try:
        train = torch.utils.data.ConcatDataset([torch.load(os.path.join("","Saves","Data.pt")),trainGroup])
        val = torch.utils.data.ConcatDataset([torch.load(os.path.join("","Saves","DataVal.pt")),testGroup])
    except ModuleNotFoundError:
        train = trainGroup
        val = testGroup
        import sys
        print("Dataset outdated and failed to load.",file=sys.stderr)
    test = torch.utils.data.ConcatDataset([val,unknowns])

    torch.save(train,os.path.join("Saves","Data.pt"))
    torch.save(test,os.path.join("Saves","DataTest.pt"))
    torch.save(val,os.path.join("Saves","DataVal.pt"))
    return


def deletefile(path):
    """
    just deletes a file if it exists. There must be a way to do this more easily.
    """
    if os.path.exists(path):
        os.remove(path)

def refreshFiles(root_path):
    """
    Deletes all of the files that might cause proboblems when running again.
        These files include:
            The dataloader
            the hyperparameters file
            the unknowns file
    """
    deletefile(os.path.join(root_path,"Saves","hyperparam","hyperParam.csv"))
    deletefile(os.path.join(root_path,"Saves","unknown","unknowns.csv"))
    # deletefile(os.path.join(root_path,"Saves","Data.pt"))
    # deletefile(os.path.join(root_path,"Saves","DataTest.pt"))
    #os.remove(os.path.join(root_path,"src","main","test"))
    #os.mkdir(os.path.join(root_path,"src","main","test"))


def convert_to_1d(y_test, y_pred):
    """
    changes a pair of two dimentional lists into single dimentional lists. This is used to collapse batches into a single row.
    """
    y_test_final = []
    y_pred_final = []
    for i in range(len(y_test)):
        for j in range(len(y_pred[i])):
            y_test_final.append(y_test[i][j])
            y_pred_final.append(y_pred[i][j])
    return y_test_final, y_pred_final


def write_hist_to_file(lst, num_epochs, type=""):
    """
    Adds lines to history.csv. This is used to keep a log of how well the algorithm works during training.
    This also saves the same information in an algorithm specific file that we thought would be useful at some point.
    """
    if Config.unit_test_mode:
        return
    for l in lst:
        l["type"] = type
    if os.path.exists(os.path.join("Saves","history.csv")):
        hist = pd.read_csv(os.path.join("Saves","history.csv"), index_col=0)
        hist = pd.concat((hist, pd.DataFrame.from_dict(lst)))
    else:
        hist = pd.DataFrame.from_dict(lst)
    hist.to_csv(os.path.join("Saves","history.csv"))
    with open(os.path.join('Saves',f'history{type}.txt'), 'a') as fp:
        # fp.write(f"history for {num_epochs} \n")
        fp.write("\n")
        for item in lst:
            # write each item on a new line
            fp.write(f"num_epochs {num_epochs} " + str(item) + "\n")
        print('Writing history Done')


def write_scores_to_file(lst, num_epochs, type=""):
    """
    *seems to be broken

    Adds a line to scores.csv. This is used to keep a log of how well the algorithm works during evaluation.
    This also saves the same information in an algorithm specific file that we thought would be useful at some point.
    """
    if Config.unit_test_mode:
        return
    thisRun = pd.DataFrame.from_dict(lst)
    thisRun["type"] = type
    if os.path.exists(os.path.join("Saves","scores.csv")):
        hist = pd.read_csv(os.path.join("Saves","scores.csv"), index_col=0)
        hist.loc[len(hist)] = thisRun.iloc[0]
    else:
        hist = thisRun

    hist.to_csv(os.path.join("Saves","scores.csv"))
    with open(os.path.join('Saves',f'scores{type}.txt'), 'a') as fp:
        fp.write("\n")
        for item in lst:
            # write each item on a new line
            fp.write(f"num_epochs {num_epochs} " + str(item).format(num_epochs) + "\n")
        print('Writing scores Done')


def write_batch_to_file(loss, num, modeltype="", batchtype=""):
    """
    Writes each batch to a file containing the loss, batch number, model type, and batch type.
        loss - the value output by the loss function, how bad the algorithm is doing
        batch number - the number of batch it was in the epoch
        model type - the endlayer type while running
        batch type - if this was either a training or evaluation batch
    """
    if Config.unit_test_mode:
        return
    thisRun = pd.DataFrame([[loss.item(), num, modeltype, batchtype]],
                           columns=["Loss", "Batch Number", "Model Type", "Batch Type"])
    # thisRun["Loss"] = loss.detach()
    # thisRun["Batch Number"] = num
    # thisRun["Model Type"] = modeltype
    # thisRun["Batch Type"] = batchtype
    if os.path.exists(os.path.join("Saves","batch.csv")):
        hist = pd.read_csv(os.path.join("Saves","batch.csv"), index_col=0)
        hist.loc[len(hist)] = thisRun.iloc[0]
    else:
        hist = thisRun

    hist.to_csv(os.path.join("Saves","batch.csv"))


def store_values(history: list, Y_predict: list, Y_test: list, num_epochs: int, end_type: str):
    """
    This stores the history and current evaluation of the model.
    to make these logs store_values requires several things:
        history - a list of dictonaries that is generated by AttackTrainingClassification.fit()
        Y_predict - a list of predicted values, possibly gotten from AttackTrainingClassification.store
        Y_test - the actual values for Y_predict, possibly gotten from AttackTrainingClassification.store
        num_epochs - the number of epochs the model ran for
        end_type - the type of endlayer used (Config's "OOD Type")
    """
    y_test, y_pred = convert_to_1d(Y_test, Y_predict)
    recall = plots.recall_score(y_test, y_pred, average='weighted', zero_division=0)
    precision = plots.precision_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = 2 * (precision * recall) / (precision + recall)
    # auprc = average_precision_score(y_test, y_pred, average='samples')
    score_list = [recall, precision, f1]
    write_hist_to_file(history, num_epochs, end_type)
    write_scores_to_file(score_list, num_epochs, end_type)

    
def create_params_Fscore(path, score, threshold = None):
    """
    An old version of output storage that contains the current settings of Config and the final F1 score of the test.
    We have devised create_params_All() and addMeasurement() to replace this to store more tyoes of results dynamically.

    this took the parameters:
        path - path to the root directory. Usually left as ""
        score - the final F1 score of the algorithm
        threshold - I think this is supposed to be the threshold? I dont know why.
    """
    if Config.unit_test_mode:
        return
    params = pd.read_csv(os.path.join(path,"Saves","hyperparam","hyperParam.csv"),index_col=0)

    if threshold != None:
        params["threshold"] = threshold

    params["Fscore"] = [score,"Score"]
    if os.path.exists(os.path.join(path,"Saves","fscore.csv")):
        hist = pd.read_csv(os.path.join(path,"Saves","fscore.csv"))
        hist.loc[len(hist)] = params.iloc[0]
    else:
        temp = params.iloc[0]
        params.iloc[0] = params.iloc[1]
        params.iloc[1] = temp
        hist = params
    
    
    hist.to_csv(os.path.join(path,"Saves","fscore.csv"),index=False)

def create_params_All(path="",name="Scoresall.csv"):
    """
    Generates a new line of the file scoresAll.csv that we use to store the scores from the run.
    The new line contains all of the Config values that we are using.
    If the file does not exist this creates the file.

    you can change the path with the path parameter.
    """
    if Config.unit_test_mode:
        return
    params = pd.DataFrame(Config.parameters,columns=Config.parameters.keys())


    if os.path.exists(os.path.join(path,"Saves",name)):
        hist = pd.read_csv(os.path.join(path,"Saves",name),index_col=0)
        hist = pd.concat([hist,params.iloc[[0]]],axis=0,ignore_index=True)
    else:
        hist = params.iloc[[0]]
    
    #hist = hist.transpose()
    hist.to_csv(os.path.join(path,"Saves",name))

def create_loop_history(name:str,path=""):
    if Config.unit_test_mode:
        return
    params = pd.DataFrame([Config.loops],columns=Config.loops2)
    params["Version"] = Config.parameters["Version"][0]


    if os.path.exists(os.path.join(path,"Saves",name)):
        hist = pd.read_csv(os.path.join(path,"Saves",name),index_col=0)
        hist = pd.concat([hist,params.iloc[[0]]],axis=0,ignore_index=True)
    else:
        hist = params.iloc[[0]]
    
    #hist = hist.transpose()
    hist.to_csv(os.path.join(path,"Saves",name))

def addMeasurement(name:str,val,path="",fileName="Scoresall.csv"):
    """
    Adds a measurement to the LATEST line in the Scoresall.csv file. This may cause problems if you are running two versions at once.
    we reccomend only running one version at once. 

    parameters:
        name - measurement name
        val - measurement value
    """
    if Config.unit_test_mode:
        return
    total = pd.read_csv(os.path.join(path,"Saves",fileName),index_col=0)
    #print(f"last valid index = {total.last_valid_index()} item name= {name}, item value={val}")
    if name in total and not (pd.isnull(total.at[total.last_valid_index(),name]) or name in ["Number Of Failures"]):
        total.at[total.last_valid_index(),"A spot has already been filled?"] = "An error has occured"
    total.at[total.last_valid_index(),name] = val
    total.to_csv(os.path.join(path,"Saves",fileName))
    return total.last_valid_index()

