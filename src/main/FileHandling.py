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
        unknown_classes = Config.parameters["Unknowns_clss"][0]
        param = pd.DataFrame(unknown_classes,columns=["Unknowns"])
        param.to_csv(os.path.join(root_path,"Saves","unknown","unknowns.csv"))
        print("Files created successfully !")
    
    if not os.path.exists("Saves/roc"):
        os.mkdir("Saves/roc")
    if not os.path.exists("Saves/models"):
        os.mkdir("Saves/models")
    if not os.path.exists("Saves/conf"):
        os.mkdir("Saves/conf")

attemptload_message = True

# generateHyperparameters(hyperpath,unknownpath)

def getDatagroup():
    """
    This gets the datagroups as described in the config.

    returns:
        tuple containing:
            knowns set - torch dataloader using the classes in config knowns_clss
            unknown set - torch dataloader using the classes in congfig unknowns_clss
    """
    groupType = Config.parameters["Dataloader_Variation"][0]
    train = Dataload.DataloaderTypes[groupType](os.path.join("datasets",Config.parameters["Dataset"][0]), use=Config.parameters["Knowns_clss"][0])
    unknowns = Dataload.DataloaderTypes[groupType](os.path.join("datasets",Config.parameters["Dataset"][0]), use=Config.parameters["Unknowns_clss"][0], unknownData=True)
    return train,unknowns

# def readFromFiles(path):
def checkAttempLoad(root_path=""):
    """
    Creates the training, testing, and validaton datasets and saves them in Saves/Data.pt, Saves/DataTest.pt, and Saves/DataVal.pt.
    if Config's "attemptLoadData" is true it instead loads the datasets from the files and does not create them. 
    This is so that the validation and testing data does not get mixed up which would invalidate the validation data.
    """
    # get the data and create a test set and train set
    
    print("Reading datasets to create test and train sets")
    
    train, unknowns = getDatagroup()
    train, val = torch.utils.data.random_split(train,[len(train) - int(len(train) * Config.parameters["testlength"][0]),int(len(train) * Config.parameters["testlength"][0])]) 
    
    if len(Config.parameters["Unknowns_clss"][0])>0:
        if (Config.parameters["Mix unknowns and validation"][0]):
            test = torch.utils.data.ConcatDataset([val, unknowns])
        else:
            test = unknowns
    else:
        test=val
    
    if Config.unit_test_mode:
        return train, test, val

    if Config.parameters["attemptLoadData"][0] and os.path.exists(os.path.join(root_path,"Saves","Data.pt")):
        print("Found prior dataset to load")
        try:
            train = torch.load(os.path.join(root_path,"Saves","Data.pt"))
            test = torch.load(os.path.join(root_path,"Saves","DataTest.pt"))
            val = torch.load(os.path.join(root_path,"Saves","DataVal.pt"))
            print("Loading from data and test checkpoint ...")
        except ModuleNotFoundError:
            print("Dataset outdated and failed to load.")

    else:
        #test = unknowns
        global attemptload_message
        if attemptload_message:
            print("Saving data. Use -attemptLoad[Data,Model] 1 to use saved data or model")
            attemptload_message = False
        else:
            print("Saving data.")
        torch.save(train,os.path.join(root_path,"Saves","Data.pt"))
        torch.save(test,os.path.join(root_path,"Saves","DataTest.pt"))
        torch.save(val,os.path.join(root_path,"Saves","DataVal.pt"))
        if Config.parameters["attemptLoadData"][0]:
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

    groupType = Config.parameters["Dataloader_Variation"][0]
    known = Dataload.DataloaderTypes[groupType](os.path.join("datasets",Config.parameters["Dataset"][0]), use=changed)
    unknowns = Dataload.DataloaderTypes[groupType](os.path.join("datasets",Config.parameters["Dataset"][0]), use=Config.parameters["Unknowns_clss"][0], unknownData=True)

    
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
        # print('Writing history Done')


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
        # print('Writing scores Done')


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

class Score_saver():

    def __init__(self,path="Scoresall.csv",Record_Hyperparams=True):
        """
        Score_saver() is a class to consolidate the saving of data to the csv files. 
        When a Score_saver() object is initialized then it creates a new row onto the file, 
        the new row contains all of the current Config settingss at the start.
        It will also attempt to save some of the data as a tensorboard if the Config option is true.
        """
        self.writer = None
        self.path = path #unused at the moment
        self.name_all = {path:0}
        if Record_Hyperparams:
            self.create_params_All()
        else:
            pd.DataFrame().to_csv(self.path)
        if Config.save_as_tensorboard:
            self.tensorboard_start()
        
        

    def __call__(self,name:str,val,path="",fileName=None, recursiveList=False):
        """
        Adds the measurement to the file. The __call__ version allows the Score_saver to be called like this:
        scoresaver = Score_saver()
        scoresaver("hello",1)
        
        parameters:
            name - Name of measurement
            val - value to save under that name
            filename - overrite the orignal file to save as
        """
        if fileName is None:
            fileName = self.path
        self.addMeasurement(name,val,path,fileName,recursiveList=recursiveList)

    def create_params_All(self,name=None):
        """
        Generates a new line of the file that we use to store the scores from the run.
        The new line contains all of the Config values that we are using.
        The csv is generated in Saves/'name'. If the file does not exist this creates the file.

        parameters:
            name - name of the file to save to.
                -if name is None then name will be the Score_saver's path (defalut:"Scoresall.csv").
        """
        if Config.unit_test_mode:
            return
        
        if name is None:
            name = self.path
        params = pd.DataFrame(Config.parameters,columns=Config.parameters.keys())


        if os.path.exists(os.path.join("Saves",name)):
            hist = pd.read_csv(os.path.join("Saves",name),index_col=0)
            hist = pd.concat([hist,params.iloc[[0]]],axis=0,ignore_index=True)
        else:
            hist = params.iloc[[0]]

        self.name_all[name] = hist.last_valid_index()
        #hist = hist.transpose()
        hist.to_csv(os.path.join("Saves",name))
        if name=="Scoresall.csv" and False:
            self.addMeasurement("TESTING_INDEX_VALUE",self.name_all[name])
        

    def create_loop_history(self,name:str):
        """
        creates a CSV file like create_params_all() but this score file is for keeping track of what loops were run.
        It was created so you can cross refrence and figure out where in Scoresall.csv you need to look.

        parameters:
        name - name of the file to save to.
            -if name is None then name will be the Score_saver's path (defalut:"Scoresall.csv").
        
        """
        if Config.unit_test_mode:
            return
        params = pd.DataFrame([Config.loops],columns=Config.loops2)
        params["Version"] = Config.parameters["Version"][0]


        if os.path.exists(os.path.join("Saves",name)):
            hist = pd.read_csv(os.path.join("Saves",name),index_col=0)
            hist = pd.concat([hist,params.iloc[[0]]],axis=0,ignore_index=True)
        else:
            hist = params.iloc[[0]]

        #hist = hist.transpose()
        hist.to_csv(os.path.join("Saves",name))

    def addMeasurement(self,name:str,val,path="",fileName=None,step=0, recursiveList=0):
        """
        Adds a measurement to the LATEST line in the Scoresall.csv file. This may cause problems if you are running two versions at once.
        we reccomend only running one version at once. 

        parameters:
            name - measurement name
            val - measurement value
            fileName - file to save as, if None uses defalut of 'Scoresall.csv'
            step - tensorboard step value

        returns:
            Last valid index in the CSV
        """
        if Config.unit_test_mode:
            return
        if recursiveList>0 and (hasattr(val, '__iter__')):
            for num,v in enumerate(val):
                self.addMeasurement(name+f"_{num}", v, path, fileName, step, recursiveList-1)
            return
        if fileName is None:
            fileName = self.path
        if self.writer is not None:
            if not isinstance(val, str):
                self.writer.add_scalar(name,val,step)
                if "F1" in name or "Unknowns" in name:
                    self.writer.add_hparams({},{name:val})
            else:
                self.writer.add_text(name,val,step)

        total = pd.read_csv(os.path.join(path,"Saves",fileName),index_col=0)
        if fileName in self.name_all.keys():
            index = self.name_all[fileName]
        else:
            index = total.last_valid_index()
        #print(f"last valid index = {total.last_valid_index()},current index = {index} item name= {name}, item value={val}")
        if name in total and not (pd.isnull(total.at[index,name]) or name in ["Number Of Failures"]):
            total.at[index,"A spot has already been filled?"] = f"An error has occured. {name} already has a value"
            import sys
            print(f"Something tried to save to a file position ({name}) that has already been filled. \n This might be caused by running the model twice.",file=sys.stderr)
        total.at[index,name] = val
        total.to_csv(os.path.join(path,"Saves",fileName))
        return index

    @staticmethod
    def create_loop_history_(name:str,path=""):
        """
        A static version of create_loop_history() if needed.
        """
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

    @staticmethod
    def addMeasurement_(name:str,val,path="",fileName="Scoresall.csv"):
        """
        A static version of addmasurement()
        """
        if Config.unit_test_mode:
            return
        total = pd.read_csv(os.path.join(path,"Saves",fileName),index_col=0)
        #print(f"last valid index = {total.last_valid_index()} item name= {name}, item value={val}")
        if name in total and not (pd.isnull(total.at[total.last_valid_index(),name]) or name in ["Number Of Failures"]):
            total.at[total.last_valid_index(),"A spot has already been filled?"] = f"An error has occured. {name}-already has a value"
            import sys
            print(f"Something tried to save to a file position ({name})-that has already been filled. \n This might be caused by running the model twice.",file=sys.stderr)
        total.at[total.last_valid_index(),name] = val
        total.to_csv(os.path.join(path,"Saves",fileName))
        return total.last_valid_index()

    def tensorboard_start(self):
        """
        Starts the tensorboard recording anything passing through the Score_saver
        """
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(comment=f"Run Number[{self.name_all.get('Scoresall.csv','N/A')}]")
        self.writer.add_hparams({x:str(Config.parameters[x][0]) for x in Config.parameters.keys() if x not in ["optimizer"]},{})
        self.writer.add_hparams({x:str(Config.parameters[x]) for x in Config.parameters.keys() if x in ["optimizer"]},{})
        
    def tensorboard_True(self):
        """This is supposed to say if the tensorboard is being written to but it is unused."""
        if self.writer is None:
            return False
        return True

    def start(self):
        """
        This is an unused function that was suppose to be helping with the initilization method but able ot be recalled when nessisary.
        """
        if not self.writer is None:
            self.writer.close()
            self.writer = None
        self.tensorboard_start()
        self.create_params_All()

class items_with_classes_record():
    def __init__(self, labels:torch.Tensor):
        self.labels = labels.unsqueeze(dim=-1).cpu()
        self.items = None
        self.predict = None
    
    def __call__(self, items:torch.Tensor, file = "Saves/items.csv"):
        self.storeItems(items)
        self.useItems(file)

    def storeItems(self, items:torch.Tensor):
        self.items = items.cpu()

    def useItems(self, file = "Saves/items.csv"):
        index_names = [f"Logit{x}" for x in range(len(self.items[0]))]
        if self.predict is None:
            items_with_labels = torch.concat([self.items,self.labels],dim=1)
        else:
            items_with_labels = torch.concat([self.items,self.predict,self.labels],dim=1)
            index_names.append("Prediction")
        index_names.append("Label")
        df = pd.DataFrame(items_with_labels.T,index=index_names).T
        df.to_csv(file,mode="a",header=(not os.path.exists(file)))
        self.items = None
        self.predict = None

    def storePredictions(self, predictions:torch.Tensor):
        assert predictions.dim() == 1
        self.predict = predictions.unsqueeze(dim=-1).cpu()