#These are all the functions that dont fit elsewhere
import Config as Config
import os
import pandas as pd
import torch
import GPU
import FileHandling
from sklearn.metrics import (precision_score, recall_score, average_precision_score,accuracy_score)
from sklearn.metrics import confusion_matrix

#Translation dictionaries for algorithms that cannot have gaps in their numbers.
#So this block maps the knowns into numbers 0 to x where x is one less than the number of knowns
relabel = {Config.parameters["CLASSES"][0]:Config.parameters["CLASSES"][0]} #This one turns numbers into consecutive values
rerelabel = {Config.parameters["CLASSES"][0]:Config.parameters["CLASSES"][0]} #This one inverses numbers back into their values as specified by the dataloader
def setrelabel():
    """
    This function sets/resets the relabel dictionaries during loops where they could get messed up.

    it takes no parameters and changes the global variables relabel and rerelabel
    
    """
    global relabel,rerelabel

    #This is the mask to apply to tensors to make them ignore unknown classes.
    global mask
    mask = torch.zeros(Config.parameters["CLASSES"][0])
    for x in Config.helper_variables["knowns_clss"]:
        mask[x] = 1
    mask = mask==1 

    relabel = {Config.parameters["CLASSES"][0]:Config.parameters["CLASSES"][0]}
    rerelabel = {Config.parameters["CLASSES"][0]:Config.parameters["CLASSES"][0]}
    temp = 0
    for x in range(Config.parameters["CLASSES"][0]):
        if temp < len(Config.helper_variables["unknowns_clss"]) and x == Config.helper_variables["unknowns_clss"][temp]:
            temp = temp+1
        else:
            relabel[x] = x-temp
            rerelabel[x-temp] = x
    temp = None
setrelabel()

def makeConsecutive(logits:torch.Tensor,labels:torch.Tensor):
    """
    This function renames all of the classes so that all of the known classes are consecutive. This makes them easier to work with
    I wish I had just made the model have x outputs where x is the number of knowns instead of c outputs where c is the number of classes
    """
    global mask
    loge = logits[mask]
    newlabels = labels.clone()
    for x in Config.helper_variables["knowns_clss"]:
        newlabels[labels==x] = relabel[x]
    return loge, newlabels

    
def deleteSaves():
    """
    This deletes all model saves to prevent curruption

    Curruption is when a model has priviously been trained on data that is now in the test set
    This is possible because we randomize the dataloaders so it is best to retrain the model.
    """
    i = 0
    while os.path.exists(f"Saves/Epoch{i:03d}.pth"):
        os.remove(f"Saves/Epoch{i:03d}.pth")
        i = i+1



def printconfmat(outputs:torch.Tensor,labels:torch.Tensor):
    print(f"{confusion_matrix(labels,outputs.argmax(dim=1))}")





#Handels running the loop
def testRotate(notes=(0,0,0)):
    """
    testRotate() is one of the two functions that modifies the global variables in Config.py.
    Specifically this function is called when the config LOOP is set to 1. 
    And it will rotate the parameters in Config.py to go through all the variations of the parameters.
    After calling testRotate() running the model with the function main.run_model() will use diffrent parameters.
    It will automatically change the Config parameter LOOP to 0 and return False to indicate that it is done.

    testRotate() takes one parameter:
        notes - notes is a tuple containing three integers that mark the current position of the loop.
            At the start notes should be (0,0,0) it will then increment to (0,1,0) and it will be returned.
            At the next call of testRotate() please give it the privious output of the function to allow it to continue.
            After all parameters have been looped, testRotate() will return False instead of a tuple.
    
    testRotate() returns one of two parameters:
        notes - a tuple containing three ints that marks the current position of the loop.
        False - a false value is returned if the loop has been entirely completed.
            
    """
    global relabel,rerelabel
    stage = notes[0]
    step = notes[1]
    al = notes[2]

    deleteSaves()
    if step+1 < len(Config.loops[stage]):
        step = step+1

        if Config.loops2[stage] == "optimizer":
            Config.parameters[Config.loops2[stage]] = Config.loops[stage][step]
        elif Config.loops2[stage] == "Unknowns":
            Config.helper_variables["unknowns_clss"] = Config.loops[stage][step]
            Config.parameters["Unknowns"] = f"{len(Config.loops[stage][step])} Unknowns"
            Config.helper_variables["knowns_clss"] = Config.loopOverUnknowns(Config.helper_variables["unknowns_clss"])
            setrelabel()
        else:
            Config.parameters[Config.loops2[stage]][0] = Config.loops[stage][step]

        return (stage,step,al)

    #reset this stage
    step = 0

    if Config.loops2[stage] == "optimizer":
        Config.parameters[Config.loops2[stage]] = Config.loops[stage][step]
    elif Config.loops2[stage] == "Unknowns":
        Config.helper_variables["unknowns_clss"] = Config.loops[stage][step]
        Config.parameters["Unknowns"] = f"{len(Config.loops[stage][step])} Unknowns"
        Config.helper_variables["knowns_clss"] = Config.loopOverUnknowns(Config.helper_variables["unknowns_clss"])
        setrelabel()
    else:
        Config.parameters[Config.loops2[stage]][0] = Config.loops[stage][step]

    #Go to next stage
    if stage+1 < len(Config.loops):
        stage = stage+1
        #Skip the next rotate algorithm step and just go to rotate step
        return testRotate((stage,step,al))

    #Reset stage
    stage = 0

    if al+1 < len(Config.alg):
        al = al+1
        Config.parameters["OOD Type"][0] = Config.alg[al]
        return (stage,step,al)
    

    #Done with looping
    Config.parameters["LOOP"][0] = False
    return False

def incrementLoop(notes=(0)):
    """
    incrementLoop() is one of the two functions that modifies the global variables in Config.py.
    Specifically this function is called when the config LOOP is set to 2. 
    And it will incrementally introduce unknowns into the training data, this is done to get data for the resiliance tesing group.
    After calling incrementLoop() running the model with the function main.run_model() will continue to train the same model and use fewer unknowns.
    It will automatically change the Config parameter LOOP to 0 and return False to indicate that it is done.

    incrementLoop() takes one parameter:
        notes - integer, stores the current position of the loop.
    
    incrementLoop() returns one of two values:
        notes - an integer that marks the current position of the loop to be fed back trough the function.
        False - outputs false when the loop has been entirely completed.
    
    """
    Config.parameters["attemptLoad"][0] = 1
    notes = notes+1
    if notes >= len(Config.incGroups):
        Config.parameters["LOOP"][0] = False
        return False
    Config.helper_variables["unknowns_clss"] = Config.incGroups[notes]
    Config.parameters["Unknowns"] = f"{len(Config.incGroups[notes])} Unknowns"
    Config.helper_variables["knowns_clss"] = Config.loopOverUnknowns(Config.incGroups[notes])
    setrelabel()

    #Find diffrence with this code: https://stackoverflow.com/a/3462160
    FileHandling.incrementLoopModData(list(set(Config.incGroups[notes-1])-set(Config.incGroups[notes])))
    return notes

#Resiliance loop
def resilianceLoop():
    current = Config.parameters["loopLevel"][0]
    if current+1<len(pd.read_csv("datasets/percentages.csv", index_col=None)):
        Config.parameters["loopLevel"][0] = current+1
    else:
        Config.parameters["LOOP"][0] = 0

#This puts the notes into a readable form
#notes are how it keeps track of where in the loop it is.
def getcurrentlychanged(notes):
    """
    getcurrentlychanged() turns the notes from the function testRotate() into a readable string to tag data with.

    it takes one parameter:
        -notes, a three integer tuple.
    it outputs a string saying what algorithm is being used with what changing parameter and the current setting of that parameter
    """

    algorithm = Config.alg[notes[2]]
    currentlyChanging = Config.loops2[notes[0]]
    currentSetting = Config.loops[notes[0]][notes[1]]
    return str(algorithm)+" "+str(currentlyChanging)+" "+str(currentSetting)

#This bit of code will loop through the entire loop and print all of the variations.
def looptest():
    """
    This is the testing code for the testRotate() function.
    It will print out all of the currently possible variations of the parameters and the number of parameters that is.
    No parameters or outputs
    """
    out = pd.DataFrame(())
    out2 = pd.DataFrame(())

    count = 0
    notes = (0,0,0)
    while notes:
        current = pd.DataFrame(Config.parameters)
        current2 = pd.DataFrame(Config.helper_variables["unknowns_clss"])
        out = pd.concat([out,current.iloc[0]],axis=1)
        out2 = pd.concat([out2,current2],axis=1)
        print(getcurrentlychanged(notes))
        notes = testRotate(notes)
        count = count+1

    out = pd.concat([current.iloc[0],out],axis=1)

    out.to_csv("Testing.csv")
    out2.to_csv("Testing2.csv")
    print(f"That means the model will have to run {count} times")


class NoExamples(Exception):
    """
    If the Openmax algorithm is unable to find a correct example for every class (that is the model's prediction==class) openmax will attempt to concatinate a blank array.
    This problem causes an error to be thrown but I was unable to find the type of error to catch it. So I have made this error class.
    This error class is thrown in the Openmax algorithm if any of the classes have an empy correct predictions array.
    
    """
    pass


device = GPU.get_default_device() # selects a device, cpu or gpu

class LossPerEpoch():
    """
    This is a datacollection class. It wants to collect the sum of loss across the entirty of training.
    To do this it collects each partial sum in the local variables using addloss() and returns the total when collect() is called.
    
    """
    def __init__(self,name):
        self.loss = 0
        self.name = name

    def addloss(self,predicted:torch.Tensor, target:torch.Tensor):
        """
        addloss() stores the number of times predicted==targets.

        it takes two parameters:
            -predicted: the values that the model has predicted for each class.
            -targets: the values that are represented by each class.
        
        """
        target = GPU.to_device(target,device)
        predicted = GPU.to_device(predicted,device)
        locations = predicted!=target
        self.loss += locations.sum().item()

    def collect(self):
        """
        Collect() gets the data gathered by the addloss() function and resets the stored loss to zero.
        The data is then stored in the file Saves/Scoresall.csv in the most recent line under "Number Of Failures"
        """
        #if os.path.exists(os.path.join("Saves",self.name)):
        #    hist = pd.read_csv(os.path.join("Saves",self.name),index_col=0)
        #else:
        #    hist = pd.DataFrame([])
        #param = pd.DataFrame(Config.parameters).iloc[0]


        #current = pd.DataFrame({"Number of failures":[self.loss]})
        #current = pd.concat([param,current])
        #param["Number Of Failures"] = self.loss

        #hist = pd.concat([hist,param],axis=1)
        #hist.to_csv(os.path.join("Saves",self.name))
        FileHandling.addMeasurement("Number Of Failures",self.loss)
        self.loss = 0

def getFscore(dat):
    """
    Takes the scores saved by the model as it is running and then translates those into the four main scores.

    getFscore() parameter:
        -data in the form of a three item tuple. these should all be torch Tensors.
        -The first value is the model's prediction for a given line
        -The second value is the true value for a given line
        -The third value is what the model was tested agianst, this should be the same as the second value except in cases where some of the lines are unknown.
            If some of the lines are unknown those lines should be '15' in tested_against.
        
    Returns:
        a tuple containing:
        f1 score- the harmonic mean between precision and recall.
        recall- How many of the per-class positives did the model find.
        precision- How many of the model's guesses were correct
        accuracy- Number of places the labels matched over total number.
    """
    y_pred,y_true,y_tested_against = dat
    y_pred = y_pred / (Config.parameters["CLASSES"][0]/Config.parameters["CLASSES"][0]) #The whole config thing is if we are splitting the classes further
    y_true = y_true / (Config.parameters["CLASSES"][0]/Config.parameters["CLASSES"][0])
    y_true = y_true.to(torch.int).tolist()
    y_pred = y_pred.to(torch.int).tolist()
    y_tested_against = y_tested_against.to(torch.int).tolist()
    print(confusion_matrix(y_tested_against,y_pred))
    recall = recall_score(y_tested_against,y_pred,average='weighted',zero_division=0)
    precision = precision_score(y_tested_against,y_pred,average='weighted',zero_division=0)
    if precision==0 and recall==0:
        f1 =0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    accuracy = accuracy_score(y_tested_against,y_pred)
    return f1,recall,precision,accuracy

def getFoundUnknown(dat):
    """
    Takes the scores saved by the model as it is running and then translates those into a score representing how many of the unknowns were found.

    getFscore() parameter:
        -data in the form of a three item tuple. these should all be torch Tensors.
        -The first value is the model's prediction for a given line
        -The second value is the true value for a given line
        -The third value is what the model was tested agianst, this should be the same as the second value except in cases where some of the lines are unknown.
            If some of the lines are unknown those lines should be '15' in tested_against.
        
    Returns:
        recall (int)- How many of the per-class positives did the model find, specifically for unknowns.
    """
    y_pred,y_true,y_tested_against = dat
    y_pred = y_pred / (Config.parameters["CLASSES"][0]/Config.parameters["CLASSES"][0]) #The whole config thing is if we are splitting the classes further
    y_true = y_true / (Config.parameters["CLASSES"][0]/Config.parameters["CLASSES"][0])
    y_true = y_true.to(torch.int).tolist()
    y_pred = y_pred.to(torch.int).tolist()
    y_tested_against = y_tested_against.to(torch.int).tolist()
    recall = recall_score(y_tested_against,y_pred,average=None,zero_division=0)
    if (recall is float):
        return recall
    elif (recall is None or len(recall)==0):
        return 0
    else:
        return recall[-1]

if __name__ == "__main__":
    looptest()
    print(f"Torch cuda utilizaton percent: {torch.cuda.utilization()}")




