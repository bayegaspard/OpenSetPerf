import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import (precision_score, recall_score, average_precision_score)
import numpy as np
import torch
import time

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
    """
    run_model() takes no parameters and runs the model according to the current model configurations in Config.py
    run_model() does not return anything.
    
    """
    #This refreshes all of the copies of the Config files, the copies can be used to find the current config if something breaks.
    #At this point these copies are mostly redundent.
    FileHandling.refreshFiles(root_path)

    FileHandling.generateHyperparameters(root_path) # generate hyper parameters copy files if they did not exist.

    #Get the configs from the copy file because we had not integrated Config.py as well when this was written.
    batch_size,num_workers,attemptLoad,testlen,num_epochs,lr,threshold,model_type,datagroup,unknownVals = FileHandling.readCSVs(root_path)
    #This is an example of how we get the values from Config now.
    knownVals = Config.helper_variables["knowns_clss"]

    #This just helps translate the config strings into model types. It is mostly unnesisary.
    model_list = {"Convolutional":ModelStruct.Conv1DClassifier,"Fully_Connected":ModelStruct.FullyConnected}
    model = model_list[model_type]() # change index to select a specific architecture.

    #This initializes the data-parallization which hopefully splits the training time over all of the connected GPUs
    model = ModelStruct.ModdedParallel(model)

    #This selects what algorithm you are using.
    model.end.type = Config.parameters["OOD Type"][0]

    #This selects the default cutoff value
    model.end.cutoff = threshold

    #This creates the datasets assuming there are not saved datasets that it can load.
    #By default the saved datasets will be deleted to avoid train/test corruption but this can be disabled.
    #The dataset files are stored in saves as .pt (Pytorch) files
    train, test, val = FileHandling.checkAttempLoad(root_path)

    #These lines initialize the loaders for the datasets.
    #Trainset is for training the model.
    trainset = DataLoader(train, batch_size, num_workers=num_workers,shuffle=True,
            pin_memory=True)  # for faster processing enable pin memory to true and num_workers=4
    #Validationset is for checking if the model got things correct with the same type of data as the trainset
    validationset = DataLoader(val, batch_size, shuffle=True, num_workers=num_workers,pin_memory=True)
    #Testset is for checking if the model got things correct with the Validationset+unknowns.
    testset = DataLoader(test, batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)


    #testing
    if len(trainset)<100000 and Config.parameters["num_epochs"][0]>0:
        trainset = Dataload.recreateDL(trainset)
    if len(validationset)<100000 and Config.parameters["num_epochs"][0]>0:
        validationset = Dataload.recreateDL(validationset)
    if len(testset)<100000 and Config.parameters["num_epochs"][0]>0:
        testset = Dataload.recreateDL(testset)


    print("length of train",len(train),"\nlength of test",len(test))

    #This sets the device for each of the datasets to work with the data-parallization
    train_loader =  GPU.DeviceDataLoader(trainset, device)
    val_loader = GPU.DeviceDataLoader(validationset, device)
    test_loader = GPU.DeviceDataLoader(testset, device)

    
    #Loop 2 uses the same model for each loop so it specifically loads the model.
    if Config.parameters["LOOP"][0] == 2:
        model.loadPoint("Saves")

    #This array stores the 'history' data, I am not sure what data that is
    history_final = []
    #This gives important information to the endlayer for some of the algorithms
    model.end.prepWeibull(train_loader,device,model)


    starttime = time.time()
    #Model.fit is what actually runs the model. It outputs some kind of history array?
    history_final += model.fit(num_epochs, lr, train_loader, test_loader,val_loader, opt_func=opt_func)


    #This big block of commented code is to create confusion matricies that we thought could be misleading,
    #   so it is commented out.
    np.set_printoptions(precision=1)
    class_names = Dataload.get_class_names(knownVals) #+ Dataload.get_class_names(unknownVals)
    class_names.append("Unknown")
    class_names = Dataload.get_class_names(range(Config.parameters["CLASSES"][0]))
    for x in unknownVals:
        class_names[x] = class_names[x]+"*"
    class_names.append("*Unknowns")
    print("class names", class_names)



    FileHandling.addMeasurement(f"Length train",len(train))
    FileHandling.addMeasurement(f"Length validation",len(val))
    FileHandling.addMeasurement(f"Length test",len(test))

    #Validation values
    f1, recall, precision, accuracy = helperFunctions.getFscore(model.store)
    FileHandling.addMeasurement("Val_F1",f1)
    FileHandling.addMeasurement("Val_Recall",recall)
    FileHandling.addMeasurement("Val_Precision",precision)
    FileHandling.addMeasurement("Val_Accuracy",accuracy)


    #Sets the model to really be sure to be on evaluation mode and not on training mode. (Affects dropout)
    if not Config.parameters["LOOP"][0]:
        #More matrix stuff that we removed.
        cnf_matrix = plots.confusionMatrix(model.store) 
        plots.plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                    title=f'{Config.parameters["OOD Type"][0]} Validation', knowns = knownVals)

    #Resets the stored values that are used to generate the above values.
    model.storeReset()

    #model.evaluate() runs only the evaluation stage of running the model. model.fit() calls model.evaluate() after epochs
    model.evaluate(test_loader)
    
    model.eval()

    
    

    
    #this creates plots as long as the model is not looping. 
    # It is annoying when the model stops just to show you things when you are trying to run the model overnight
    if not Config.parameters["LOOP"][0]:
        plots.plot_all_losses(history_final)
        plots.plot_losses(history_final)
        plots.plot_accuracies(history_final)

    
    

   
    
    


    

    #Generates the values when unknowns are thrown in to the testing set.
    f1, recall, precision, accuracy = helperFunctions.getFscore(model.store)
    FileHandling.addMeasurement("Test_F1",f1)
    FileHandling.addMeasurement("Test_Recall",recall)
    FileHandling.addMeasurement("Test_Precision",precision)
    FileHandling.addMeasurement("Test_Accuracy",accuracy)
    FileHandling.addMeasurement("Found_Unknowns",helperFunctions.getFoundUnknown(model.store))

    FileHandling.create_params_Fscore(root_path,f1)

    if not Config.parameters["LOOP"][0]:
        #More matrix stuff that we removed.
        cnf_matrix = plots.confusionMatrix(model.store) 
        plots.plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                    title=f'{Config.parameters["OOD Type"][0]} Test', knowns = knownVals)


    
    #This stores and prints the final results.
    score_list = [recall,precision,f1]
    FileHandling.write_hist_to_file(history_final,num_epochs,model.end.type)
    FileHandling.write_scores_to_file(score_list,num_epochs,model.end.type)
    print("Type : ",model.end.type)
    print(f"Now changing : {plots.name_override}")
    print(f"F-Score : {f1*100:.2f}%")
    print(f"Precision : {precision*100:.2f}%")
    print(f"Recall : {recall*100:.2f}%")

    #Use Softmax to test.
    model.end.type = "Soft"
    model.storeReset()
    model.evaluate(val_loader)

    #Validation values
    f1, recall, precision, accuracy = helperFunctions.getFscore(model.store)
    FileHandling.addMeasurement("Soft_Val_F1",f1)
    FileHandling.addMeasurement("Soft_Val_Recall",recall)
    FileHandling.addMeasurement("Soft_Val_Precision",precision)
    FileHandling.addMeasurement("Soft_Val_Accuracy",accuracy)

    if not Config.parameters["LOOP"][0]:
        #More matrix stuff that we removed.
        cnf_matrix = plots.confusionMatrix(model.store) 
        plots.plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                    title=f'Soft Validation', knowns = knownVals)
    model.storeReset()
    model.evaluate(test_loader)

    #Validation values
    f1, recall, precision, accuracy = helperFunctions.getFscore(model.store)
    FileHandling.addMeasurement("Soft_Test_F1",f1)
    FileHandling.addMeasurement("Soft_Test_Recall",recall)
    FileHandling.addMeasurement("Soft_Test_Precision",precision)
    FileHandling.addMeasurement("Soft_Test_Accuracy",accuracy)

    if not Config.parameters["LOOP"][0]:
        #More matrix stuff that we removed.
        cnf_matrix = plots.confusionMatrix(model.store) 
        plots.plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                    title=f'Soft Test', knowns = knownVals)
    model.storeReset()

    #This selects what algorithm you are using.
    model.end.type = Config.parameters["OOD Type"][0]


    #This is the code to actually start showing the plots.
    #Again, we do not want this activating while running overnight.
    if not Config.parameters["LOOP"][0]:
        plt.show()


    #This loops through a list of "Threshold" values because they do not require retraining the model.
    if Config.parameters["LOOP"][0] == 1:
        model.thresholdTest(test_loader)


def main():
    """
    The main function
    This is what is run to start the model.
    Takes no arguements and returns nothing.
    
    """
    #Finds the current working directory.
    global root_path
    #If the current working directory is in the wrong location it changes the current working directory and prints an error.
    while (os.path.basename(root_path) == "main.py" or os.path.basename(root_path) == "main" or os.path.basename(root_path) == "src"):
        #checking that you are running from the right folder.
        print(f"Please run this from the source of the repository not from {os.path.basename(root_path)}. <---- Look at this!!!!")
        os.chdir("..")
        root_path=os.getcwd()

    #This is what the diffrent plots have overriding their names. If it is a loop it changes for every iteration
    plots.name_override = "Config File settings"

    #Deletes all privious model saves before running
    helperFunctions.deleteSaves()

    #Runs the model
    run_model()

    #If it is loop type 1 (changing parameters loop):
    if Config.parameters["LOOP"][0] == 1:
        step = (0,0,0) #keeps track of what is being updated.

        #Loops until the loop function disables the loop.
        while Config.parameters["LOOP"][0]:
            #The function testRotate changes the values in Config.py, it is treating those as global veriables.
            #I know this is bad code but if it is only changed in specific places it is not awful.
            step = helperFunctions.testRotate(step)

            #If it did not hit the end of the loop (Loop end returns False)
            if step:
                #Reset pyplot
                plt.clf()

                #Change the name override to accurately state what has changed
                plots.name_override = helperFunctions.getcurrentlychanged(step)
                #This is to change the level of detail on the confusion matricies (Not needed anymore)
                plt.figure(figsize=(4,4))

                #State what is changing for bugfixing.
                print(f"Now changing: {plots.name_override}")

                #Finally run the loop.
                run_model()
                FileHandling.addMeasurement("Currently Modifying",plots.name_override)

    
    #If it is loop type 2 (iterative unknowns loop):
    #Same structure as above.
    elif Config.parameters["LOOP"][0] == 2:
        step = (0) 
        while Config.parameters["LOOP"][0]:
            step = helperFunctions.incrementLoop(step)
            if step:
                plt.clf()
                plots.name_override = f"Incremental with {Config.parameters['Unknowns']} unknowns"
                plt.figure(figsize=(4,4))
                print(f"unknowns: {Config.helper_variables['unknowns_clss']}")
                run_model()
                FileHandling.addMeasurement("Currently Modifying",plots.name_override)


if __name__ == '__main__':
    main()




