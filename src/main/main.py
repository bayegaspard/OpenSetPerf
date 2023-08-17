import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
import torch
import time
import os
from sklearn.metrics import roc_auc_score, roc_curve, RocCurveDisplay
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

# user defined modules
import GPU, FileHandling
import plots
import Dataload
import ModelStruct
import Config
import helperFunctions
import GenerateImages

root_path = os.getcwd()

if __name__ == "__main__":
    print(torch.__version__)


#useful variables
opt_func = Config.parameters["optimizer"]
device = GPU.get_default_device() # selects a device, cpu or gpu

def run_model(measurement=None, graphDefault=False):
    """
    run_model() takes up to one parameter and runs the model according to the current model configurations in Config.py
    parameter:
        graphDefault - by default create graphs
        Measurement - is a function that stores data gathered from the model, the data is in the form of (type_of_data,value_of_data)
    run_model() does not return anything but outputs are saved in Saves/
    
    """
    if measurement is None:
        measurement = FileHandling.Score_saver()
    #This refreshes all of the copies of the Config files, the copies can be used to find the current config if something breaks.
    #At this point these copies are mostly redundent.
    # FileHandling.refreshFiles(root_path)

    # FileHandling.generateHyperparameters(root_path) # generate hyper parameters copy files if they did not exist.

    #This is an example of how we get the values from Config now.
    knownVals = Config.parameters["Knowns_clss"][0]

    #This creates the datasets assuming there are not saved datasets that it can load.
    #By default the saved datasets will be deleted to avoid train/test corruption but this can be disabled.
    #The dataset files are stored in saves as .pt (Pytorch) files
    train, test, val = FileHandling.checkAttempLoad(root_path)

    #This just helps translate the config strings into model types. It is mostly unnesisary.
    model_list = {"Convolutional":ModelStruct.Conv1DClassifier,"Fully_Connected":ModelStruct.FullyConnected}
    model = model_list[Config.parameters["model"][0]](mode=Config.parameters["OOD Type"][0],numberOfFeatures=FileHandling.getDatagroup()[0].data_length) # change index to select a specific architecture.

    #This initializes the data-parallization which hopefully splits the training time over all of the connected GPUs
    model = ModelStruct.ModdedParallel(model)#I dont atcutally think this works the way we are using it, trying something new.
    assert isinstance(model.module,ModelStruct.AttackTrainingClassification)#adding this purely for the linter, so that it knows exactly what the model is.

    #This selects the default cutoff value
    model.module.end.cutoff = Config.parameters["threshold"][0]


    #These lines initialize the loaders for the datasets.
    #Trainset is for training the model.
    trainset = DataLoader(train, Config.parameters["batch_size"][0], num_workers=Config.parameters["num_workers"][0],shuffle=True,pin_memory=True)  # for faster processing enable pin memory to true and num_workers=4
    #Validationset is for checking if the model got things correct with the same type of data as the trainset
    validationset = DataLoader(val, Config.parameters["batch_size"][0], shuffle=True, num_workers=Config.parameters["num_workers"][0],pin_memory=True)
    #Testset is for checking if the model got things correct with the Validationset+unknowns.
    testset = DataLoader(test, Config.parameters["batch_size"][0], shuffle=True, num_workers=Config.parameters["num_workers"][0], pin_memory=False)

    #                               made this into a unit test
    # for batch in validationset:
    #     for x in Config.parameters["Unknowns_clss"][0]:
    #         assert x not in batch[1][1]
    #     for x in batch[1]:
    #         assert x[1] in Config.parameters["Knowns_clss"][0]


    #Recreating the datasets so that they are in memory.
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
        model.module.loadPoint("Saves")


    #This array stores the 'history' data, I am not sure what data that is
    history_final = []
    #This gives important information to the endlayer for some of the algorithms
    model.module.end.prepWeibull(trainset,device,model)


    starttime = time.time()
    #Model.fit is what actually runs the model. It outputs some kind of history array?
    history_final += model.module.fit(Config.parameters["num_epochs"][0], Config.parameters["learningRate"][0], train_loader, test_loader,val_loader, opt_func=opt_func, measurement=measurement)

    model.module.batchSaveMode(function=measurement)

    #This big block of commented code is to create confusion matricies that we thought could be misleading,
    #   so it is commented out.
    np.set_printoptions(precision=1)
    class_names = Dataload.get_class_names(knownVals) #+ Dataload.get_class_names(unknownVals)
    class_names.append("Unknown")
    class_names = Dataload.get_class_names(range(Config.parameters["CLASSES"][0]))
    for x in Config.parameters["Unknowns_clss"][0]:
        class_names[x] = class_names[x]+"*"
    class_names.append("*Unknowns")
    #print("class names", class_names)



    measurement(f"Length train",len(train))
    measurement(f"Length validation",len(val))
    measurement(f"Length test",len(test))

    #Validation values
    f1, recall, precision, accuracy = helperFunctions.getFscore(model.store)
    measurement("Val_F1",f1)
    measurement("Val_Recall",recall)
    measurement("Val_Precision",precision)
    measurement("Val_Accuracy",accuracy)


    #Sets the model to really be sure to be on evaluation mode and not on training mode. (Affects dropout)
    if not Config.parameters["LOOP"][0] and graphDefault:
        #More matrix stuff that we removed.
        cnf_matrix = plots.confusionMatrix(model.store) 
        plots.plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                    title=f'{Config.parameters["OOD Type"][0]} Validation', knowns = knownVals)



    runExistingModel(model,test_loader,"Test",history_final,class_names, graphDefault=graphDefault,print_vals=True,measurement=measurement)

    


    #AUTOTHRESHOLD

    # if model.end.type != "Soft":
        # model.thresholdTest(test_loader)
        # roc = RocCurveDisplay.from_predictions(model.end.rocData[0],model.end.rocData[1],name=model.end.type)
        # roc.plot()
        # plt.show()
    if (not torch.all(model.module.end.rocData[0])) and (not torch.all(model.module.end.rocData[0]==False)):
        if isinstance(model.module.end.rocData[0],torch.Tensor):
            model.module.end.rocData[0] = model.end.rocData[0].cpu().numpy()
        if isinstance(model.module.end.rocData[1],torch.Tensor):
            model.module.end.rocData[1] = model.module.end.rocData[1].cpu().numpy()
        
        #isnan gotten from https://stackoverflow.com/a/913499
        if (np.isnan( model.module.end.rocData[1]).any()):
            model.module.end.cutoff = -1
        else:
            roc_data = pd.DataFrame(roc_curve(model.module.end.rocData[0],model.module.end.rocData[1]))
            if hasattr(measurement,"writer") and measurement.writer is not None:
                measurement.writer.add_pr_curve("PR curve for unknowns vs knowns",np.array(model.module.end.rocData[0]).squeeze(),np.array(model.module.end.rocData[1]).squeeze())
            #https://stackoverflow.com/a/62329743 (unused)
            
            new_row = ((1-roc_data.iloc[0])*roc_data.iloc[1])
            #New row code: https://stackoverflow.com/a/72084365 (why was this so difficult, it is literally just adding a new row?)
            roc_data = pd.concat([roc_data,new_row.to_frame().T]).reset_index(drop=True)
            roc_data.to_csv(f"Saves/roc/ROC_data_{Config.parameters['OOD Type'][0]}.csv")
            if len(roc_data.iloc[2][roc_data.iloc[1]>0.95])>0:
                model.module.end.cutoff = roc_data.iloc[2][roc_data.iloc[1]>0.95].iloc[0]
                if model.module.end.type == "Energy":
                    model.module.end.cutoff = -model.module.end.cutoff

        runExistingModel(model,test_loader,"AUTOTHRESHOLD_Test",history_final,class_names,measurement=measurement)
        runExistingModel(model,val_loader,"AUTOTHRESHOLD_Val",history_final,class_names,measurement=measurement)

        measurement("AUTOTHRESHOLD",model.end.cutoff)
        measurement("AUTOTHRESHOLD_Trained_on_length",len(model.module.end.rocData[0]))

        if isinstance(model.module.end.rocData[0],torch.Tensor):
            model.module.end.rocData[0] = model.module.end.rocData[0].cpu().numpy()
        if isinstance(model.module.end.rocData[1],torch.Tensor):
            model.module.end.rocData[1] = model.module.end.rocData[1].cpu().numpy()
        if not (np.isnan( model.module.end.rocData[1]).any()):
            model.module.end.cutoff = roc_data.iloc[2][roc_data.iloc[3].idxmax()]
            if model.module.end.type == "Energy":
                model.module.end.cutoff = -model.module.end.cutoff
            runExistingModel(model,test_loader,"AUTOTHRESHOLD2_Test",history_final,class_names,measurement=measurement)
            runExistingModel(model,val_loader,"AUTOTHRESHOLD2_Val",history_final,class_names,measurement=measurement)







    if False:
        #Use Softmax to test.
        model.end.type = "Soft"
        model.storeReset()
        runExistingModel(model,val_loader,"Soft_Val",history_final,class_names,measurement=measurement)
        model.storeReset()
        runExistingModel(model,test_loader,"Soft_Test",history_final,class_names,measurement=measurement)


    


    plt.close()

def runExistingModel(model:ModelStruct.AttackTrainingClassification,data,name,history_final,class_names,measurement=None,graphDefault = False, print_vals = False):
    """
    Runs an existing and loaded model.
    
    Parameters:
        Model - loaded model object with preset weights
        data - dataloader used to generate the data
        name - string to specify the name of the collected data
        history_final - results from fitting the model
        class_names - generated class names for the confusion matrix
        measurement (optional) - Function that is passed the results from the model in the form of (type_of_data,value_of_data)
        graphDefault - the default for desplaying graphs, normally False
    """
    if measurement is None:
        measurement = FileHandling.Score_saver()
    #Resets the stored values that are used to generate the above values.
    model.storeReset()

    #model.evaluate() runs only the evaluation stage of running the model. model.fit() calls model.evaluate() after epochs
    model.evaluate(data)
    
    model.eval()

    
    #this creates plots as long as the model is not looping. 
    # It is annoying when the model stops just to show you things when you are trying to run the model overnight
    if not Config.parameters["LOOP"][0] and graphDefault:
        plots.plot_all_losses(history_final)
        plots.plot_losses(history_final)
        plots.plot_accuracies(history_final)


    #Generates the values when unknowns are thrown in to the testing set.
    f1, recall, precision, accuracy = helperFunctions.getFscore(model.store)
    measurement(f"{name}_F1",f1)
    measurement(f"{name}_Recall",recall)
    measurement(f"{name}_Precision",precision)
    measurement(f"{name}_Accuracy",accuracy)
    unknowns_scores = helperFunctions.getFoundUnknown(model.store)
    measurement(f"{name}_Found_Unknowns",unknowns_scores[0]) #recall
    measurement(f"{name}_Unknowns_accuracy",unknowns_scores[1])

    FileHandling.create_params_Fscore(root_path,f1)

    if not Config.parameters["LOOP"][0] and graphDefault:
        #More matrix stuff that we removed.
        cnf_matrix = plots.confusionMatrix(model.store) 
        plots.plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title=f'{Config.parameters["OOD Type"][0]} Test', knowns = Config.parameters["Knowns_clss"][0])


    
    #This stores and prints the final results.
    score_list = [recall,precision,f1]
    FileHandling.write_hist_to_file(history_final,Config.parameters["num_epochs"][0],model.end.type)
    FileHandling.write_scores_to_file(score_list,Config.parameters["num_epochs"][0],model.end.type)
    if print_vals:
        print("Type : ",model.end.type)
        print(f"Now changing : {plots.name_override}")
        print(f"F-Score : {f1*100:.2f}%")
        print(f"Precision : {precision*100:.2f}%")
        print(f"Recall : {recall*100:.2f}%")

def loopType1(main=run_model,measurement=None):
    """
    Tests if loop type 1 is true and if it is runs loop type 1.
    Note, this should be run after the model is run for the first time.

    Loop 1 loops through the changes found the Config.loops and Config.loops2 arrays. 
    (loops2 is the names for the values in loops, could have made it a dictionary)
    If a value is not being changed it is assumed to be the default which is in position 0 of the specific array in loops.

    parameters:
        main - the main function to run, this should be run_model unless being tested.
        measurement - Function that is passed the results from the model in the form of (type_of_data,value_of_data)
    """
    if measurement is None:
        measurement = FileHandling.Score_saver()
    #If it is loop type 1 (changing parameters loop):
    if Config.parameters["LOOP"][0] == 1:
        FileHandling.Score_saver.create_loop_history_(name="LoopRan.csv")
        step = (0,0,0) #keeps track of what is being updated.
        measurement("Currently Modifying","Default")
        measurement("Type of modification","Default")
        measurement("Modification Level","Default")
        measurement.start()

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
                main()
                measurement("Currently Modifying",plots.name_override)
                measurement("Type of modification",helperFunctions.getcurrentlychanged_Stage(step))
                index = measurement("Modification Level",helperFunctions.getcurrentlychanged_Step(step))
                if not index is None:
                    FileHandling.Score_saver.addMeasurement_(plots.name_override,index,fileName="LoopRan.csv")
                else:
                    FileHandling.Score_saver.addMeasurement_(plots.name_override,"Done",fileName="LoopRan.csv")

def loopType2(main=run_model,measurement=None):
    """
    Tests if loop type 2 is true and if it is runs loop type 2.
    Note, this should be run after the model is run for the first time.

    Loop 2 will train a model on a specific set of classes 
    and then slowly introduce new classes into the training set from the unknown set.
    This should tell us how well the model retrains after something new is added.

    parameters:
        main - the main function to run, this should be run_model unless being tested.
        measurement - Function that is passed the results from the model in the form of (type_of_data,value_of_data)
    """
    if measurement is None:
        measurement = FileHandling.Score_saver()
    #If it is loop type 2 (iterative unknowns loop):
    #Same structure as above.
    if Config.parameters["LOOP"][0] == 2:
        step = (0) 
        measurement("Currently Modifying",f"Incremental TRAINING with {Config.parameters['Unknowns']} unknowns")
        measurement.start()
        while Config.parameters["LOOP"][0]:
            step = helperFunctions.incrementLoop(step)
            if step:
                plt.clf()
                plots.name_override = f"Incremental with {Config.parameters['Unknowns']} unknowns"
                plt.figure(figsize=(4,4))
                print(f"unknowns: {Config.parameters['Unknowns_clss'][0]}")
                main()
                measurement("Currently Modifying",plots.name_override)

def loopType3(main=run_model,measurement=None):
    """
    Tests if loop type 3 is true and if it is runs loop type 3.
    Note, this should be run after the model is run for the first time.

    Loop type 3 runs through each line of the percentages file (found in Datasets)
    and makes a dataloader of the percentage mix of classes specified in the file.
    Such as 25% class 1, 12% class 2... and so forth. 

    parameters:
        main - the main function to run, this should be run_model unless being tested.
        measurement - Function that is passed the results from the model in the form of (type_of_data,value_of_data)
    """
    if measurement is None:
        measurement = FileHandling.Score_saver()
    if Config.parameters["LOOP"][0] == 3:
        measurement.start()
        while Config.parameters["LOOP"][0]:
            helperFunctions.resilianceLoop()
            plt.clf()
            plots.name_override = f"Reisiliance with {Config.parameters['Unknowns']} unknowns"
            plt.figure(figsize=(4,4))
            measurement(f"Row of percentages", Config.parameters['loopLevel'])
            main()

def loopType4(main=run_model,measurement=None):
    """
    Loop type 4 runs thrugh every line of datasets/hyperparamList.csv and changes any valid Config parameters to match.
    This means that you can predefine specific lists of hyperparameters to loop through. 

    parameters:
        main - the main function to run, this should be run_model unless being tested.
        measurement - Function that is passed the results from the model in the form of (type_of_data,value_of_data)
    """
    if measurement is None:
        measurement = FileHandling.Score_saver()
    if Config.parameters["LOOP"][0] == 4:
        row = 0
        while Config.parameters["LOOP"][0]:
            measurement(f"Row of defined hyperparameter csv: ", row)
            measurement.start()
            row = helperFunctions.definedLoops(row=row)
            plots.name_override = f"Predefined loop row {row}"
            main()

def main_start():
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

    measurement = FileHandling.Score_saver()

    #Runs the model
    run_model(measurement=measurement)

    loopType1(run_model,measurement)
    loopType2(run_model,measurement)
    loopType3(run_model,measurement)
    loopType4(run_model,measurement)
    if hasattr(measurement,"writer"):
        measurement.writer.close()
    GenerateImages.main()
    


if __name__ == '__main__':
    main_start()




