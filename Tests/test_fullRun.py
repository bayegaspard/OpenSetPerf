#three lines from https://xxx-cook-book.gitbooks.io/python-cook-book/content/Import/import-from-parent-folder.html
import os
import sys
root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)
sys.path.append(f"{root_folder}/src/main")
sys.path.append(f"{root_folder}/src/CodeFromImplementations")
import src.main.EndLayer as EndLayer
import src.main.Config as Config
import src.main.main as main
import torch
import src.main.FileHandling as FileHandling
import src.main.Dataload as Dataload
from torch.utils.data import DataLoader

Config.parameters["num_epochs"][0] = 1
Config.parameters["MaxPerClass"][0] = 10

def testrun():
    Config.unit_test_mode = True
    main.run_model()

def testrunall():
    Config.unit_test_mode = True
    for x in Config.alg:
        Config.parameters["OOD Type"][0] = x
        main.run_model()

def testMostSamples():
    Config.unit_test_mode = True
    Config.datapoints_per_class.sort()
    Config.parameters["MaxPerClass"][0] = Config.datapoints_per_class[-1]
    train, test, val = FileHandling.checkAttempLoad()

    #These lines initialize the loaders for the datasets.
    #Trainset is for training the model.
    trainset = DataLoader(train, Config.parameters["batch_size"][0], num_workers=Config.parameters["num_workers"][0],shuffle=True,pin_memory=True)  # for faster processing enable pin memory to true and num_workers=4
    #Validationset is for checking if the model got things correct with the same type of data as the trainset
    validationset = DataLoader(val, Config.parameters["batch_size"][0], shuffle=True, num_workers=Config.parameters["num_workers"][0],pin_memory=True)
    #Testset is for checking if the model got things correct with the Validationset+unknowns.
    testset = DataLoader(test, Config.parameters["batch_size"][0], shuffle=True, num_workers=Config.parameters["num_workers"][0], pin_memory=False)


    #testing
    if len(trainset)<100000 and Config.parameters["num_epochs"][0]>0:
        trainset = Dataload.recreateDL(trainset)
    if len(validationset)<100000 and Config.parameters["num_epochs"][0]>0:
        validationset = Dataload.recreateDL(validationset)
    if len(testset)<100000 and Config.parameters["num_epochs"][0]>0:
        testset = Dataload.recreateDL(testset)