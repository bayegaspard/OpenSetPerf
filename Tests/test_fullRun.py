#three lines from https://xxx-cook-book.gitbooks.io/python-cook-book/content/Import/import-from-parent-folder.html
import os
import sys
root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)
sys.path.append(f"{root_folder}/src/main")
sys.path.append(f"{root_folder}/src/CodeFromImplementations")
import src.main.EndLayer as EndLayer
# import src.main.Config as Config
import src.main.main as main
import torch
import numpy as np
import src.main.FileHandling as FileHandling
import src.main.Dataload as Dataload
import src.main.ModelStruct as ModelStruct
from torch.utils.data import DataLoader

main.Config.parameters["num_epochs"][0] = 1
main.Config.parameters["num_workers"][0] = 0
main.Config.parameters["MaxPerClass"][0] = 10
main.Config.parameters["LOOP"][0] = 0
main.Config.datasetRandomOffset = False

# def testrun():
#     """
#     Tests if everything runs properly in a single run.
#     """
#     main.Config.unit_test_mode = True
#     main.run_model(graphDefault=False)

def testrunall():
    """
    Tries all of the algorithms. Except Openmax
    """
    main.Config.unit_test_mode = True
    for x in ["Soft","Open","Energy","COOL","DOC","iiMod", "SoftThresh","Var"]:
        main.Config.parameters["OOD Type"][0] = x
        main.run_model(graphDefault=False)

# def testrunEnergy():
#     """
#     Tries Energy as it gets some of the more intresting outputs even if it does not train much.
#     """
#     main.Config.unit_test_mode = True
#     main.Config.parameters["OOD Type"][0] = "Energy"
#     main.run_model(graphDefault=False)

# def testrunDOC():
#     """
#     DOC is being problematic.
#     """
#     main.Config.unit_test_mode = True
#     main.Config.parameters["OOD Type"][0] = "DOC"
#     main.run_model(graphDefault=False)

# def testrunOpen():
#     """
#     Openmax is being problematic.
#     """
#     main.Config.unit_test_mode = True
#     main.Config.parameters["OOD Type"][0] = "Open"
#     main.run_model(graphDefault=False)

def testLoadDataset():
    main.torch.manual_seed(1)
    train1, test1, val1 = FileHandling.checkAttempLoad("")
    main.torch.manual_seed(1)
    train2, test2, val2 = FileHandling.checkAttempLoad("")
    for x,y in zip(train1,train2):
        assert torch.all(x[0] == y[0])
    
def testLoadDatasetfromSave():
    main.Config.parameters["attemptLoadModel"][0] = 1
    main.Config.parameters["attemptLoadData"][0] = 1
    main.torch.manual_seed(1)
    train1, test1, val1 = FileHandling.checkAttempLoad("")
    main.torch.manual_seed(1)
    train2, test2, val2 = FileHandling.checkAttempLoad("")
    for x,y in zip(train1,train2):
        assert torch.all(x[0] == y[0])

def testfindSave():
    """Finds if the save exists"""
    epochFound = -1
    x = 0
    while epochFound == -1 and x < len(main.Config.alg):
        main.Config.parameters["OOD Type"][0] = main.Config.alg[x]
        x += 1
        epochFound = ModelStruct.AttackTrainingClassification.findloadEpoch()
    assert epochFound != -1, "CANNOT FIND MODEL"
        
    pathFound = ModelStruct.AttackTrainingClassification.findloadPath(epochFound)
    assert os.path.exists(pathFound) #The path is invalid.

def testrunFromSave():
    """
    Tests if saves work and if they result in the same answer if given the same seed.
    """
    class historyCheck(FileHandling.Score_saver):
        def __init__(self):
            self.writer = None
            self.level = 0
            self.vals = {}
        
        def __call__(self,itemDescription,item, fileName=""):
            if fileName == "BatchSaves.csv":
                #Obviously the batches are going to be different.
                return
            if self.level == 0:
                self.addtoLoopNames(itemDescription,item)
            else:
                self.checkifinloop(itemDescription,item)
        def addtoLoopNames(self,itemDescription,item):
            assert isinstance(itemDescription,str)
            self.vals[itemDescription] = item
        def checkifinloop(self,itemDescription,item):
            #np.nan is strange: https://stackoverflow.com/a/52124109
            assert (item==(self.vals[itemDescription])) or (np.isnan(item) and np.isnan(self.vals[itemDescription]))
    
    main.Config.unit_test_mode = True
    main.Config.parameters["num_workers"][0] = 0
    main.Config.parameters["MaxPerClass"][0] = 10
    main.Config.parameters["LOOP"][0] = 0
    main.Config.parameters["num_epochs"][0] = 0
    measurement = historyCheck()
    main.torch.manual_seed(1)
    main.run_model(measurement,graphDefault=False)
    measurement.level = 1
    main.torch.manual_seed(1)
    main.run_model(measurement,graphDefault=False)
