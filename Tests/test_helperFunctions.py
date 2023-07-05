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
import src.main.FileHandling as FileHandling
import src.main.Dataload as Dataload
from torch.utils.data import DataLoader
import src.main.helperFunctions as helperFunctions
from src.main.Config import parameters as parameters
from src.main.Config import class_split as class_split 

def testRemovedVals():
    tensor = torch.tensor(list(range(parameters["CLASSES"][0])))
    newTensor = tensor.clone()
    newTensor = helperFunctions.renameClasses(newTensor)
    for x in class_split["unknowns_clss"]:
        assert not x in newTensor

    tensor = newTensor.clone()
    for x in range(len(newTensor)):
        newTensor[x] = torch.tensor(helperFunctions.relabel[newTensor[x].item()])
    assert newTensor.max() <= parameters["CLASSES"][0]-len(class_split["unknowns_clss"])
    for x in range(len(newTensor)):
        newTensor[x] = torch.tensor(helperFunctions.rerelabel[newTensor[x].item()])

    assert torch.all(newTensor==tensor)