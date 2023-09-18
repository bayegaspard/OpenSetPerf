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
import src.main.GenerateImages as GenerateImages
import pytest


def test_images():
    if not os.path.exists("Saves/Scoresall.csv"):
        #https://docs.pytest.org/en/7.1.x/how-to/skipping.html
        pytest.skip("No score file to test graphing with.")
    GenerateImages.main(save=False)

def test_validation_dataset():
    train, test, val = FileHandling.checkAttempLoad("")
    val = DataLoader(val, parameters["batch_size"][0], shuffle=True, num_workers=0, pin_memory=False)
    for batch in val:
        for x in parameters["Unknowns_clss"][0]:
            assert x not in batch[1][1]
        for x in batch[1]:
            assert x[1] in parameters["Knowns_clss"][0]

def test_testing_dataset():
    FileHandling.Config.parameters["Mix unknowns and validation"][0] = 0
    assert FileHandling.Config.parameters["Mix unknowns and validation"][0] != 3 #LOOP == 3 causes problems by not having unknowns in loop level 0
    FileHandling.Config.unit_test_mode = True
    train, test, val = FileHandling.checkAttempLoad("")
    test = DataLoader(test, parameters["batch_size"][0], shuffle=True, num_workers=0, pin_memory=False)
    for batch in test:
        for x in parameters["Knowns_clss"][0]:
            assert x not in batch[1][1]
        for x in batch[1]:
            assert x[1] in parameters["Unknowns_clss"][0]