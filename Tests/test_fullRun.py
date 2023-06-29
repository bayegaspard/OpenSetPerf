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

main.Config.parameters["num_epochs"][0] = 1
main.Config.parameters["MaxPerClass"][0] = 10

def testrun():
    main.Config.unit_test_mode = True
    main.run_model()

def testrunall():
    main.Config.unit_test_mode = True
    for x in main.Config.alg:
        main.Config.parameters["OOD Type"][0] = x
        main.run_model()


def testrunFromSave():
    main.Config.unit_test_mode = True
    main.Config.parameters["num_epochs"][0] = 0
    main.run_model()