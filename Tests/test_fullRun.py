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
main.Config.parameters["LOOP"][0] = 0

def testrun():
    """
    Tests if everything runs properly in a single run.
    """
    main.Config.unit_test_mode = True
    main.run_model()

def testrunall():
    """
    Tries all of the algorithms.
    """
    main.Config.unit_test_mode = True
    for x in main.Config.alg:
        main.Config.parameters["OOD Type"][0] = x
        main.run_model()


def testrunFromSave():
    """
    Tests if saves work and if they result in the same answer if given the same seed.
    """
    global vals
    vals = {}
    def addtoLoopNames(itemDescription,item):
        global vals
        assert isinstance(itemDescription,str)
        vals[itemDescription] = item
    def checkifinloop(itemDescription,item):
        global vals
        assert item<(vals[itemDescription]*1.01) and item>(vals[itemDescription]*0.99)
    main.Config.unit_test_mode = True
    main.Config.parameters["num_epochs"][0] = 0
    main.torch.manual_seed(1)
    main.run_model(addtoLoopNames)
    main.torch.manual_seed(1)
    main.run_model(checkifinloop)
