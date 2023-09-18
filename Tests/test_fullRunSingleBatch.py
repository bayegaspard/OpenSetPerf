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

def testSingleBatch():
    """
    Batches with only single items in them were having problems this should test for them.
    """
    main.Config.unit_test_mode = True
    main.Config.parameters["batch_size"][0] = 1
    for x in main.Config.alg:
        main.Config.parameters["OOD Type"][0] = x
        main.run_model(graphDefault=False)

# def testSingleBatch_Energy():
#     main.Config.unit_test_mode = True
#     main.Config.parameters["batch_size"][0] = 1
#     main.Config.parameters["OOD Type"][0] = "Energy"
#     main.run_model(graphDefault=False)