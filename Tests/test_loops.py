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

main.Config.parameters["MaxPerClass"][0] = 10



def testLoop1Uniqueness():
    global loopNames
    global count
    count = 0
    loopNames = set()
    def nothing():
        pass
    def addtoLoopNames(itemDescription,item):
        global count
        count= (count+1)%2
        if count ==1:
            global loopNames
            assert not item in loopNames
            loopNames.add(item)
    
    main.Config.parameters["LOOP"][0] = 1
    main.loopType1(nothing,addtoLoopNames)

def testLoop2Uniqueness():
    global loopNames
    global count
    count = 0
    loopNames = set()
    def nothing():
        pass
    def addtoLoopNames(itemDescription,item):
        global count
        count= (count+1)%2
        if count ==1:
            global loopNames
            assert not item in loopNames
            loopNames.add(item)
    
    main.Config.parameters["LOOP"][0] = 1
    main.loopType2(nothing,addtoLoopNames)