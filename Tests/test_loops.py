#three lines from https://xxx-cook-book.gitbooks.io/python-cook-book/content/Import/import-from-parent-folder.html
import os
import sys
root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)
sys.path.append(f"{root_folder}/src/main")
sys.path.append(f"{root_folder}/src/CodeFromImplementations")
# import src.main.Config as Config
import src.main.main as main

main.Config.parameters["MaxPerClass"][0] = 10



def testLoop1Uniqueness():
    """
    Tests if Loop 1 has all unique values
    """
    global loopNames
    global count
    count = 0
    loopNames = set()
    def nothing():
        pass
    def addtoLoopNames(itemDescription,item):
        global count
        count= (count+1)%3
        if count ==1:
            global loopNames
            assert not item in loopNames
            loopNames.add(item)
        assert isinstance(itemDescription,str)
    
    main.Config.parameters["LOOP"][0] = 1
    main.loopType1(nothing,addtoLoopNames)
    main.FileHandling.addMeasurement("For Unit Test","True",fileName="LoopRan.csv")

def testLoop2Uniqueness():
    """
    Tests if loop 2 has unique vlaues. Note that Loop 2 is not used anymore
    """
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
        assert isinstance(itemDescription,str)
    
    main.Config.parameters["LOOP"][0] = 2
    main.loopType2(nothing,addtoLoopNames)

def testLoop3looping():
    """
    Tests if loop 3 works properly
    """
    def nothing():
        pass
    def nothing2(item1,item2):
        pass
    main.Config.parameters["loopLevel"] = [0,"testing"]
    main.Config.parameters["LOOP"][0] = 3
    main.loopType3(nothing,nothing2)

def testLoop4looping():
    """
    Tests if loop 3 works properly
    """
    def nothing():
        pass
    def nothing2(item1,item2):
        pass
    main.Config.parameters["loopLevel"] = [0,"testing"]
    main.Config.parameters["LOOP"][0] = 4
    main.loopType4(nothing,nothing2)

def test_loop1HasValues():
    assert main.Config.parameters["OOD Type"][0] in main.Config.alg

    global section
    section = None
    def nothing():
        pass
    def addtoLoopNames(itemDescription,item):
        global section
        if itemDescription == "Type of modification":
            section = item
        elif itemDescription == "Modification Level":
            if not section is None:
                global loopNames
                if not section in ["optimizer","Unknowns"]:
                    assert item == "Default" or item in str(main.Config.parameters[section][0])
                section = None
    
    main.Config.parameters["LOOP"][0] = 1
    main.loopType1(nothing,addtoLoopNames)