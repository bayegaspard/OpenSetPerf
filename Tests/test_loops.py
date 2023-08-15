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
main.Config.unit_test_mode = True


def testLoop1UniquenessAndChanges():
    """
    Tests if Loop 1 is working.
    Specifically that it is not repeating itself and that Config globals are being changed.
    """
    assert main.Config.parameters["OOD Type"][0] in main.Config.alg
    class looptype1Uniqueness(main.FileHandling.Score_saver):
        def __init__(self):
            self.loopNames = set()
            self.count = 0
            self.writer = None
            self.section = None


        def __call__(self,itemDescription,item,fileName=None):
            self.count= (self.count+1)%3
            if self.count ==1:
                self.loopNames
                assert not item in self.loopNames
                self.loopNames.add(item)
            assert isinstance(itemDescription,str)
            self.checkChange(itemDescription,item)
        
        def checkChange(self,itemDescription,item):
            """
            Checks if the values in Config actually change. 
            This used to be a separate test that caused issues with both tests writing to the same variables.
            """
            if itemDescription == "Type of modification":
                self.section = item
            elif itemDescription == "Modification Level":
                if not self.section is None:
                    if not self.section in ["optimizer","Unknowns"]:
                        assert item == "Default" or item in str(main.Config.parameters[self.section][0])
                    self.section = None


        def start(self):
            pass

    def nothing():
        pass
        
    main.Config.parameters["LOOP"][0] = 1
    measurement = looptype1Uniqueness()
    main.loopType1(nothing,measurement)
    measurement.addMeasurement("For Unit Test","True",fileName="LoopRan.csv")

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
    class looptype2Uniqueness(main.FileHandling.Score_saver):
        def __init__(self):
            self.loopNames = set()
            self.count = 0
            self.writer = None


        def __call__(self,itemDescription,item,fileName=None):
            self.count= (self.count+1)%2
            if self.count ==1:
                self.loopNames
                assert not item in self.loopNames
                self.loopNames.add(item)
            assert isinstance(itemDescription,str)
        
        def start(self):
            pass

    measurement = looptype2Uniqueness()
    main.Config.parameters["LOOP"][0] = 2
    main.loopType2(nothing,measurement)

def testLoop3looping():
    """
    Tests if loop 3 works properly
    """
    def nothing():
        pass
    class looptype3Uniqueness(main.FileHandling.Score_saver):
        def __init__(self):
            self.writer = None


        def __call__(self,itemDescription,item,fileName=None):
            pass
        
        def start(self):
            pass

    measurement = looptype3Uniqueness()
    main.Config.parameters["loopLevel"] = [0,"testing"]
    main.Config.parameters["LOOP"][0] = 3
    main.loopType3(nothing,measurement)

def testLoop4looping():
    """
    Tests if loop 4 works properly
    """

    def nothing():
        pass

    class looptype4Uniqueness(main.FileHandling.Score_saver):
        def __init__(self):
            self.writer = None
            self.rowNumber =0

        def __call__(self,item1,item2,fileName=None):
            assert self.rowNumber == item2
            self.rowNumber+=1
        
        def start(self):
            pass
    
    measurement = looptype4Uniqueness()
    main.Config.parameters["loopLevel"] = [0,"testing"]
    main.Config.parameters["LOOP"][0] = 4
    main.loopType4(nothing,measurement)
