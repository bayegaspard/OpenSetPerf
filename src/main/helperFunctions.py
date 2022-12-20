#These are all the functions that dont fit elsewhere
import Config as Config


#Translation dictionaries for algorithms that cannot have gaps in their numbers.
relabel = {15:15}
rerelabel = {15:15}
temp = 0
for x in range(15):
    if temp < len(Config.helper_variables["unknowns_clss"]["unknowns"]) and x == Config.helper_variables["unknowns_clss"]["unknowns"][temp]:
        temp = temp+1
    else:
        relabel[x] = x-temp
        rerelabel[x-temp] = x
temp = None

#This is to test all of the algorithms one after the other
startedOn = Config.parameters["OOD Type"][0]
startedEpochs = Config.parameters["num_epochs"][0]
def testRotate():
    current = Config.parameters["OOD Type"][0]
    if current == "Soft":
        Config.parameters["OOD Type"][0] = "Open"
        Config.parameters["num_epochs"][0] = 0
    elif current == "Open":
        Config.parameters["OOD Type"][0] = "Energy"
        Config.parameters["num_epochs"][0] = 0
    elif current == "Energy":
        Config.parameters["OOD Type"][0] = "COOL"
        Config.parameters["num_epochs"][0] = startedEpochs
    elif current == "COOL":
        Config.parameters["OOD Type"][0] = "DOC"
        Config.parameters["num_epochs"][0] = startedEpochs
    elif current == "DOC":
        Config.parameters["OOD Type"][0] = "Soft"
        Config.parameters["num_epochs"][0] = startedEpochs
    if Config.parameters["OOD Type"][0] == startedOn:
        return False
    return True
