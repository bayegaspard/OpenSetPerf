if __name__ == "__main__":
    #---------------------------------------------Imports------------------------------------------
    import numpy as np
    import torch
    import torch.utils.data
    import torch.nn as nn
    import torch.optim as optim
    import matplotlib.pyplot as plt
    import os
    import glob

    #three lines from https://xxx-cook-book.gitbooks.io/python-cook-book/content/Import/import-from-parent-folder.html
    import sys
    root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.append(root_folder)

    #this seems really messy
    from HelperFunctions.LoadPackets import NetworkDataset
    from HelperFunctions.Evaluation import correctValCounter
    from HelperFunctions.ModelLoader import Network

    #pick a device
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")

    #------------------------------------------------------------------------------------------------------

    #---------------------------------------------Hyperparameters------------------------------------------
    torch.manual_seed(0)    #beware contamination
    BATCH = 5000
    CUTOFF = 0.85
    noise = 0.15
    temperature = 0.001
    epochs = 5
    checkpoint = "/checkpoint.pth"
    #------------------------------------------------------------------------------------------------------

    #---------------------------------------------Model/data set up----------------------------------------

    NAME = "src/"+os.path.basename(os.path.dirname(__file__))

    #I looked up how to make a dataset, more information in the LoadImages file

    path_to_dataset = "datasets" #put the absolute path to your dataset , type "pwd" within your dataset folder from your teminal to know this path.

    def getListOfCSV(path):
        return glob.glob(path+"/*.csv")

    data_total = NetworkDataset(getListOfCSV(path_to_dataset))
    unknown_data = NetworkDataset(getListOfCSV(path_to_dataset),benign=False)

    CLASSES = len(data_total.classes)

    data_train, data_test = torch.utils.data.random_split(data_total, [len(data_total)-1000,1000])

    training =  torch.utils.data.DataLoader(dataset=data_train, batch_size=BATCH, shuffle=True, num_workers=1)
    testing = torch.utils.data.DataLoader(dataset=data_test, batch_size=BATCH, shuffle=False)
    unknowns = torch.utils.data.DataLoader(dataset=unknown_data, batch_size=BATCH, shuffle=False)


    model = Network(CLASSES).to(device)
    soft = correctValCounter(CLASSES, cutoff= CUTOFF)
    odin = correctValCounter(CLASSES, cutoff= CUTOFF)

    if os.path.exists(NAME+checkpoint):
        model.load_state_dict(torch.load(NAME+checkpoint))


    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0000e+00, 2.8636e+02, 3.8547e+02, 3.9218e+02, 4.1337e+02, 9.8373e+00,
            2.2084e+02, 2.0665e+05, 1.5084e+03, 3.4863e+03, 1.0824e+05, 6.3142e+04,
            1.1562e+03, 1.4303e+01, 1.7754e+01])[:CLASSES]).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    #------------------------------------------------------------------------------------------------------

    #---------------------------------------------Training-------------------------------------------------

    for e in range(epochs):
        lost_amount = 0

        for batch, (X, y) in enumerate(training):

            X = (X).to(device)
            y = y.to(device)

            _, output = model(X)
            lost_points = criterion(output, y)
            optimizer.zero_grad()
            lost_points.backward()


            optimizer.step()
            optimizer.zero_grad()

            lost_amount += lost_points.item()

        
        #--------------------------------------------------------------------------------

        #--------------------------------------Testing-----------------------------------

        model.eval()
        for batch,(X,y) in enumerate(testing):
            X = X.to(device)
            y = y.to("cpu")


            _, output = model(X)
            output = output.to("cpu")

            odin.odinSetup(X,model,temperature,noise)


            soft.evalN(output,y)
            odin.evalN(output,y, type="Odin")
            
        
        
        print(f"-----------------------------Epoc: {e+1}-----------------------------")
        print(f"lost: {100*lost_amount/len(data_train)}")
        soft.PrintEval()
        odin.PrintEval()
        odin.zero()
        soft.zero()
        
        if e%5 == 4:
            torch.save(model.state_dict(), NAME+checkpoint)

        model.train()
        scheduler.step()


    #------------------------------------------------------------------------------------------------------

    #---------------------------------------------Unknowns-------------------------------------------------

    model.eval()
    for batch,(X,y) in enumerate(unknowns):
        X = (X).to(device)
        y = y.to("cpu")


        _, output = model(X)
        output = output.to("cpu")

        odin.odinSetup(X,model,temperature,noise)

        soft.evalN(output,y, indistribution=False)
        odin.evalN(output,y, indistribution=False, type="Odin")

    soft.PrintUnknownEval()
    odin.PrintUnknownEval()
    soft.zero()
    odin.zero()

    model.train()

    #------------------------------------------------------------------------------------------------------