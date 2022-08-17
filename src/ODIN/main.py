import time
start_time = time.time()

if __name__ == "__main__":
    #---------------------------------------------Imports------------------------------------------
    import numpy as np
    import torch
    import torch.utils.data
    import torch.nn as nn
    import torch.optim as optim
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
    AUTOCUTOFF = True
    noise = 0.15
    temperature = 5
    epochs = 1
    checkpoint = "/checkpointFinal.pth"
    #------------------------------------------------------------------------------------------------------

    #---------------------------------------------Model/data set up----------------------------------------

    NAME = "src/"+os.path.basename(os.path.dirname(__file__))

    #I looked up how to make a dataset, more information in the LoadImages file

    path_to_dataset = "datasets" #put the absolute path to your dataset , type "pwd" within your dataset folder from your teminal to know this path.

    def getListOfCSV(path):
        return glob.glob(path+"/*.csv")

    data_total = NetworkDataset(getListOfCSV(path_to_dataset),ignore=[1,2,3,11,12,13,14])
    unknown_data = NetworkDataset(getListOfCSV(path_to_dataset),ignore=[0,3,4,5,6,7,8,9,10])

    CLASSES = len(data_total.classes)

    data_train, data_test = torch.utils.data.random_split(data_total, [len(data_total)-10000,10000])

    training =  torch.utils.data.DataLoader(dataset=data_train, batch_size=BATCH, shuffle=True, num_workers=1)
    testing = torch.utils.data.DataLoader(dataset=data_test, batch_size=BATCH, shuffle=False)
    unknowns = torch.utils.data.DataLoader(dataset=unknown_data, batch_size=BATCH, shuffle=False)


    model = Network(CLASSES).to(device)
    soft = correctValCounter(CLASSES, cutoff= CUTOFF, confusionMat=True)
    odin = correctValCounter(CLASSES, cutoff= CUTOFF, confusionMat=True)

    if os.path.exists(NAME+checkpoint):
        model.load_state_dict(torch.load(NAME+checkpoint))


    criterion = nn.CrossEntropyLoss(weight=torch.tensor([ 0.1467,  2.6045, 42.0071, 56.5461, 57.5315, 60.6388,  1.4431, 32.3960])[:CLASSES]).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.5, momentum=0.5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

    #for timing
    epoch_avrg = 0

    #------------------------------------------------------------------------------------------------------

    #---------------------------------------------Training-------------------------------------------------

    for e in range(epochs):
        epoch_start = time.time()
        lost_amount = 0
        loss_avrg = 0

        for batch, (X, y) in enumerate(training):

            X = (X).to(device)
            y = y.to(device)

            _, output = model(X)
            #output = torch.nn.functional.softmax(output,dim=1)
            lost_points = criterion(output, y)
            optimizer.zero_grad()
            lost_points.backward()

            loss_avrg = (loss_avrg*batch + lost_points)/(batch+1)
            if batch%10 == 0:
                print(f"loss avrage: {loss_avrg}")

            optimizer.step()
            optimizer.zero_grad()

            lost_amount += lost_points.item()

        epoch_time = time.time() - epoch_start
        epoch_avrg = (epoch_avrg*e + time.time())/(e+1)
        print(f"Epoch took: {epoch_time} seconds")

        #--------------------------------------------------------------------------------

        #--------------------------------------Autocutoff--------------------------------
        model.eval()

        #make a call about where the cutoff is
        if AUTOCUTOFF:
            for batch, (X, y) in enumerate(training):

                #odin:
                odin.odinSetup(X,model,temperature,noise)


                _, output = model(X)

                soft.cutoffStorage(output.detach(), "Soft")
                odin.cutoffStorage(output.detach(), "Odin")
            soft.autocutoff(0.73)
            odin.autocutoff(0.67)
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
        soft.storeConfusion("CONFUSIONSOFT.CSV")
        odin.storeConfusion("CONFUSIONODIN.CSV")
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

        soft.evalN(output,y, indistribution=False, offset=-CLASSES)
        odin.evalN(output,y, indistribution=False, type="Odin", offset=-CLASSES)

    soft.PrintUnknownEval()
    odin.PrintUnknownEval()
    soft.zero()
    odin.zero()

    model.train()

    #------------------------------------------------------------------------------------------------------

    print(f"Program took: {time.time() - start_time}")