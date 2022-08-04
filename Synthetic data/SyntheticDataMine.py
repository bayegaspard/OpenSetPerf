if __name__ =="__main__":
    import torch
    import glob
    import pandas as pd
    #Based off of SMOTE 
    #https://www.youtube.com/watch?v=JvFrJacbt6U <-video I watched about SMOTE
    #From my understanding SMOTE picks a random point between two valid points in the class

    #four lines from https://xxx-cook-book.gitbooks.io/python-cook-book/content/Import/import-from-parent-folder.html
    import os
    import sys
    root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.append(root_folder)

    #this seems really messy
    import HelperFunctions.LoadPackets as LoadPackets



    path_to_dataset = "datasets" #put the absolute path to your dataset , type "pwd" within your dataset folder from your teminal to know this path.

    def getListOfCSV(path):
        return glob.glob(path+"/*.csv")


    

    for x in range(14):
        createdlist=pd.DataFrame()
        ignore = list(range(15))
        ignore.remove(x+1)

        data_original = LoadPackets.NetworkDataset(getListOfCSV(path_to_dataset), ignore=ignore)
        data_original.isOneHot = False
        count = len(data_original)


        CLASSES = 1

        originalSet =  torch.utils.data.DataLoader(dataset=data_original, batch_size=100000, num_workers=2, persistent_workers=True)

        while count<2000000:
            for X,y in originalSet:

                rand = torch.rand(1)
                X2 = torch.cat([X[len(X)-1].unsqueeze(dim=0),X[:len(X)-1]], dim=0)

                new_data = X*rand + X2*(1-rand)
                new_data = torch.cat([new_data,y.unsqueeze(dim=1)],dim=1)
                createdlist = pd.concat([createdlist,pd.DataFrame(new_data.numpy())])
                count+=len(new_data)
                
        createdlist.to_csv("datasets/SMOTE.csv", mode='a', index=False, header=False)

    print("Done")