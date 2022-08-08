#implementation from figure 13 of this paper: https://arxiv.org/pdf/2108.00071.pdf
#imbalenced learn https://pypi.org/project/imbalanced-learn/
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=54, k_neighbors = 5)

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


    # data_most = LoadPackets.NetworkDataset(getListOfCSV(path_to_dataset), ignore=[1,2,3,4,5,6,7,8,9,10,11,12,13,14])
    # most =  torch.utils.data.DataLoader(dataset=data_most, batch_size=2000000, num_workers=2, persistent_workers=True)

    createdlist=pd.DataFrame()
    ignore = list(range(15))
    ignore = None

    data_original = LoadPackets.NetworkDataset(getListOfCSV(path_to_dataset), ignore=ignore)
    

    originalSet =  torch.utils.data.DataLoader(dataset=data_original, batch_size=1000000, num_workers=2, persistent_workers=True)
    for a,b in originalSet:
        a = [x.numpy() for x in a]
        b = [torch.argmax(x).numpy() for x in b]
        new_data, new_targets = sm.fit_resample(pd.DataFrame(a),pd.DataFrame(b))

        new_data = pd.concat([new_data,new_targets],axis=1)

        createdlist = pd.concat([createdlist,new_data])

    createdlist.to_csv(f"datasets/SMOTE.csv", header=False)

    print("Done")