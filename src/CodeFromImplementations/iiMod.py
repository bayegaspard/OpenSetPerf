#Our Implementation of "Learning a Neural-network-based Representation for Open Set Recognition" (https://arxiv.org/pdf/1802.04365.pdf)
#All the equations are labeled with the number that they used in the paper. 
import torch

#TODO
#The network
#is trained on a dataset containing known and unknown classes,



if __name__ =="__main__":
    #three lines from https://xxx-cook-book.gitbooks.io/python-cook-book/content/Import/import-from-parent-folder.html
    import os
    import sys
    root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.append(root_folder)
    sys.path.append(root_folder+"/main")

import ModelStruct
import Config as Config

def Algorithm_1(data_loader, model:ModelStruct.AttackTrainingClassification, train=False):
    #Note, this masking is only due to how we are handling model outputs.
    #If I was to design things again I would have designed the model outputs not to need this masking.
    mask = torch.zeros(Config.parameters["CLASSES"][0])
    for x in Config.parameters["Knowns_clss"][0]:
        mask[x] = 1
    mask = mask==1 #Apparently it only wants bools

    totalout = []
    totallabel = []
    for (X,Y) in data_loader:
        #Getting the correct column (Nessisary for our label design)
        y = Y[:,0]
        Z = model(X).cpu()    #Step 2
        Z = Z[:,mask] #selects only the rows that are not unknowns. (On second thought we probibly should have made the model be able to change how many classes it can output)
        means = class_means(Z,y) #Step 4
        intra = intra_spread(Z,means,y) #Step 5
        inter = inter_sparation(means) #Step 6
        ii_loss = intra-inter #Step 7
        if train:
            ii_loss.backward() #Step 8
        totalout.append(Z)
        totallabel.append(y)

    return class_means(torch.cat(totalout,dim=0),torch.cat(totallabel,dim=0)) #Step 9 and 10

#This is algorithm 1 but one batch at a time       
def singleBatch(batch, model:ModelStruct.AttackTrainingClassification, train=True):

    X,Y = batch
    #Getting the correct column (Nessisary for our label design)
    Y = Y[:,0]
    Z = model(X).cpu()    #Step 2
    means = class_means(Z,Y) #Step 4
    intra = intra_spread(Z,means,Y) #Step 5
    inter = inter_sparation(means) #Step 6
    ii_loss = intra-inter #Step 7
    if train:
        ii_loss.backward() #Step 8
    return class_means(Z,Y) #Step 9 and 10

#Equation 2
def class_means(Z:torch.Tensor,Y:torch.Tensor):
    means = []
    for y in Config.parameters["Knowns_clss"][0]:
    # for y in [0,1,2]:
        #Technically only this part is actually equation 2 but it seems to want to output a value for each class.
        mask = (Y==y).cpu()
        Cj = mask.sum().item()
        sumofz = Z[mask].sum(dim=0)
        means.append(sumofz/Cj)
    return means

#Equation 1
def intra_spread(Z:torch.Tensor,means:list,Y:torch.Tensor)->torch.Tensor:
    intraspread = torch.tensor(0,dtype=torch.float32)
    N = len(Y)
    K = range(len(Config.parameters["Knowns_clss"][0]))
    # K = range(len([0,1,2]))
    #For each class in the knowns
    for j in K:
        #The mask will only select items of the correct class
        mask = (Y==Config.parameters["Knowns_clss"][0][j]).cpu()
        # mask = Y==[0,1,2][j]
        distanceVector = means[j]-Z[mask]
        
        intraspread += (torch.linalg.norm(distanceVector,dim=0)**2).sum()
    
    return intraspread/N

#Equation 3
def inter_sparation(means)->torch.Tensor:
    K = range(len(Config.parameters["Knowns_clss"][0]))
    # K = range(len([0,1,2]))
    minimum = torch.linalg.norm(means[0]-means[1])
    #I know I shouldn't use loops in python but I do not understand what this is doing enough to condense it.
    for m in range(len(K)):
        for n in range(m+1,len(K)):
            distance=means[m]-means[n]
            normalDistance = torch.linalg.norm(distance)
            minimum = minimum if minimum<normalDistance else normalDistance
    return minimum

#Equation 5
def outlier_score(x:torch.Tensor,means:torch.Tensor)->torch.Tensor:
    mask = torch.zeros(Config.parameters["CLASSES"][0])
    for i in Config.parameters["Knowns_clss"][0]:
        mask[i] = 1
    mask = (mask==1).cpu() #Apparently it only wants bools

    meansTensor = torch.stack(means)
    distances = torch.add(meansTensor,x.cpu()[mask],alpha=-1)
    normalDistance = torch.linalg.norm(distances,dim=1)
    return normalDistance.min()

#Equation 6
def iimod(Z, means):
    mask = torch.zeros(len(Z[0]))
    for x in Config.parameters["Knowns_clss"][0]:
        mask[x] = 1
    mask = mask==1 #Apparently it only wants bools
    Z = Z.cpu()*mask
    Z2= Z[:,mask] #selects only the rows that are not unknowns. (On second thought we probibly should have made the model be able to change how many classes it can output)
    for z in range(len(Z2)):
        meansTensor = torch.stack(means)
        distances = torch.add(meansTensor,Z2[z],alpha=-1)
        normalDistance = torch.linalg.norm(distances,dim=1)
        Z2[z] = normalDistance
    return Z




#testing
if __name__ == "__main__":
    Config.parameters["Knowns_clss"][0] = [0,1,2]
    Config.parameters["CLASSES"][0] = 4
    test = torch.rand((4,4))
    testLabels = torch.tensor((0,1,2,2))
    print(f"Test tensor\n{test}")
    means = class_means(test[:,:3],testLabels)
    print(f"Means\n{means}")
    intra = intra_spread(test[:,:3],means,testLabels)
    print(f"intra spread: {intra}")
    inter = inter_sparation(means)
    print(f"inter sparation: {inter}")
    total = intra-inter
    print(f"ii-loss: {total}")
    out = outlier_score(test[0],means)
    print(f"Test of class, should be zero: {out}")
    out = outlier_score(torch.rand(4),means)
    print(f"Test of random, should be non-zero: {out}")
    Z = iimod(test,means)
    print(f"Test tensor\n{test}")
    print(f"Modified tensor\n{Z}")