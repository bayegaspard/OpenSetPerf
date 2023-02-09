#Our Implementation of https://arxiv.org/pdf/1802.04365.pdf
import torch
import src.main.ModelStruct as ms
import src.main.Config as Config

def Algorithm_1(data_loader, model:ms.AttackTrainingClassification):
    for (X,Y) in data_loader:
        #Getting the correct column
        Y = Y[:,0]
        Z = model(X)

def class_means(Z:torch.Tensor,Y:torch.Tensor):
    means = []
    for y in Config.helper_variables["knowns_clss"]:
        mask = Y==y
        means.append(Z[mask].sum(dim=1)/mask.sum().item())
    return means

def intra_spread(Z:torch.Tensor,means:list,Y:torch.Tensor):
    intraspread = 0
    for j in range(len(Config.helper_variables["knowns_clss"])):
        mask = Y==Config.helper_variables["knowns_clss"][j]
        
        #The mask will only select items of the correct class
        intraspread += torch.linalg.norm(means[j]-Z[mask],dim=-1).sum().item()