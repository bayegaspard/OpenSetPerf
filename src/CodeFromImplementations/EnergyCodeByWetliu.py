#The code here is what was used in the github repo for the paper: https://arxiv.org/pdf/2010.03759.pdf
# Repo: https://github.com/wetliu/energy_ood
import torch
import torch.nn.functional as F
import numpy as np



#this code was from line 112 of energy_ood/CIFAR/test.py
to_np = lambda x: x.data.cpu().numpy()


def energyLossMod(loss,x,in_set,args):
    #This code was lines 192-196 of energy_ood/CIFAR/train.py and it is an addition to the training loss to account for energy.

    # cross-entropy from softmax distribution to uniform distribution
    if args.score == 'energy':
        Ec_out = -torch.logsumexp(x[len(in_set[0]):], dim=1)
        Ec_in = -torch.logsumexp(x[:len(in_set[0])], dim=1)
        loss += 0.1*(torch.pow(F.relu(Ec_in-args.m_in), 2).mean() + torch.pow(F.relu(args.m_out-Ec_out), 2).mean())


    return loss


def energyScoreCalc(_score, output,args):
    #This code was from lines 133-134 of energy_ood/CIFAR/test.py
    if args.score == 'energy':
                    _score.append(-to_np((args.T*torch.logsumexp(output / args.T, dim=1))))
    

    return _score
