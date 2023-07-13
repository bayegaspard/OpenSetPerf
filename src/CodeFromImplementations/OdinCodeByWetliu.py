#This code was found in https://github.com/wetliu/energy_ood Under Utils/score_calculation.py lines 13-84
#I did not write these functions.
#Line 26 and line 60 were changed because it is assuming that you have a GPU and I do not have a GPU so I removed ".cuda()"
#Commented out line 69 and 70 because they appear to not be nessisary with the model I am using.
#THIS IS NOW UNUSED. 
import torch
import numpy as np
import torch.nn.functional as F
#these two were lines 4 and 3 in the original.
import torch.nn as nn
from torch.autograd import Variable



to_np = lambda x: x.data.cpu().numpy()
concat = lambda x: np.concatenate(x, axis=0)

def get_ood_scores_odin(loader, net, bs, ood_num_examples, T, noise, in_dist=False):
    _score = []
    _right_score = []
    _wrong_score = []

    net.eval()
    for batch_idx, (data, target) in enumerate(loader):
        if batch_idx >= ood_num_examples // bs and in_dist is False:
            break
        data = data
        data = Variable(data, requires_grad = True)

        output = net(data)
        smax = to_np(F.softmax(output, dim=1))

        odin_score = ODIN(data, output,net, T, noise)
        _score.append(-np.max(odin_score, 1))

        if in_dist:
            preds = np.argmax(smax, axis=1)
            targets = target.numpy().squeeze()
            right_indices = preds == targets
            wrong_indices = np.invert(right_indices)

            _right_score.append(-np.max(smax[right_indices], axis=1))
            _wrong_score.append(-np.max(smax[wrong_indices], axis=1))

    if in_dist:
        return concat(_score).copy(), concat(_right_score).copy(), concat(_wrong_score).copy()
    else:
        return concat(_score)[:ood_num_examples].copy()


def ODIN(inputs, outputs, model, temper, noiseMagnitude1):
    # Calculating the perturbation we need to add, that is,
    # the sign of gradient of cross entropy loss w.r.t. input
    criterion = nn.CrossEntropyLoss()

    maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)

    # Using temperature scaling
    outputs = outputs / temper

    labels = Variable(torch.LongTensor(maxIndexTemp))
    loss = criterion(outputs, labels)
    loss.backward()

    # Normalizing the gradient to binary in {0, 1}
    gradient =  torch.ge(inputs.grad.data, 0)
    gradient = (gradient.float() - 0.5) * 2
    
    gradient[:,0] = (gradient[:,0] )/(63.0/255.0)
    #gradient[:,1] = (gradient[:,1] )/(62.1/255.0)
    #gradient[:,2] = (gradient[:,2] )/(66.7/255.0)
    #gradient.index_copy_(1, torch.LongTensor([0]).cuda(), gradient.index_select(1, torch.LongTensor([0]).cuda()) / (63.0/255.0))
    #gradient.index_copy_(1, torch.LongTensor([1]).cuda(), gradient.index_select(1, torch.LongTensor([1]).cuda()) / (62.1/255.0))
    #gradient.index_copy_(1, torch.LongTensor([2]).cuda(), gradient.index_select(1, torch.LongTensor([2]).cuda()) / (66.7/255.0))

    # Adding small perturbations to images
    tempInputs = torch.add(inputs.data,  -noiseMagnitude1, gradient)
    outputs = model(Variable(tempInputs))
    outputs = outputs / temper
    # Calculating the confidence after adding perturbations
    nnOutputs = outputs.data.cpu()
    nnOutputs = nnOutputs.numpy()
    nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
    nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)

    return nnOutputs