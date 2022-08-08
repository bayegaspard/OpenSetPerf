import torch
import numpy as np
import torch.nn.functional as F

#four lines from https://xxx-cook-book.gitbooks.io/python-cook-book/content/Import/import-from-parent-folder.html
import sys
import os
root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)

import HelperFunctions.Evaluation as Evaluation

class EvaluationWithPlots(Evaluation.correctValCounter):

    def __init__(self,num_classes: int, type="Soft", cutoff=0.85, confusionMat=False, temp = 1, noise=0):
        super().__init__(num_classes, type,cutoff,confusionMat,temp,noise)

    
    def zero(self):
        super().zero()
        #plotting stuff
        self.plotting = torch.zeros((2,35))
        self.plotting[1] += torch.tensor([x for x in range(len(self.plotting[1]))])
        self.highestPercentStorage = torch.zeros(1)

#Plotting
    def cutoffPlotVals(self, scores):
        plotting = []
        temporary = torch.tensor(np.array(scores))
        for x in range(35):
            plotting.append(temporary.less_equal(x).sum())
        self.plotting[0] += torch.tensor(plotting)