from pandas import cut
import torch
import numpy as np
import torch.nn.functional as F
import Evaluation


class EvaluationWithPlots(Evaluation.correctValCounter):

    def __init__(self,num_classes: int, cutoff=0.85, confusionMat=False, temp = 1, noise=0):
        super().__init__(num_classes,cutoff,confusionMat,temp,noise)

    
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