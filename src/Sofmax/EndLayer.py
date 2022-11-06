import torch
import numpy as np
import torch.nn.functional as F
import pandas as pd

#three lines from https://xxx-cook-book.gitbooks.io/python-cook-book/content/Import/import-from-parent-folder.html
import os
import sys
root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)


class EndLayers():

    def __init__(self,num_classes: int,cutoff=0.25, type="Soft"):
        self.cutoff = cutoff
        self.classCount = num_classes
        self.type = type
        self.DOO = 3    #Degree of Overcompleteness for COOL
        self.args = None
        self.Save_score = []        #this is really not great but I don't have time to find something better.



    def endlayer(self, output_true:torch.Tensor, y:torch.Tensor, type=None, offset=0):
        #check if a type is specified
        if type is None:
            type = self.type
        
        if type == "COOL":
            type = "Soft"

        #modify outputs
        if type != "Open":
            output_modified = self.typesOfMod[type](self,output_true)
        else:
            output_modified = self.typesOfMod[type](self,output_true,y)


        #reject from unknown, foundunknowns is a mask to be applied to the output vector
        output_complete = self.typesOfUnknown[type](self,output_modified)


        return output_complete




    #---------------------------------------------------------------------------------------------
    #This is the section for adding unknown column

    def softMaxUnknown(self,percentages:torch.Tensor):
        self.Save_score.append(percentages.max(dim=1)[0].mean())
        #this adds a row that is of the cutoff amout so unless there is another value greater it will be unknown
        batchsize = len(percentages)
        unknownColumn = self.cutoff * torch.ones(batchsize,device=percentages.device)
        return torch.cat((percentages,unknownColumn.unsqueeze(1)),dim=1)

    def openMaxUnknown(self,percentages:torch.Tensor):
        #Openmax already has a column for how much it thinks something is unknown
        return percentages
    
    def energyUnknown(self, percentages:torch.Tensor):
        if self.args is None:
            self.setArgs()
        import CodeFromImplementations.EnergyCodeByWetliu as Eng



        #The energy score code likes to output as a list
        scores = []
        Eng.energyScoreCalc(scores,percentages,self.args)
        scores = torch.tensor(np.array(scores),device=percentages.device)
        #after converting it to a tensor, the wrong dimention is expanded
        scores = -scores.squeeze().unsqueeze(dim=1)
        print(scores.sum()/len(scores))
        #Just store this for later
        self.Save_score.append(scores.mean())
        #once the dimentions are how we want them we test if it is above the cutoff
        scores = scores.greater_equal(self.cutoff).to(torch.int)
        #Then we run precentages through a softmax to get a nice score
        percentages = torch.softmax(percentages,dim=1)
        #Finally we join the results as an unknown class in the output vector
        return torch.cat((percentages,scores),dim=1)
        
    def odinUnknown(self, percentages:torch.Tensor):
        print("ODIN is not working at the moment")
        return percentages.max(dim=1,keepdim=True)[0].greater_equal(self.cutoff)

    #all functions here return a mask with 1 in all valid locations and 0 in all invalid locations
    typesOfUnknown = {"Soft":softMaxUnknown, "Open":openMaxUnknown, "Energy":energyUnknown, "Odin":odinUnknown}

    #---------------------------------------------------------------------------------------------
    #This is the section for modifying the outputs for the final layer

    def softMaxMod(self,percentages:torch.Tensor):
        return torch.softmax(percentages, dim=1)

    

    def setArgs(self, classes=None, weibullThreshold=0.9, weibullTail=20, weibullAlpha=3, score="energy", m_in=-1, m_out=0, temp=None):
        param = pd.read_csv("hyperParam.csv")
        unknowns = pd.read_csv("unknowns.csv")
        unknowns = unknowns["unknowns"].to_list()
        if temp is None:
            temp = float(param["Temperature"][0])
        if classes is None:
            classes = int(param["CLASSES"][0])-len(unknowns)

        class argsc():
            def __init__(self):
                #OpenMax
                self.train_class_num = classes
                self.weibull_threshold = weibullThreshold
                self.weibull_tail = weibullTail
                self.weibull_alpha = weibullAlpha
                #EnergyOOD
                self.score = score
                self.m_in=m_in
                self.m_out = m_out
                self.T = temp
        args = argsc()
        
        self.args = args
    
    def prepWeibull(self,trainloader,device,net):
        net.eval()
        self.weibulInfo = {"loader":trainloader,"device":device,"net":net}
        
        net.train()

    def openMaxMod(self,percentages:torch.Tensor, labels:torch.Tensor):

        if self.args == None:
            self.setArgs()

        try:
            import CodeFromImplementations.OpenMaxByMaXu as Open
        except ImportError:
            print("Warning: OpenMax has been skipped!")
            errorreturn = torch.zeros((percentages.size()))
            unknownColumn =torch.ones(len(percentages)).unsqueeze(1)
            errorreturn = torch.cat((errorreturn,unknownColumn),1)
            self.Save_score.append(torch.zeros(0))
            return errorreturn

        try:
            scores_open = Open.openmaxevaluation([percentages.detach()],[labels.detach()],self.args,self.weibulInfo)
        except LookupError:
            print("OpenMax failed! Skipping Openmax")
            #Note: usual reason for failure is having no correct examples for at least 1 class.
            errorreturn = torch.zeros((percentages.size()))
            unknownColumn =torch.ones(len(percentages)).unsqueeze(1)
            errorreturn = torch.cat((errorreturn,unknownColumn),1)
            self.Save_score.append(torch.zeros(0))
            return errorreturn
        print(scores_open)
        scores = torch.tensor(np.array(scores_open))
        self.Save_score.append(scores.squeeze().mean())
        scores.squeeze_().unsqueeze_(0)
        return torch.cat((percentages,scores),dim=0)
    
    def energyMod(self, percentages:torch.Tensor):
        return percentages
        
    def odinSetup(self, X, model, temp, noise):
        #most of this line is from the ODIN implementation (line 27)
        self.OdinX = torch.autograd.Variable(X, requires_grad = True)
        self.model = model
        self.temp = temp
        self.noise = noise

    def odinMod(self, percentages:torch.Tensor):
        print("ODIN is not working at the moment")
        import CodeFromImplementations.OdinCodeByWetliu as Odin
        self.model.openMax = False
        new_percentages = torch.tensor(Odin.ODIN(self.OdinX,self.model(self.OdinX), self.model, self.temp, self.noise))
        self.model.openMax = True
        return new_percentages[:len(percentages)]

    def COOL_Label_Mod(self, targets:torch.Tensor):
        new = []
        for target in targets:
            #DOO stands for degree of overcompletness and it is the number of nodes per class.
            l = torch.zeros(self.DOO,self.classCount)
            val = 1/self.DOO
            for i in range(self.DOO):
                l[(i),(target)] = val
            #l = torch.nn.functional.one_hot(targets, num_classes=self.classCount).unsqueeze(dim=1).repeat_interleave(3,dim=1)/self.DOO
            new.append(l)
        targets = torch.stack(new)
        return targets

    def COOL_predict_Mod(self, prediction:torch.Tensor):
        new = []
        for predict in prediction:
            l = torch.ones(self.classCount)
            for i in range(self.DOO):
                for j in range(self.classCount):
                    l[j] = l[j]*predict[(j)+(i)*self.classCount]*self.DOO
            new.append(l)
        targets = torch.stack(new)
        return targets

    def iiMod(self, percentages:torch.Tensor):
        #https://arxiv.org/pdf/1802.04365.pdf
        return percentages

    #all functions here return a tensor, sometimes it has an extra column for unknowns
    typesOfMod = {"Soft":softMaxMod, "Open":openMaxMod, "Energy":energyMod, "Odin":odinMod}


    
    #-------------------------------------------------------------------------------------
    #crates a cutoff with a certian percent labeled to be in distribution

    def cutoffStorage(self, newNumbers:torch.Tensor, type=None):
        if type is None:
            type = self.type
        if type == "Energy":
            import CodeFromImplementations.EnergyCodeByWetliu as Eng
            scores = []
            Eng.energyScoreCalc(scores,newNumbers)
            highestPercent = -torch.tensor(np.array(scores)).squeeze(dim=0)
        else:
            numbers = self.typesOfMod[type](self,newNumbers)
            highestPercent = torch.max(numbers,dim=1)[0]

            if type == "Open":
                highestPercent = highestPercent*(torch.max(numbers,dim=1)[1]!=self.classCount).int()
            

        self.highestPercentStorage = torch.cat((self.highestPercentStorage,highestPercent),dim=0)


    def findcutoff(self, numbers:torch.Tensor, percent):
        sortednums = numbers.sort(descending=True)[0]
        val = sortednums[int(percent*len(sortednums))]
        return val

    def autocutoff(self, percent=0.95):
        self.cutoff = self.findcutoff(self.highestPercentStorage, percent)