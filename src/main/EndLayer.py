import torch
import numpy as np
import torch.nn.functional as F
import pandas as pd
import Config
import helperFunctions
import time
from sklearn.metrics import confusion_matrix

#three lines from https://xxx-cook-book.gitbooks.io/python-cook-book/content/Import/import-from-parent-folder.html
import os
import sys
root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)


root_path = os.getcwd()

import helperFunctions

class EndLayers():

    def __init__(self,num_classes: int,cutoff=0.25, type="Soft"):
        self.cutoff = cutoff
        self.classCount = num_classes
        self.type = type
        self.DOO = Config.parameters["Degree of Overcompleteness"][0]    #Degree of Overcompleteness for COOL
        self.weibulInfo = None
        self.resetvals()



    def endlayer(self, output_true:torch.Tensor, y:torch.Tensor, type=None):
        startTime = time.time()
        self.rocData[0] = y==Config.parameters["CLASSES"][0]
        if 1==2:
            print(f"Argmax")
            helperFunctions.printconfmat(output_true.cpu(),y.cpu())
        #check if a type is specified
        if type is None:
            type = self.type
        
        #modify outputs if nessisary for algorithm
        output_modified = self.typesOfMod.get(type,self.typesOfMod["none"])(self,output_true)

        #This is supposted to add an extra column for unknowns
        output_complete = self.typesOfUnknown[type](self,output_modified)

        if False:
            print(f"Alg")
            if output_complete.ndim == 2:
                helperFunctions.printconfmat(output_complete.cpu(),y.cpu())
            else:
                print(f"{confusion_matrix(y.cpu(),output_complete.cpu())}")
        return output_complete

    #setup
    def setArgs(self, classes=None, weibullThreshold=0.9, weibullTail=20, weibullAlpha=3, score="energy", m_in=-1, m_out=0, temp=None):
        param = pd.read_csv(os.path.join("Saves","hyperparam","hyperParam.csv"))
        unknowns = pd.read_csv(os.path.join("Saves","unknown","unknowns.csv"))
        unknowns = unknowns["Unknowns"].to_list()
        if temp is None:
            temp = float(param["Temperature"][0])
        if classes is None:
            classes = len(Config.helper_variables["knowns_clss"])

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


            def selectKnowns(self, modelOut, labels:torch.Tensor):
                labels = labels.clone()
                lastval = -1
                label = list(range(Config.parameters["CLASSES"][0]))
                newout = []
                for val in Config.helper_variables["unknowns_clss"]:
                    label.remove(val)
                    if val > lastval+1:
                        newout.append(modelOut[:,lastval+1:val])
                    lastval = val

                newout = torch.cat(newout, dim=1)

                i = 0
                for l in label:
                    labels[labels==l] = i
                    i+=1
                return newout, labels
        args = argsc()
        
        self.args = args
    
    def prepWeibull(self,trainloader,device,net):
        net.eval()
        self.weibulInfo = {"loader":trainloader,"device":device,"net":net, "weibull":None}
        
        net.train()

    def noChange(self,X:torch.Tensor):
        return X

    #---------------------------------------------------------------------------------------------
    #This is the section for adding unknown column

    def softMaxUnknown(self,percentages:torch.Tensor):
        self.Save_score.append(percentages.max(dim=1)[0].mean())
        #this adds a row that is of the cutoff amout so unless there is another value greater it will be unknown
        batchsize = len(percentages)
        unknownColumn = torch.zeros(batchsize,device=percentages.device)
        self.Save_score.append(unknownColumn)
        self.rocData[1] = unknownColumn
        return torch.cat((percentages,unknownColumn.unsqueeze(1)),dim=1)

    def normalThesholdUnknown(self,percentages:torch.Tensor):
        self.rocData[1] = percentages.max(dim=1)[0]
        self.Save_score.append(self.rocData[1].mean())
        #this adds a row that is of the cutoff amout so unless there is another value greater it will be unknown
        batchsize = len(percentages)
        unknownColumn = self.cutoff * torch.ones(batchsize,device=percentages.device)
        return torch.cat((percentages,unknownColumn.unsqueeze(1)),dim=1)

    def openMaxUnknown(self,percentages:torch.Tensor):
        failed = False
        #from sklearn.metrics import confusion_matrix
        #print(f"{confusion_matrix(labels.cpu(),percentages.argmax(dim=1).cpu(),labels=Config.helper_variables['knowns_clss'])}")

        # relabeledPercentages = helperFunctions.renameClasses(percentages)

        try:
            import CodeFromImplementations.OpenMaxByMaXu as Open
        except ImportError:
            print("Openmax will be skipped as not all of its libraries could be loaded.")
            failed = True

        if self.args == None:
            self.setArgs()
        
        if not failed:
            try:
                if self.weibulInfo["weibull"] is None:
                    print("Getting Weibull")
                    self.weibulInfo["weibull"]=Open.weibull_fittting(self.args,self.weibulInfo)
                if (self.weibulInfo["weibull"]==False):
                    print("Openmax already failed")
                    failed = True
                else:
                    scores_open = Open.openmaxevaluation(percentages.detach(),self.args,self.weibulInfo,self.weibulInfo["weibull"])
            except NotImplementedError:
                print("Warning: OpenMax has failed to load!")
                failed = True
            #except LookupError:
            except helperFunctions.NoExamples:
                print("OpenMax failed to idenitify at least 1 class!")
                self.weibulInfo["weibull"]=False
                #Note: usual reason for failure is having no correct examples for at least 1 class.
                failed = True
                
        
        if failed:
            errorreturn = torch.zeros((percentages.size()))
            unknownColumn = torch.zeros(len(percentages)).unsqueeze(1)
            errorreturn = torch.cat((errorreturn,unknownColumn),1)
            self.rocData[1] = unknownColumn
            self.Save_score.append(torch.zeros(0))
            return errorreturn
            
        percentages = percentages*helperFunctions.mask.clone().to(device=percentages.device)
        percentages = torch.concat([percentages,torch.zeros([len(percentages),1],device=percentages.device)],dim=1)
        
        #print(scores_open)
        scores = torch.tensor(np.array(scores_open),device=percentages.device)
        percentages[:,torch.concat([helperFunctions.mask,torch.zeros(1)==0])] = scores.to(dtype=torch.float32)
        self.rocData[1] = percentages[:,helperFunctions.mask].max(dim=1)[0]
        self.Save_score.append(scores.squeeze())
        #scores.squeeze_().unsqueeze_(0)
        
        #return torch.cat((percentages,scores),dim=0)
        return percentages.to(device=percentages.device)
    
    def energyUnknown(self, percentages:torch.Tensor):
        if self.args is None:
            self.setArgs()
        import CodeFromImplementations.EnergyCodeByWetliu as Eng # useful link to import in relative directories https://stackoverflow.com/questions/4383571/importing-files-from-different-folder



        #The energy score code likes to output as a list
        scores = []
        Eng.energyScoreCalc(scores,percentages,self.args)
        scores = torch.tensor(np.array(scores),device=percentages.device)
        #after converting it to a tensor, the wrong dimention is expanded
        scores = -scores.squeeze().unsqueeze(dim=1)
        #This was to print the scores it was going to save
        #print(scores.sum()/len(scores))
        #Just store this for later
        self.rocData[1]=-scores
        self.Save_score.append(scores.mean())
        #once the dimentions are how we want them we test if it is above the cutoff
        scores = scores.less_equal(self.cutoff).to(torch.int)
        #Then we run precentages through a softmax to get a nice score
        percentages = torch.softmax(percentages,dim=1)
        #Finally we join the results as an unknown class in the output vector
        return torch.cat((percentages,scores),dim=1)
        
    def odinUnknown(self, percentages:torch.Tensor):
        print("ODIN is not working at the moment")
        return percentages.max(dim=1,keepdim=True)[0].greater_equal(self.cutoff)

    def DOCUnknown(self, percentages:torch.Tensor):
        import CodeFromImplementations.DeepOpenClassificationByLeishu02 as DOC
        if self.docMu is None:
            print("Mu Standards need to be collected")
            if self.weibulInfo is None:
                return
            else:
                self.docMu = DOC.muStandardsFromDataloader(Config.helper_variables["knowns_clss"],self.weibulInfo["loader"],self.weibulInfo["net"])
                self.Save_score = [torch.tensor(self.docMu)[:,1]]
        
        self.rocData[1] = []
        newPredictions = DOC.runDOC(percentages.detach().cpu().numpy(),self.docMu,Config.helper_variables["knowns_clss"],self.rocData[1])
        newPredictions = torch.tensor(newPredictions)
        for x in range(len(newPredictions)):
            newPredictions[x] = torch.tensor(helperFunctions.rerelabel[newPredictions[x].item()])
        return newPredictions

    def iiUnknown(self, percentages):
        import CodeFromImplementations.OpenNet as OpenNet
        unknowns = []
        for i in range(len(percentages)):
            unknowns.append(OpenNet.outlier_score(percentages[i],self.iiLoss_means))
        unknowns = torch.stack(unknowns)
        percentages = OpenNet.iimod(percentages,self.iiLoss_means)
        
        self.rocData[1] = unknowns#I do not know if this is correct
        unknowns = 2*unknowns.greater_equal(self.cutoff)

        return torch.cat([percentages,unknowns.unsqueeze(dim=-1)],dim=-1)


    #all functions here return a mask with 1 in all valid locations and 0 in all invalid locations
    typesOfUnknown = {"Soft":softMaxUnknown, "Open":openMaxUnknown, "Energy":energyUnknown, "Odin":odinUnknown, "COOL":normalThesholdUnknown, "SoftThresh":normalThesholdUnknown, "DOC":DOCUnknown, "iiMod": iiUnknown}

    #---------------------------------------------------------------------------------------------
    #This is the section for modifying the outputs for the final layer

    def softMaxMod(self,percentages:torch.Tensor):
        return torch.softmax(percentages, dim=1)

    def odinMod(self, percentages:torch.Tensor):
        print("ODIN is not working at the moment")
        import CodeFromImplementations.OdinCodeByWetliu as Odin
        self.model.openMax = False
        new_percentages = torch.tensor(Odin.ODIN(self.OdinX,self.model(self.OdinX), self.model, self.temp, self.noise))
        self.model.openMax = True
        return new_percentages[:len(percentages)]

    def FittedLearningEval(self, percentages:torch.Tensor):
        import CodeFromImplementations.FittedLearningByYhenon as fitted
        per = percentages.softmax(dim=1)
        store = []
        for x in per:
            store.append(fitted.infer(x,self.DOO,self.classCount))
        store = np.array(store)
        return torch.tensor(store)

    def DOCmod(self, logits:torch.Tensor):
        percent = torch.sigmoid(helperFunctions.renameClasses(logits))
        return percent

    def iiLoss_Means(self, percentages:torch.Tensor):
        import CodeFromImplementations.OpenNet as OpenNet
        self.iiLoss_means = OpenNet.Algorithm_1(self.weibulInfo["loader"],self.weibulInfo["net"])
        return percentages

    #all functions here return a tensor, sometimes it has an extra column for unknowns
    typesOfMod = {"Soft":softMaxMod, "Odin":odinMod, "COOL":FittedLearningEval, "SoftThresh":softMaxMod, "DOC":DOCmod, "iiMod":iiLoss_Means, "none":noChange}

    #---------------------------------------------------------------------------------------------
    #This is the section for training label modification

    def FittedLearningLabel(self,labelList:torch.Tensor):
        import CodeFromImplementations.FittedLearningByYhenon as fitted
        store = []
        for x in labelList:
            store.append(fitted.build_label(x,self.classCount,self.DOO))
        store = np.array(store)
        return torch.tensor(store,device=labelList.device)

    typesOfLabelMod = {"COOL":FittedLearningLabel}

    def labelMod(self,labelList:torch.Tensor):
        try:
            return self.typesOfLabelMod[self.type](self,labelList)
        except:
            return self.noChange(labelList)

    #---------------------------------------------------------------------------------------------
    #Some have specific training methods

    def iiTrain(self,batch,model):
        import CodeFromImplementations.OpenNet as OpenNet
        OpenNet.singleBatch(batch,model)

    typesOfTrainMod = {"iiMod":iiTrain}

    def trainMod(self,batch,model):
        try:
            return self.typesOfTrainMod[self.type](self,batch,model)
        except:
            return

    #---------------------------------------------------------------------------------------------
    #This is the section for resetting each epoch

    def resetvals(self):
        self.args = None    #This is the arguements for OPENMAX
        self.Save_score = []    #This is saving the score values for threshold for testing
        self.docMu = None    #This is saving the muStandards from DOC so that they dont need to be recalculated 
        if (not self.weibulInfo is None) and (not self.weibulInfo["weibull"] is None):
            self.weibulInfo["weibull"] = None
        self.rocData = [[],[]]  #This is the data for ROC of unknowns. First value is 1 if known data and 0 if unknown, second is the number before theshold.

    


