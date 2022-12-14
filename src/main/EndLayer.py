import torch
import numpy as np
import torch.nn.functional as F
import pandas as pd
import Config


#three lines from https://xxx-cook-book.gitbooks.io/python-cook-book/content/Import/import-from-parent-folder.html
import os
import sys
root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)
import Config


root_path = os.getcwd()


class EndLayers():

    def __init__(self,num_classes: int,cutoff=0.25, type="Soft"):
        self.cutoff = cutoff
        self.classCount = num_classes
        self.type = type
        self.DOO = Config.parameters["Degree of Overcompleteness"][0]    #Degree of Overcompleteness for COOL
        self.weibulInfo = None
        self.resetvals()



    def endlayer(self, output_true:torch.Tensor, y:torch.Tensor, type=None, Train=False):
        #check if a type is specified
        if type is None:
            type = self.type
        
        #modify outputs
        if type != "Open":
            output_modified = self.typesOfMod[type](self,output_true)
        else:
            #Openmax needs labels but does not use them? Should I modify this?
            output_modified = self.typesOfMod[type](self,output_true,y)

        #This is supposted to add an extra column for unknowns
        output_complete = self.typesOfUnknown[type](self,output_modified)


        return output_complete




    #---------------------------------------------------------------------------------------------
    #This is the section for adding unknown column

    def softMaxUnknown(self,percentages:torch.Tensor):
        self.Save_score.append(percentages.max(dim=1)[0].mean())
        #this adds a row that is of the cutoff amout so unless there is another value greater it will be unknown
        batchsize = len(percentages)
        unknownColumn = self.cutoff * torch.zeros(batchsize,device=percentages.device)
        return torch.cat((percentages,unknownColumn.unsqueeze(1)),dim=1)

    def normalThesholdUnknown(self,percentages:torch.Tensor):
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
        import EnergyCodeByWetliu as Eng



        #The energy score code likes to output as a list
        scores = []
        Eng.energyScoreCalc(scores,percentages,self.args)
        scores = torch.tensor(np.array(scores),device=percentages.device)
        #after converting it to a tensor, the wrong dimention is expanded
        scores = -scores.squeeze().unsqueeze(dim=1)
        #This was to print the scores it was going to save
        #print(scores.sum()/len(scores))
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

    def DOCUnknown(self, percentages:torch.Tensor):
        import CodeFromImplementations.DeepOpenClassificationByLeishu02 as DOC
        if self.docMu is None:
            print("Mu Standards need to be collected")
            if self.weibulInfo is None:
                return
            else:
                self.docMu = DOC.muStandardsFromDataloader(Config.helper_variables["knowns_clss"],self.weibulInfo["loader"],self.weibulInfo["net"])
                self.Save_score = [torch.tensor(self.docMu)[:,1]]
        
        newPredictions = DOC.runDOC(percentages.detach().numpy(),self.docMu,Config.helper_variables["knowns_clss"])
        return torch.tensor(newPredictions)

    #all functions here return a mask with 1 in all valid locations and 0 in all invalid locations
    typesOfUnknown = {"Soft":softMaxUnknown, "Open":openMaxUnknown, "Energy":energyUnknown, "Odin":odinUnknown, "COOL":normalThesholdUnknown, "SoftThresh":normalThesholdUnknown, "DOC":DOCUnknown}

    #---------------------------------------------------------------------------------------------
    #This is the section for modifying the outputs for the final layer

    def softMaxMod(self,percentages:torch.Tensor):
        return torch.softmax(percentages, dim=1)

    

    def setArgs(self, classes=None, weibullThreshold=0.9, weibullTail=20, weibullAlpha=3, score="energy", m_in=-1, m_out=0, temp=None):
        param = pd.read_csv(os.path.join(root_path,"Saves","hyperparam","hyperParam.csv"))
        unknowns = pd.read_csv(os.path.join(root_path,"Saves","unknown","unknowns.csv"))
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


            def selectKnowns(self, modelOut, labels:torch.Tensor):
                labels = labels.clone()
                lastval = -1
                label = list(range(15))
                newout = []
                for val in Config.helper_variables["unknowns_clss"]["unknowns"]:
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
        self.weibulInfo = {"loader":trainloader,"device":device,"net":net}
        
        net.train()

    def openMaxMod(self,percentages:torch.Tensor, labels:torch.Tensor):
        failed = False
        
        try:
            import CodeFromImplementations.OpenMaxByMaXu as Open
        except ImportError:
            print("Openmax will be skipped as not all of its libraries could be loaded.")
            failed = True

        if self.args == None:
            self.setArgs()
        
        if not failed:
            try:
                scores_open = Open.openmaxevaluation([percentages.detach()],[labels.detach()],self.args,self.weibulInfo)
            except LookupError:
                print("OpenMax failed to idenitify at least 1 class!")
                #Note: usual reason for failure is having no correct examples for at least 1 class.
                failed = True
            except NotImplementedError:
                print("Warning: OpenMax has failed to load!")
                failed = True
                
        
        if failed:
            errorreturn = torch.zeros((percentages.size()))
            unknownColumn =torch.ones(len(percentages)).unsqueeze(1)
            errorreturn = torch.cat((errorreturn,unknownColumn),1)
            self.Save_score.append(torch.zeros(0))
            return errorreturn
            
        #print(scores_open)
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

    def FittedLearningEval(self, percentages:torch.Tensor):
        import CodeFromImplementations.FittedLearningByYhenon as fitted
        store = []
        for x in percentages:
            store.append(fitted.infer(x,self.DOO,self.classCount))
        store = np.array(store)
        return torch.tensor(store)

    def DOCmod(self, logits:torch.Tensor):
        percent = torch.sigmoid(logits)
        return percent

    def iiMod(self, percentages:torch.Tensor):
        #https://arxiv.org/pdf/1802.04365.pdf
        return percentages

    #all functions here return a tensor, sometimes it has an extra column for unknowns
    typesOfMod = {"Soft":softMaxMod, "Open":openMaxMod, "Energy":energyMod, "Odin":odinMod, "COOL":FittedLearningEval, "SoftThresh":softMaxMod, "DOC":DOCmod}

    #---------------------------------------------------------------------------------------------
    #This is the section for training label modification

    def noChange(self,X:torch.Tensor):
        return X

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
    #This is the section for resetting each epoch

    def resetvals(self):
        self.args = None    #This is the arguements for OPENMAX
        self.Save_score = []    #This is saving the score values for threshold for testing
        self.docMu = None    #This is saving the muStandards from DOC so that they dont need to be recalculated 

    


    