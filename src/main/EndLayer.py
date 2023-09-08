import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
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
root_folder = os.path.abspath(os.path.dirname(root_folder))
sys.path.append(root_folder)

root_path = os.getcwd()

import helperFunctions
from src.main.helperFunctions import NoExamples

class EndLayers(nn.Module):

    def __init__(self,num_classes: int,cutoff=0.25, type="Soft"):
        """
        Endlayers is a module that takes the final layer of the direct neural network and appplies one of several functions to it to separate unknowns.

        parameters:
            num_classes - the number of classes to sort into.
            cutoff - what value to use to cut things off and declare as unknowns. Not applied to Softmax (that would be softmax threshold.).
            type - what function to use.
        """
        super().__init__()
        self.cutoff = cutoff
        self.classCount = num_classes
        self.end_type = type
        self.DOO = Config.parameters["Degree of Overcompleteness"][0]    #Degree of Overcompleteness for COOL
        self.weibulInfo = None
        self.resetvals()



    def forward(self, output_true:torch.Tensor, y:torch.Tensor, type=None) -> torch.Tensor:
        """
        Module forward command, is used by __call__(). 
        This function takes the logits from the privious models and evaluates them according to the selected function.

        Note: this function requires self.prepWeibull() to have been called earlier. That stores some external data that some functions require.

        parameters:
            output_true - The true base level output logits from the last fully connected layer.
            y - The true labels of each item. Used to generate a precision recall curve score.
            type - Type override for what function to use.
        """
        startTime = time.time()
        if 1==2:
            print(f"Argmax")
            helperFunctions.printconfmat(output_true.cpu(),y.cpu())
        #check if a type is specified
        if type is None:
            type = self.end_type
        
        if type == "Energy":
            #Energy kind of reverses things.
            self.rocData[0] = y==Config.parameters["CLASSES"][0]
        else:
            self.rocData[0] = y!=Config.parameters["CLASSES"][0]

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
        """
        Internal function for setting arguements that are used for some functions.
        This function creates a class Args that stores the values so that they can be called with args.value
        We have done this to preserve some of the original code for implementations.
        """
        param = pd.read_csv(os.path.join("Saves","hyperparam","hyperParam.csv"))
        unknowns = pd.read_csv(os.path.join("Saves","unknown","unknowns.csv"))
        unknowns = unknowns["Unknowns"].to_list()
        if temp is None:
            temp = float(param["Temperature"][0])
        if classes is None:
            classes = len(Config.parameters["Knowns_clss"][0])

        class argsc():
            def __init__(self):
                """
                This class is purely for formatting because the original implementation used an arguement parser
                """
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
                """
                I think this is another version of helerFunctions.renameClassesLabeled(). That is what? The third version?
                TODO: clean this up. We don't need three different versions of a function that does this.
                """
                labels = labels.clone()
                lastval = -1
                label = list(range(Config.parameters["CLASSES"][0]))
                newout = []
                for val in Config.parameters["Unknowns_clss"][0]:
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
        """
        This stores a trainloader and training net for future refrence.
        Some of the algorithms want to create some sort of average values over the train loader. So this is to facilitate that.
        """
        self.weibulInfo = {"loader":trainloader,"device":device,"net":net, "weibull":None}


    def noChange(self,X:torch.Tensor):
        """
        A nothing function if no function is to be applied.
        """
        return X

    #---------------------------------------------------------------------------------------------
    #This is the section for adding unknown column

    def softMaxUnknown(self,percentages:torch.Tensor):
        """
        This is just Softmax. It adds a column of zeros to fit in with the rest of the algorithms.
        """
        self.Save_score.append(percentages.max(dim=1)[0].mean())
        #this adds a row that is of the cutoff amout so unless there is another value greater it will be unknown
        batchsize = len(percentages)
        unknownColumn = torch.zeros(batchsize,device=percentages.device)
        self.Save_score.append(unknownColumn)
        self.rocData[1] = unknownColumn
        return torch.cat((percentages,unknownColumn.unsqueeze(1)),dim=1)

    def normalThesholdUnknown(self,percentages:torch.Tensor):
        """
        This is a verson of softmax with threshold. It was turned down because "we cannot make changes to Softmax"
        """
        self.rocData[1] = percentages.max(dim=1)[0]
        self.Save_score.append(self.rocData[1].mean())
        #this adds a row that is of the cutoff amout so unless there is another value greater it will be unknown
        batchsize = len(percentages)
        unknownColumn = self.cutoff * torch.ones(batchsize,device=percentages.device)
        return torch.cat((percentages,unknownColumn.unsqueeze(1)),dim=1)

    def openMaxUnknown(self,percentages:torch.Tensor):
        """
        This runs the Openmax code we found. It has many problems so we used a lot of try/except blocks.
        The percentages are modified based on a weibul model. The weibul model is generated from the training data.

        In order the try/except blocks are:
            Import error:
                One of the Libraries that Openmax is dependent on cannot be loaded on one of our computers. 
                (Possibly from age or from 32 bit applications not being spported)
                This try/except block skips it if it cannot be loaded.
            Not Implemented error:
                Not sure why this one exists, probibly same reason as the import error.
            No Examples:
                Openmax requres the underlying algorithm to get at least one example of every class correct in order to build its model.
                Due to our unbalenced dataset that often fails. This catch will allow it to stop trying and continue the training.
        """
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
            except NoExamples:
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
        self.rocData[1] = percentages[:,:len(helperFunctions.mask)][:,helperFunctions.mask].max(dim=1)[0]
        self.Save_score.append(scores.squeeze())
        #scores.squeeze_().unsqueeze_(0)
        
        #return torch.cat((percentages,scores),dim=0)
        return percentages.to(device=percentages.device)
    
    def energyUnknown(self, percentages:torch.Tensor):
        """
        Modifies the output logits with energy based distribution and adds a column for unknowns.
        """
        if self.args is None:
            self.setArgs()
        import CodeFromImplementations.EnergyCodeByWetliu as Eng # useful link to import in relative directories https://stackoverflow.com/questions/4383571/importing-files-from-different-folder



        #The energy score code likes to output as a list
        scores = []
        Eng.energyScoreCalc(scores,percentages,self.args)
        scores = torch.tensor(np.array(scores),device=percentages.device)
        #after converting it to a tensor, the wrong dimention is expanded
        scores = -scores.squeeze(dim=0).unsqueeze(dim=1)
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
        """
        Unused but it used to modify percentages with ODIN, a previous idea from the creators of Energy OOD.
        """
        print("ODIN is not working at the moment")
        return percentages.max(dim=1,keepdim=True)[0].greater_equal(self.cutoff)

    def DOCUnknown(self, percentages:torch.Tensor):
        """
        I do not understand how DOC works but this runs the code we were able to find and returns a one hot prediction metrix.
        """
        import CodeFromImplementations.DeepOpenClassificationByLeishu02 as DOC
        if self.docMu is None:
            # print("Mu Standards need to be collected")
            if self.weibulInfo is None:
                return
            else:
                self.docMu = DOC.muStandardsFromDataloader(Config.parameters["Knowns_clss"][0],self.weibulInfo["loader"],self.weibulInfo["net"])
                self.Save_score = [torch.tensor(self.docMu)[:,1]]
        
        self.rocData[1] = []
        newPredictions = DOC.runDOC(percentages.detach().cpu().numpy(),self.docMu,Config.parameters["Knowns_clss"][0],self.rocData[1])
        newPredictions = torch.tensor(newPredictions)
        for x in range(len(newPredictions)):
            newPredictions[x] = torch.tensor(helperFunctions.rerelabel[newPredictions[x].item()])

        #to fit with the rest of the endlayers I am setting this back to a one hot vector even though we are just going to colapse it again.
        oneHotPredictions = F.one_hot(newPredictions,num_classes=Config.parameters["CLASSES"][0]+1).float()

        return oneHotPredictions

    def iiUnknown(self, percentages):
        """
        Uses the intra-spread and inter-separation to classify items as unknowns or knowns. 
        This score is then checked against the cutoff value and if it is less than the cutoff the unknown column is set to 2.
        The rest of the columns are limited in the range (0,1) so argmax will always evaluate to the unknown if the cutoff is not reached.
        """
        import CodeFromImplementations.iiMod as iiMod
        unknowns = []
        for i in range(len(percentages)):
            unknowns.append(iiMod.outlier_score(percentages[i],self.iiLoss_means))
        unknowns = torch.stack(unknowns)
        percentages = iiMod.iimod(percentages,self.iiLoss_means).softmax(dim=1)
        
        self.rocData[1] = unknowns#I do not know if this is correct
        unknowns = 2*unknowns.less_equal(self.cutoff)

        return torch.cat([percentages,unknowns.unsqueeze(dim=-1)],dim=-1)


    #all functions here return a mask with 1 in all valid locations and 0 in all invalid locations
    typesOfUnknown = {"Soft":softMaxUnknown, "Open":openMaxUnknown, "Energy":energyUnknown, "Odin":odinUnknown, "COOL":normalThesholdUnknown, "SoftThresh":normalThesholdUnknown, "DOC":DOCUnknown, "iiMod": iiUnknown}

    #---------------------------------------------------------------------------------------------
    #This is the section for modifying the outputs for the final layer

    def softMaxMod(self,percentages:torch.Tensor):
        """
        Just runs softmax
        """
        return torch.softmax(percentages, dim=1)

    def odinMod(self, percentages:torch.Tensor):
        """
        Some prerequisites for ODIN. ODIN does not currently work.
        """
        print("ODIN is not working at the moment")
        import CodeFromImplementations.OdinCodeByWetliu as Odin
        self.model.openMax = False
        new_percentages = torch.tensor(Odin.ODIN(self.OdinX,self.model(self.OdinX), self.model, self.temp, self.noise))
        self.model.openMax = True
        return new_percentages[:len(percentages)]

    def FittedLearningEval(self, percentages:torch.Tensor):
        """
        Collapses COOL into the standard number of classes from the increased compettitive number of classes.
        After that it is just standard Softmax Unknown.
        """
        import CodeFromImplementations.FittedLearningByYhenon as fitted
        per = percentages.softmax(dim=1)
        store = []
        for x in per:
            store.append(fitted.infer(x,self.DOO,self.classCount))
        store = np.array(store)
        return torch.tensor(store)

    def DOCmod(self, logits:torch.Tensor):
        """
        DOC uses a sigmoid layer.
        """
        percent = torch.sigmoid(helperFunctions.renameClasses(logits))
        return percent

    def iiLoss_Means(self, percentages:torch.Tensor):
        """
        This just calculates and saves the means for each class for use in iiUnknown()
        """
        import CodeFromImplementations.iiMod as iiMod
        self.iiLoss_means = iiMod.Algorithm_1(self.weibulInfo["loader"],self.weibulInfo["net"])
        return percentages

    #all functions here return a tensor, sometimes it has an extra column for unknowns
    typesOfMod = {"Soft":softMaxMod, "Odin":odinMod, "COOL":FittedLearningEval, "SoftThresh":softMaxMod, "DOC":DOCmod, "iiMod":iiLoss_Means, "none":noChange}

    #---------------------------------------------------------------------------------------------
    #This is the section for training label modification

    def FittedLearningLabel(self,labelList:torch.Tensor):
        """
        COOL changes the structure of the model to have a multiple of the number of classes. 
        This modifies the labels for training so that cross entropy training works with this expanded end layer.
        """
        import CodeFromImplementations.FittedLearningByYhenon as fitted
        store = []
        for x in labelList:
            store.append(fitted.build_label(x,self.classCount,self.DOO))
        store = np.array(store)
        return torch.tensor(store,device=labelList.device)

    typesOfLabelMod = {"COOL":FittedLearningLabel}

    def labelMod(self,labelList:torch.Tensor):
        """
        A way of creating a dictionary with a default value. There might be a simpler way of doing this.
        """
        try:
            return self.typesOfLabelMod[self.end_type](self,labelList)
        except:
            return self.noChange(labelList)

    #---------------------------------------------------------------------------------------------
    #Some have specific training methods

    def iiTrain(self,batch,model):
        """
        iimod adds a training function that needs to be used in order to group the output logits.
        (Energy also adds an optional training function as well but we did not add it (iimod was more of a last minute addition))
        """
        import CodeFromImplementations.iiMod as iiMod
        iiMod.singleBatch(batch,model)

    typesOfTrainMod = {"iiMod":iiTrain}

    def trainMod(self,batch,model):
        """
        A way of creating a dictionary with a default value. There might be a simpler way of doing this.
        """
        try:
            return self.typesOfTrainMod[self.end_type](self,batch,model)
        except:
            return

    #---------------------------------------------------------------------------------------------
    #This is the section for resetting each epoch

    def resetvals(self):
        """
        Resets the values in the Endlayer object.
        """
        self.args = None    #This is the arguements for OPENMAX
        self.Save_score = []    #This is saving the score values for threshold for testing
        self.docMu = None    #This is saving the muStandards from DOC so that they dont need to be recalculated 
        if (not self.weibulInfo is None) and (not self.weibulInfo["weibull"] is None):
            self.weibulInfo["weibull"] = None
        self.rocData = [[],[]]  #This is the data for ROC of unknowns. First value is 1 if known data and 0 if unknown, second is the number before theshold.

    


    def distance_by_batch(self,labels:torch.Tensor,outputs:torch.Tensor,means:list):
        """
        Finds the distances using iiMod's intra_spread function.
        """
        from CodeFromImplementations.iiMod import intra_spread
        return intra_spread(outputs[:,Config.parameters["Knowns_clss"][0]][labels!=Config.parameters["CLASSES"][0]],means,labels[labels!=Config.parameters["CLASSES"][0]])