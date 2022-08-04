import torch
import numpy as np
import torch.nn.functional as F

#three lines from https://xxx-cook-book.gitbooks.io/python-cook-book/content/Import/import-from-parent-folder.html
import os
import sys
root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)

class correctValCounter():

    def __init__(self,num_classes: int, type="Soft",cutoff=0.85, confusionMat=False, temp = 1, noise=0):
        self.cutoff = cutoff
        self.classCount = num_classes
        #storage of confusion matrixes can be activated
        self.conf = confusionMat
        #ODIN stuff
        self.temp = temp
        self.noise = noise

        #keep track of the type
        self.type = type
        self.zero()

    
    def evalN(self, output_true:torch.Tensor, y:torch.Tensor, type=None, offset=0, unknownRejection=True, indistribution = None):
        #evaluate normally
        if type is None:
            type = self.type
        #modify outputs
        output = self.typesOfMod[type](self,output_true)
        test = output.detach().numpy()

        #if it is not told if the data is in distribution, it will assume based on the offset.
        if indistribution == None:
            self.indistribution = offset==0
        else:
            self.indistribution = indistribution

        #format inputs
        if output.dim() == 2:
            out_val= torch.argmax(output[:,:self.classCount], dim=1)
        else:
            out_val = torch.tensor(output)
            
        out_oh = F.one_hot(out_val,self.classCount)

        if y.dim() == 2:
            y_val = torch.argmax(y, dim=1)-offset
            y_oh = F.one_hot(y_val,max(y_val.max().item(),self.classCount)+1)[:,:self.classCount]
        else:
            y_val = y-offset
            y_oh = F.one_hot(y_val,self.classCount)
        

        self.count+=len(out_val)
        self.count_by_class+=torch.bincount(y_val,minlength=self.classCount)[:self.classCount]

        if unknownRejection:
            #reject from unknown, foundunknowns is a mask to be applied to the onehot vector
            foundknowns = self.typesOfUnknown[type](self,output)
            out_oh = out_oh*foundknowns

        #holds percentages
        percentages = output[:,:self.classCount]
        if type=="Energy":
            percentages = F.softmax(percentages,dim=1)

        #finds where both are ones
        matches = out_oh * y_oh

        #this adds the chosen percentages to a list for future plotting
        self.plotPercentageList+=(percentages).flatten().tolist()

        #matches is the one hot vector and self.matches is the total of how many matches have been found per class
        self.matches += (matches).swapdims(0,1).sum(dim=1)

        #confusion matrix
        if self.conf:
            for num, x in enumerate(zip(y_val,out_val)):
                if out_oh[num].max().item() != 0:
                    self.matrix[x[0].item()][x[1].item()+1] += 1
                else:
                    self.matrix[x[0].item()][0] += 1


        #totals the number of times each class was guessed
        self.totals_for_guesses += out_oh.swapdims(0,1).sum(dim=1)

        #this calculates the sums of the softmax percentages of those it got correct
        self.correct_percentage_total += (matches*percentages).sum().item()


        #this totals the softmax percentages of those it gets wrong.
        self.incorrect_percentage_total += ((torch.ones(matches.shape)-matches)*percentages).sum().item()

        #this totals the percentages from predicted values that were wrong (specifically, where the highest values were incorrect)
        wrong_guesses = F.one_hot(out_val,self.classCount)-matches
        self.totalWrong += wrong_guesses.sum().item()
        self.wrong_percentage_total += (wrong_guesses*percentages).sum().item()

        #keep track of the percentages
        self.percentages_total += percentages.sum().item()
        #totals the percentages of all of the classes that were "guessed"(had highest probibility)
        self.mean_highest += (out_oh*percentages).swapdims(0,1).sum(dim=1)

    def PrintEval(self):
        correct = self.matches.sum().item()
        print(f"Count: {self.totals_for_guesses.sum().item()}/{self.count}")
        print(f"Correct: {100*correct/self.count}%")
        if correct!=0:
            print(f"Correct Mean Percentage: {100*self.correct_percentage_total/correct}")
        print(f"Cutoff: {self.cutoff}")
        total_incorrect = self.count*self.classCount-correct
        if total_incorrect!=0:
            print(f"Incorrect Mean Percentage: {100*self.incorrect_percentage_total/total_incorrect}")
        if self.totalWrong!=0:
            print(f"Wrong Mean Percentage: {100*self.wrong_percentage_total/self.totalWrong}")
        print(f"Accuracy: {self.accuracy().mean().item()}")

    def PrintUnknownEval(self):

        print("-----------------------------Unknowns-----------------------------")
        print("It guessed:")
        print(self.totals_for_guesses)
        print(f"That is: {self.totals_for_guesses.sum().item()} out of a total number of: {self.count} possible")
        print("It found the upper case for: ")
        print(self.matches)
        print(f"{100*(self.matches.sum()/self.totals_for_guesses.sum()).item()}% of those guessed were correct.")
        print("With mean percentages of:")
        print(100*self.mean_highest/self.totals_for_guesses)
        print(f"Percent guessed correctly: {self.matches.sum()/self.count}")
        print(f"Accuracy: {self.accuracy().mean().item()}")

    def storeConfusion(self, path):
        if self.conf:
            import pandas as pd
            df = pd.DataFrame(self.matrix.numpy())
            df.to_csv("csvOutput/"+path)


    def zero(self):
        #General stuff
        self.totalWrong = 0
        self.correct_percentage_total = 0
        self.incorrect_percentage_total = 0
        self.wrong_percentage_total = 0
        self.count = 0
        self.count_by_class = torch.zeros(self.classCount)
        self.totals_for_guesses = torch.zeros(self.classCount)
        self.mean_highest = torch.zeros(self.classCount)
        self.matches = torch.zeros(self.classCount)
        self.percentages_total = 0
        #data storage stuff
        self.plotPercentageList = []
        if self.conf:
            self.matrix = torch.zeros((self.classCount,self.classCount+1))
        self.highestPercentStorage = torch.zeros(1)
        

    #---------------------------------------------------------------------------------------------
    #This is the section for deturmining unknown masks

    def softMaxUnknown(self,percentages:torch.Tensor):
        return percentages.greater_equal(self.cutoff)

    def openMaxUnknown(self,percentages:torch.Tensor):
        #if the highest probibility is the unknown it throws the value out
        highest = torch.argmax(percentages, dim=1)
        yStar = (highest != self.classCount).to(torch.int)
        #extend the array so that it is batchSize X classSize                     it needs to be rotated
        #                                                                  I think I got the direction right
        yStarExt = torch.cat([yStar.unsqueeze(dim=0) for x in range(self.classCount)]).rot90(k=-1)
        return (yStarExt * percentages[:,:self.classCount].greater(self.cutoff))
    
    def energyUnknown(self, percentages:torch.Tensor):
        import CodeFromImplementations.EnergyCodeByWetliu as Eng

        scores = []
        Eng.energyScoreCalc(scores,percentages)
        yStar = torch.tensor(np.array(scores)).less_equal(-self.cutoff).squeeze(dim=0)
        return torch.cat([yStar.unsqueeze(dim=0) for x in range(self.classCount)]).rot90(k=-1)
        
    def odinUnknown(self, percentages:torch.Tensor):
        return percentages.max(dim=1,keepdim=True)[0].greater_equal(self.cutoff)

    #all functions here return a mask with 1 in all valid locations and 0 in all invalid locations
    typesOfUnknown = {"Soft":softMaxUnknown, "Open":openMaxUnknown, "Energy":energyUnknown, "Odin":odinUnknown}

    #---------------------------------------------------------------------------------------------
    #This is the section for modifying the outputs for the final layer

    def softMaxMod(self,percentages:torch.Tensor):
        return torch.softmax(percentages,dim=1)

    def setWeibull(self, weibull):
        self.weibullmodel = weibull

    def openMaxMod(self,percentages:torch.Tensor):
        if self.weibullmodel == None:
            print("Warning: OpenMax needs a Weibull model!")
            return None

        import CodeFromImplementations.OpenMaxByMaXu as Open

        output_open = []
        for logits in percentages:
            #this is where the openmax is run, I did not create the openmax
            output_open_new, _ = Open.openmax(self.weibullmodel, list(range(self.classCount)),logits.detach().numpy()[np.newaxis,:], 0.1, alpha=min(self.classCount, 10))
            output_open.append(torch.tensor(output_open_new).unsqueeze(dim=0))
        output_open = torch.cat(output_open, dim=0)
        return output_open
    
    def energyMod(self, percentages:torch.Tensor):
        return percentages
        
    def odinSetup(self, X, model, temp, noise):
        #most of this line is from the ODIN implementation (line 27)
        self.OdinX = torch.autograd.Variable(X, requires_grad = True)
        self.model = model
        self.temp = temp
        self.noise = noise

    def odinMod(self, percentages:torch.Tensor):
        import CodeFromImplementations.OdinCodeByWetliu as Odin
        self.model.openMax = False
        new_percentages = torch.tensor(Odin.ODIN(self.OdinX,self.model(self.OdinX), self.model, self.temp, self.noise))
        self.model.openMax = True
        return new_percentages[:len(percentages)]

    #all functions here return a tensor, sometimes it has an extra column for unknowns
    typesOfMod = {"Soft":softMaxMod, "Open":openMaxMod, "Energy":energyMod, "Odin":odinMod}


    #-------------------------------------------------------------------------------------
    #This is the section for the confusion matrix
    #The equations were from https://en.wikipedia.org/wiki/Sensitivity_and_specificity

    def N(self):
        #Eegative
        #this is a clumsy implementation that counts the negitive results.
        total = self.count_by_class.sum().item()
        if self.indistribution:
            return -self.count_by_class+total
        else:
            return torch.ones(self.count_by_class.shape)*total

    def TP(self):
        #True Positive
        return self.matches

    def FP(self):
        #False Positive
        #every guess minus those guessed correctly
        return self.totals_for_guesses-self.matches

    def FN(self):
        #False Negative
        return self.count_by_class-self.matches
    
    def TN(self):
        #True Negative
        #This is messy, it is the false positives minus the total number of negatives.
        return (self.N()) - self.FP()


    def TPR(self):
        #True Positive Rate
        return  self.TP()/self.count_by_class
    
    def FNR(self):
        #False Negative Rate
        return 1-self.TPR()

    def FPR(self):
        #False Positive Rate
        return (self.FP())/(-self.count_by_class+self.count_by_class.sum().item())

    def TNR(self):
        #True Negative Rate
        return 1-self.FPR()

    def PPR(self):
        #Predicted Positive Rate
        return self.TP()/self.totals_for_guesses
    
    def FOR(self):
        #False Omission Rate
        return self.FN()/(-self.totals_for_guesses + self.count_by_class.sum().item())
    
    def FDR(self):
        #False Detection Rate
        return 1-self.PPR()

    def NPV(self):
        #Negaitive Predicted Value
        return 1-self.FOR()

    def fScore(self):
        #Positive predictive Value
        precision = self.PPR()
        #True Positive Rate
        recall = self.TPR()
        return 2*(precision*recall)/(precision+recall)

    def accuracy(self):
        return (self.TP()+self.TN())/(torch.zeros(self.classCount)+self.count_by_class.sum().item())

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
