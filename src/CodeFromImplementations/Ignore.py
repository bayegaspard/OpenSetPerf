#ignore this file, it was my attempt at doing things  but I don't have the heart to throw it away
#Fitted Learning
import torch

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

# if self.end.type == "COOL":
#     labels = self.end.COOL_Label_Mod(labels)
#     out = torch.split(out.unsqueeze(dim=1), 15, dim=2)
#     out = torch.cat(out, dim=1)

#Finding values for theshold
    #-------------------------------------------------------------------------------------
    #crates a cutoff with a certian percent labeled to be in distribution

    def cutoffStorage(self, newNumbers:torch.Tensor, type=None):
        if type is None:
            type = self.type
        if type == "Energy":
            import EnergyCodeByWetliu as Eng
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

