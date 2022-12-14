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


#Some unused commented code from FileHandling.
#-------------------------------------------------------------------------------------
#     if not os.path.exists(path):
#         os.mkdir(path)
#     torch.save(net.state_dict(),path+f"/Epoch{epoch:03d}.pth")
#     if phase is not None:
#         file = open("Saves/phase","w")
#         file.write(str(phase))
#         file.close()

# def loadPoint(net:AttackClassification, path:str):
#     if not os.path.exists(path):
#         os.mkdir(path)
#     i = 999
#     epochFound = 0
#     while i >= 0:
#         if os.path.exists(path+f"/Epoch{i:03d}.pth"):
#             net.load_state_dict(torch.load(path+f"/Epoch{i:03d}.pth"))
#             print(f"Loaded  model /Epoch{i:03d}.pth")
#             epochFound = i
#             i = -1
#         i = i-1
#     if i != -2:
#         print("No model to load found.")
#     elif os.path.exists(modelsavespath+"phase"):
#         file = open(modelsavespath+"phase","r")
#         phase = file.read()
#         file.close()
#         return int(phase),epochFound
#     return -1, -1
#
#
# def attemptedLoadcheck(model):
#     if attemptLoad and os.path.exists(modelsavespath+"phase"):
#         file = open(modelsavespath+"phase","r")
#         try:
#             startphase = int(file.read())
#         except:
#             startphase = 0
#         file.close()
#
#         model = cnn.Conv1DClassifier()
#         # model = nn.DataParallel(model)
#         model.to(device)
#         _,e = loadPoint(model, modelsavespath+"Saves")
#         e = e
#     for x in ["Soft","Energy","Open"]:
#         phase += 1
#         if phase<startphase:
#             continue
#         elif e==0:
#             model = Net()
#             model.to(device)
#         model.end.type=x
#         if x == "Open":
#             model.end.prepWeibull(train_loader,device,model)
#
#         Y_test = []
#         y_pred =[]
#         history_finaltyped = []
#         history_finaltyped += fit(num_epochs-e, lr, model, train_loader, val_loader, opt_func)
#         plots.store_values(history_finaltyped, y_pred, Y_test, num_epochs, x)
#         e=0
#         del model
#     model = Net()
#     model.to(device)
#     if attemptLoad:
#         loadPoint(model,modelsavespath+"Saves")
#     phase += 1
#
# def phasecheck(modelsavespath)
#     if attemptLoad and os.path.exists(modelsavespath+"phase"):
#         file = open(modelsavespath+"phase","w")
#         startphase = "0"
#         file.close()