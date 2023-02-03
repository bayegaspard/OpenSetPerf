from torch import nn
import torch
from torch.nn import functional as F
import os

### user defined functions
import Config
from EndLayer import EndLayers
import GPU
import FileHandling
import helperFunctions



class ModdedParallel(nn.DataParallel):
    # From https://github.com/pytorch/pytorch/issues/16885#issuecomment-551779897
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
        


class AttackTrainingClassification(nn.Module):
    def __init__(self):
        super().__init__()

        numClasses = 15
        if Config.parameters['Datagrouping'][0] == "DendrogramChunk":
            numClasses = numClasses*32

        self.activation = nn.ReLU()
        if Config.parameters["Activation"][0] == "Sigmoid":
            self.activation = nn.Sigmoid()
        if Config.parameters["Activation"][0] == "Tanh":
            self.activation = nn.Tanh()
        if Config.parameters["Activation"][0] == "Leaky":
            self.activation = nn.LeakyReLU()
        if Config.parameters["Activation"][0] == "Elu":
            self.activation = nn.ELU()
        if Config.parameters["Activation"][0] == "PRElu":
            self.activation = nn.PReLU()
        if Config.parameters["Activation"][0] == "Swish":
            print("Swish is not implemented yet")
        if Config.parameters["Activation"][0] == "maxout":
            print("maxout is not implemented yet")
        if Config.parameters["Activation"][0] == "Softplus":
            self.activation = nn.Softplus()
        if Config.parameters["Activation"][0] == "Softmax":
            #why softmax?
            self.activation = nn.Softmax(dim=1)


        self.fc1 = nn.Linear(11904, Config.parameters["Nodes"][0])
        self.fc2 = nn.Linear(Config.parameters["Nodes"][0], numClasses)


        self.addedLayers = torch.nn.Sequential()
        for x in range(Config.parameters["Number of Layers"][0]):
            self.addedLayers.append(torch.nn.Linear(Config.parameters["Nodes"][0],Config.parameters["Nodes"][0]))
            self.addedLayers.append(self.activation)

        # self.COOL = nn.Linear(256, 15*n)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(int(Config.parameters["Dropout"][0]))

        self.end = EndLayers(numClasses, type="Soft", cutoff=Config.parameters["threshold"][0])
        self.batchnum = 0
        self.device = GPU.get_default_device()
        self.store = GPU.to_device(torch.tensor([]), self.device), GPU.to_device(torch.tensor([]), self.device), GPU.to_device(torch.tensor([]), self.device)

        self.COOL = nn.Linear(Config.parameters["Nodes"][0], numClasses*self.end.DOO)
        # self.model = model
        # self.batch = batch
        # self.to_device = to_device
        # self.device = device
        # self.Y_Pred = Y_Pred
        # self.Y_test = Y_test
        self.los = False

        
    # Specify how the data passes in the neural network
    def forward(self, x: torch.Tensor):
        # x = to_device(x, device)
        x = x.float()
        x = x.unsqueeze(1)
        #print(f"start: {x.shape}")
        x = self.layer1(x)
        #print(f"middle: {x.shape}")
        x = self.layer2(x)
        #print(f"end: {x.shape}")
        x = self.flatten(x)
        x = self.activation(self.fc1(x))
        x = self.addedLayers(x)
        x = self.dropout(x)
        if self.end.type != "COOL":
            x = self.fc2(x)
        else:
            x = self.COOL(x)
        # print("in forward", F.log_softmax(x, dim=1))
        return x
        # return F.log_softmax(x, dim=1)



    def training_step(self, batch):
        data, labels = batch
        labels = labels[:,0]    #Select the data we want not the metadata
        out = self(data)  # Generate predictions
        labels = self.end.labelMod(labels)

        
        #Not sure if this is nessiary. 
        if self.end == "DOC":
            out = nn.Sigmoid()(out)
        
        


        # out = DeviceDataLoader(out, device)
        loss = F.cross_entropy(out, labels)  # Calculate loss
        torch.cuda.empty_cache()
        # print("loss from training step ... ", loss)
        return loss

    @torch.no_grad()
    def evaluate(self, validationset):
        self.eval()
        self.batchnum = 0
        outputs = [self.validation_step(batch) for batch in validationset]  ### reverted bac
        return self.validation_epoch_end(outputs)

    def accuracy(self, outputs:torch.Tensor, labels):
        if outputs.ndim == 2:
            preds = torch.argmax(outputs, dim=1)
        else:
            #DOC already applies an argmax equivalent so we do not apply one here.
            preds = outputs
        # print("preds from accuracy", preds)
        # print("labels from accuracy", labels)
        # Y_Pred.append(preds.tolist()[:])
        # Y_test.append(labels.tolist()[:])
        # preds = torch.tensor(preds)

        #First is the guess, second is the actual class and third is the class to consider correct.
        self.store = torch.cat((self.store[0], preds)), torch.cat((self.store[1], labels[:,1])),torch.cat((self.store[2], labels[:,0]))
        return torch.tensor(torch.sum(preds == labels[:,0]).item() / len(preds))
        # def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):

    def fit(self, epochs, lr, train_loader, val_loader, opt_func):
        #print("test1.1")
        history = []
        optimizer = opt_func(self.parameters(), lr)
        self.los = helperFunctions.LossPerEpoch("TestingDuringTrainEpochs.csv")
        # torch.cuda.empty_cache()
        if epochs > 0:
            for epoch in range(epochs):
                #print("test1.2")
                self.end.resetvals()
                self.store = GPU.to_device(torch.tensor([]), self.device), GPU.to_device(torch.tensor([]), self.device), GPU.to_device(torch.tensor([]), self.device)
                # Training Phase
                self.train()
                train_losses = []
                num = 0
                for batch in train_loader:
                    #print("Batch")
                    # batch = to_device(batch,device)
                    # batch = DeviceDataLoader(batch, device)
                    loss = self.training_step(batch)

                    FileHandling.write_batch_to_file(loss, num, self.end.type, "train")
                    train_losses.append(loss.detach())
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    num += 1

                #print("test1.3")
                # Validation phase
                self.savePoint(f"Saves", epoch, Config.helper_variables["phase"])
                result = self.evaluate(val_loader)
                #print("test1.4")
                result['train_loss'] = torch.stack(train_losses).mean().item()
                result["epoch"] = epoch
                #print("test1.5")
                self.epoch_end(epoch, result)
                #print("result", result)

                history.append(result)
                #print("test1.6")
                self.los.collect()
        else:
            # Validation phase
            self.loadPoint("Saves")
            result = self.evaluate(val_loader)
            result['train_loss'] = -1
            self.epoch_end(0, result)
            #print("result", result)
            history.append(result)
        return history

    def validation_step(self, batch):
        #self.eval()
        #self.savePoint("test", phase=Config.helper_variables["phase"])
        data, labels_extended = batch
        self.batchnum += 1
        labels = labels_extended[:,0]
        out = self(data)  # Generate predictions
        zeross = GPU.to_device(torch.zeros(len(out),1),self.device)
        loss = F.cross_entropy(torch.cat((out,zeross),dim=1), labels)  # Calculate loss
        out = self.end.endlayer(out,
                                labels)  # <----Here is where it is using Softmax TODO: make this be able to run all of the versions and save the outputs.
        # out = self.end.endlayer(out, labels, type="Open")
        # out = self.end.endlayer(out, labels, type="Energy")

        

        # Y_pred = out
        # Y_test = labels
        # print("y-test from validation",Y_test)
        # print("y-pred from validation", Y_pred)
        if out.ndim == 2:
            unknowns = out[:, 15].mean()
            test = torch.argmax(out, dim=1)
        else:
            unknowns = torch.zeros(out.shape)

        #This is just for datacollection.
        if self.los:
            self.los.addloss(torch.argmax(out,dim=1),labels)

        out = GPU.to_device(out, self.device)
        acc = self.accuracy(out, labels_extended)  # Calculate accuracy
        FileHandling.write_batch_to_file(loss, self.batchnum, self.end.type, "Saves")
        #print("validation accuracy: ", acc)
        return {'val_loss': loss.detach(), 'val_acc': acc, "val_avgUnknown": unknowns}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        batch_unkn = self.end.Save_score
        self.end.Save_score = []
        epoch_unkn = torch.stack(batch_unkn).mean()  # Combine Unknowns

        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item(), "val_avgUnknown": epoch_unkn.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(epoch,
                                                                                         result['train_loss'],
                                                                                         result['val_loss'],
                                                                                         result['val_acc']))

    def savePoint(net, path: str, epoch=0, phase=None):
        if not os.path.exists(path):
            os.mkdir(path)
        torch.save(net.state_dict(), path + f"/Epoch{epoch:03d}.pth")
        if phase is not None:
            file = open("Saves/phase", "w")
            file.write(str(phase))
            file.close()

    def loadPoint(net, path: str):
        if not os.path.exists(path):
            os.mkdir(path)
        i = 999
        epochFound = 0
        while i >= 0:
            if os.path.exists(path + f"/Epoch{i:03d}.pth"):
                net.load_state_dict(torch.load(path + f"/Epoch{i:03d}.pth"))
                print(f"Loaded  model /Epoch{i:03d}.pth")
                epochFound = i
                i = -1
            i = i - 1
        if i != -2:
            print("No model to load found.")
        elif os.path.exists("Saves/phase"):
            file = open("Saves/phase", "r")
            phase = file.read()
            file.close()
            return int(phase), epochFound
        return -1, -1
    
        






class Conv1DClassifier(AttackTrainingClassification):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 32, 3),
            self.activation,
            nn.MaxPool1d(4),
            nn.Dropout(int(Config.parameters["Dropout"][0])))
        self.layer2 = nn.Sequential(
            nn.Conv1d(32, 64, 3),
            self.activation,
            nn.MaxPool1d(2),
            nn.Dropout(int(Config.parameters["Dropout"][0])))

        
        

        



class FullyConnected(AttackTrainingClassification):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(1504,12000),
            self.activation,
            nn.Dropout(int(Config.parameters["Dropout"][0])))
        self.layer2 = nn.Sequential(
            nn.Linear(12000,11904),
            self.activation,
            nn.Dropout(int(Config.parameters["Dropout"][0])))





