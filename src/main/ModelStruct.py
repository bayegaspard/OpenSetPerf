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
from sklearn.metrics import (precision_score, recall_score, average_precision_score)

device = GPU.get_default_device()

class ModdedParallel(nn.DataParallel):
    """
    If the default torch DataParallel cannot find an atribute than it tries to get it from a contained module.
    """
    # From https://github.com/pytorch/pytorch/issues/16885#issuecomment-551779897
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
        


class AttackTrainingClassification(nn.Module):
    """This is the Default Model for the project"""
    def __init__(self):
        super().__init__()

        #This is the length of the packets in the dataset we are currently using.
        fullyConnectedStart = 11904

        #There are 15 classes
        numClasses = Config.parameters["CLASSES"][0]
        if Config.parameters['Datagrouping'][0] == "DendrogramChunk":
            numClasses = numClasses*32
        
        #These are for DOC, it has a special model structure. Because of that we allow it to override what we have.
        if Config.parameters['OOD Type'][0] == "DOC":
            self.DOC_kernels = nn.ModuleList()
            for x in Config.DOC_kernels:
                self.DOC_kernels.append(nn.Conv1d(1, 32, x,device=device))
            fullyConnectedStart= 1501*len(Config.DOC_kernels)

        #This (poorly made) menu switches between the diffrent options for activation functions.
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
            self.activation = nn.PReLU(device=device)
        if Config.parameters["Activation"][0] == "Swish":
            print("Swish is not implemented yet")
        if Config.parameters["Activation"][0] == "maxout":
            print("maxout is not implemented yet")
        if Config.parameters["Activation"][0] == "Softplus":
            self.activation = nn.Softplus()
        if Config.parameters["Activation"][0] == "Softmax":
            #why softmax?
            self.activation = nn.Softmax(dim=1)


        #We use two normal fully connected layers after the CNN specific layers (or substiute layers)
        self.fc1 = nn.Linear(fullyConnectedStart, Config.parameters["Nodes"][0],device=device)
        self.fc2 = nn.Linear(Config.parameters["Nodes"][0], numClasses,device=device)


        self.addedLayers = torch.nn.Sequential()
        #If the config says to add more layers, that is done here.
        for x in range(Config.parameters["Number of Layers"][0]):
            self.addedLayers.append(torch.nn.Linear(Config.parameters["Nodes"][0],Config.parameters["Nodes"][0],device=device))
            self.addedLayers.append(self.activation)

        # self.COOL = nn.Linear(256, 15*n)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(int(Config.parameters["Dropout"][0]))

        self.end = EndLayers(numClasses, type="Soft", cutoff=Config.parameters["threshold"][0])
        self.batchnum = 0
        self.storeReset()
        #COOL neeeds its own outputl layer.
        self.COOL = nn.Linear(Config.parameters["Nodes"][0], numClasses*self.end.DOO,device=device)

        self.los = False
        self.mode = None

        
    # Specify how the data passes in the neural network
    def forward(self, x: torch.Tensor):
        """Runs the model through all the standard layers
        
        also uses the Compettitive Overcomplete Output Layer alternative layer if the setting is for COOL.
        """
        # x = to_device(x, device)
        x = x.float()
        x = x.unsqueeze(1)
        
        if self.mode == None:
            self.mode = self.end.type
        if self.mode != "DOC":
            x = self.layer1(x)
            x = self.layer2(x)
        else:
            #Gotten the device location from https://discuss.pytorch.org/t/dataparallel-arguments-are-located-on-different-gpus/42054/5
            # print(f"model:{self.fc1.weight.device}")
            # print(f"layer:{self.DOC_kernels[0].weight.device}")
            # print(f"data:{x.device}")
            xs = [alg(x) for alg in self.DOC_kernels]
            xs = [a.max(dim=1)[0] for a in xs]
            x = torch.concat(xs,dim=-1)
        

        x = self.flatten(x)
        x = self.activation(self.fc1(x))
        x = self.addedLayers(x)
        x = self.dropout(x)
        if self.end.type != "COOL":
            x = self.fc2(x)
        else:
            x = self.COOL(x)
        
        return x
        

    def fit(self, epochs, lr, train_loader, test_loader,val_loader, opt_func):
        """
        Trains the model on the train_loader and evaluates it off of the val_loader. Also it stores all of the results in model.store.
        It also generates a new line of ScoresAll.csv that stores all of the data for this model. (note: if you are running two threads at once the data will be overriden)

        parameters:
            epochs- the number of epochs to run the training for
            lr- the learning rate to start at, the schedular will change the learning rate over time and can be set in the Config
            train_loader- the torch dataloader to train the model on
            test_loader- Unused but this should be the data that you want to do the final test of the model on, also in the form of a torch dataloader
            val_loader- the model is tested every epoch against this data
            opt_func- the optimization function to use

        returns:
            history- a list of tuples, each containing:
                val_loss - the loss from the validation stage
                val_acc - the accuract from the validation  stage. Note: this accuracy is not used in the save.
                val_avgUnknown - internal metric that is not used
                epoch - the epoch that this data was taken
                train_loss - the average training loss per batch of this epoch
        """
        if Config.parameters["attemptLoad"][0] == 1:
            startingEpoch = self.loadPoint("Saves/models")
        else:
            startingEpoch = 0
        history = []
        optimizer = opt_func(self.parameters(), lr)
        if isinstance(Config.parameters["SchedulerStep"][0],float) and Config.parameters["SchedulerStep"][0] !=0:
            sch = torch.optim.lr_scheduler.StepLR(optimizer, Config.parameters["SchedulerStepSize"][0], Config.parameters["SchedulerStep"][0])
        else:
            sch = None
        self.los = helperFunctions.LossPerEpoch("TestingDuringTrainEpochs.csv")
        FileHandling.create_params_All()
        # torch.cuda.empty_cache()
        if epochs > 0:
            for epoch in range(epochs):
                self.end.resetvals()
                self.storeReset()
                # Training Phase
                self.train()
                train_losses = []
                num = 0
                for batch in train_loader:
                    #print("Batch")
                    # batch = to_device(batch,device)
                    # batch = DeviceDataLoader(batch, device)
                    loss = self.training_step(batch)

                    #FileHandling.write_batch_to_file(loss, num, self.end.type, "train")
                    train_losses.append(loss.detach())
                    self.end.trainMod(batch,self)
                    #print(loss)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    num += 1

                if not sch is None:
                    sch.step()
                
                # Validation phase
                self.savePoint(f"Saves/models", epoch+startingEpoch)
                result = self.evaluate(val_loader)
                result['train_loss'] = torch.stack(train_losses).mean().item()
                FileHandling.addMeasurement(f"Epoch{epoch+startingEpoch} loss",result['train_loss'])
                result["epoch"] = epoch+startingEpoch
                self.epoch_end(epoch+startingEpoch, result)
                #print("result", result)

                history.append(result)
                self.los.collect()
        else:
            # Validation phase
            epoch = self.loadPoint("Saves/models")
            result = self.evaluate(val_loader)
            result['train_loss'] = -1
            self.epoch_end(epoch, result)
            #print("result", result)
            history.append(result)
        return history


    def training_step(self, batch):
        """Preforms a step for training the model but does not begin backpropigation
        
        Parameters:
            Batch- a torch dataloader batch. Which is a tuple of tensors.

        Returns:
            Loss- a torch loss that signifies how far away from the expected targets the model got.
        
        """
        data, labels = batch
        #Our labels have two values per line so that we can tell what the unknowns are.
        labels = labels[:,0]    #Select the data we want not the metadata
        out = self(data)  # Generate predictions
        labels = self.end.labelMod(labels)

        
        #Not sure if this is nessiary. 
        if self.end == "DOC":
            out = nn.Sigmoid()(out)
        
        
        

        # out = DeviceDataLoader(out, device)
        loss = F.cross_entropy(out, labels)  # Calculate loss
        #torch.cuda.empty_cache()
        # print("loss from training step ... ", loss)
        return loss

    def validation_step(self, batch):
        """
        Takes a batch from the validation loader and evaluates it using the endlayer.

        parameters:
            batch- a single batch from a torch dataloader
        
        returns:
            dictionary of:
                val_loss - the loss from the validation stage
                val_acc - the accuract from the validation  stage. Note: this accuracy is not used in the save.
                val_avgUnknown - internal metric that is not used
        """
        self.eval()
        #self.savePoint("test", phase=Config.helper_variables["phase"])
        data, labels_extended = batch
        self.batchnum += 1
        labels = labels_extended[:,0]
        
        out = self(data)  # Generate predictions
        #zeross = GPU.to_device(torch.zeros(len(out),1),device)
        zeross = GPU.to_device(torch.zeros(len(out),1),device)
        loss = F.cross_entropy(torch.cat((out,zeross),dim=1), labels)  # Calculate loss
        out = self.end.endlayer(out,
                                labels).to(labels.device)  # <----Here is where it is using Softmax TODO: make this be able to run all of the versions and save the outputs.
        #loss = F.cross_entropy(torch.cat((out,zeross),dim=1), labels)  # Calculate loss
        # out = self.end.endlayer(out, labels, type="Open")
        # out = self.end.endlayer(out, labels, type="Energy")

        

        # Y_pred = out
        # Y_test = labels
        # print("y-test from validation",Y_test)
        # print("y-pred from validation", Y_pred)
        if out.ndim == 2:
            unknowns = out[:, Config.parameters["CLASSES"][0]].mean()
            test = torch.argmax(out, dim=1)
        else:
            unknowns = torch.zeros(out.shape)

        #This is just for datacollection.
        if self.los:
            if self.end.type == "DOC" or self.end.type == "COOL":
                self.los.addloss(out,labels)
            else:
                self.los.addloss(torch.argmax(out,dim=1),labels)

        out = GPU.to_device(out, device)
        acc = self.accuracy(out, labels_extended)  # Calculate accuracy
        #FileHandling.write_batch_to_file(loss, self.batchnum, self.end.type, "Saves")
        #print("validation accuracy: ", acc)
        return {'val_loss': loss.detach(), 'val_acc': acc, "val_avgUnknown": unknowns}


    @torch.no_grad()
    def evaluate(self, validationset):
        """
        Evaluates the given dataset on this model.
        
        parameters:
            torch dataset to iterate through.
        
        returns:
            A dictionary of all of the mean values from the run consisting of:
                val_loss - the loss from the validation stage
                val_acc - the accuract from the validation  stage
                val_avgUnknown - internal metric that is not used
        """
        self.eval()
        self.batchnum = 0
        outputs = [self.validation_step(batch) for batch in validationset]  ### reverted bac
        return self.validation_epoch_end(outputs)

    def accuracy(self, outputs:torch.Tensor, labels):
        """
        Finds the final accuracy of the batch and saves the predictions and true values to model.store

        parameters:
            outputs- a torch Tensor containing all of the outputs from the models forward pass.
            labels- a torch Tensor containing all of the true and tested against values corresponding to the output results
                labels[:,0]- should be the values that you are training the model to get
                labels[:,1:]- should be the true values or other metadata that you want to store associated with a datapoint.
        
        returns:
            A dictionary of all of the mean values from the run.
        """
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

        #Find out if something failed, if it did get no accuracy
        if outputs.max() == 0:
            return torch.tensor(0.0)


        #First is the guess, second is the actual class and third is the class to consider correct.
        self.store = torch.cat((self.store[0], preds)), torch.cat((self.store[1], labels[:,1])),torch.cat((self.store[2], labels[:,0]))
        return torch.tensor(torch.sum(preds == labels[:,0]).item() / len(preds))
        # def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):



    def validation_epoch_end(self, outputs):
        """
        Takes the output of each epoch and takes the mean values. Returns a dictionary of those mean values.

        returns:
            dictionary of:
                val_loss - the loss from the validation stage
                val_acc - the accuract from the validation  stage
                val_avgUnknown - internal metric that is not used
        """
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        batch_unkn = self.end.Save_score
        self.end.Save_score = []
        if len(batch_unkn) != 0:
            epoch_unkn = torch.stack(batch_unkn).mean()  # Combine Unknowns
        else:
            epoch_unkn = torch.tensor(0)

        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item(), "val_avgUnknown": epoch_unkn.item()}

    def epoch_end(self, epoch, result):
        """
        prints the results using the epoch 
        """
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(epoch,
                                                                                         result['train_loss'],
                                                                                         result['val_loss'],
                                                                                         result['val_acc']))

    def savePoint(net, path: str, epoch=0):
        """
        Saves the model

        parameters:
            path- path to the folder to store the model in.
            epoch- the number of epochs the model has trained.

        """
        if not os.path.exists(path):
            os.mkdir(path)
        torch.save(net.state_dict(), path + f"/Epoch{epoch:03d}{Config.parameters['OOD Type'][0]}.pth")

    def loadPoint(net, path: str):
        """
        Loads the most trained model from the path. Note: will break if trying to load a model with different configs.

        parameters:
            path- path to the folder where the models are stored.

        returns:
            epochFound - the number of epochs the model that was found has run for.
        """
        if not os.path.exists(path):
            os.mkdir(path)
        i = 999
        epochFound = -1
        while i >= 0:
            if os.path.exists(path + f"/Epoch{i:03d}{Config.parameters['OOD Type'][0]}.pth"):
                net.load_state_dict(torch.load(path + f"/Epoch{i:03d}{Config.parameters['OOD Type'][0]}.pth"))
                print(f"Loaded  model /Epoch{i:03d}{Config.parameters['OOD Type'][0]}.pth")
                epochFound = i
                i = -1
            i = i - 1
        if epochFound == -1:
            print("No model to load found.")
        return epochFound

    
    #This loops through all the thresholds without resetting the model.
    def thresholdTest(net,val_loader):
        """
        This tests the results from val_loader at various thresholds and saves it to scoresAll.csv
        """
        net.end.type = Config.parameters["OOD Type"][0]
        net.loadPoint("Saves/models")
        thresh = Config.thresholds
        for y in range(len(thresh)):
            x = thresh[y]
            #reset
            net.end.resetvals()
            net.store = GPU.to_device(torch.tensor([]), device), GPU.to_device(torch.tensor([]), device), GPU.to_device(torch.tensor([]), device)
            net.end.cutoff = x
            
            #evaluate
            net.evaluate(val_loader)

            #get the data
            y_pred,y_true,y_tested_against = net.store

            #evaluate the data
            y_true = y_true.to(torch.int).tolist()
            y_pred = y_pred.to(torch.int).tolist()
            y_tested_against = y_tested_against.to(torch.int).tolist()
            recall = recall_score(y_tested_against,y_pred,average='weighted',zero_division=0)
            precision = precision_score(y_tested_against,y_pred,average='weighted',zero_division=0)
            f1 = 2 * (precision * recall) / (precision + recall)
            #save the f1 score
            FileHandling.create_params_Fscore("",f1,x)
            FileHandling.addMeasurement(f"Threshold {x} Fscore",f1)

            #Generate and save confusion matrix
            # if plots.name_override:
            #     if y > 0:
            #         plots.name_override = plots.name_override.replace(f" Threshold:{thresh[y-1]}",f" Threshold:{x}")
            #     else:
            #         plots.name_override = plots.name_override+f" Threshold:{x}"
            # else:
            #     plots.name_override = f"Default_setting Threshold:{x}"

            # class_names = Dataload.get_class_names(range(15))
            # for x in Config.helper_variables["unknowns_clss"]["unknowns"]:
            #     class_names[x] = class_names[x]+"*"
            # class_names.append("*Unknowns")
            #cnf_matrix = plots.confusionMatrix(y_true.copy(), y_pred.copy(), y_tested_against.copy()) 

            #plots.plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
            #                title='Confusion matrix', knowns = Config.helper_variables["knowns_clss"])

    def storeReset(self):
        """
        Resets the storage for the model.
        """
        self.store = GPU.to_device(torch.tensor([]), device), GPU.to_device(torch.tensor([]), device), GPU.to_device(torch.tensor([]), device)
    
    
        






class Conv1DClassifier(AttackTrainingClassification):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 32, 3,device=device),
            self.activation,
            nn.MaxPool1d(4),
            nn.Dropout(int(Config.parameters["Dropout"][0])))
        self.layer2 = nn.Sequential(
            nn.Conv1d(32, 64, 3,device=device),
            self.activation,
            nn.MaxPool1d(2),
            nn.Dropout(int(Config.parameters["Dropout"][0])))

        
        

        



class FullyConnected(AttackTrainingClassification):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(1504,12000,device=device),
            self.activation,
            nn.Dropout(int(Config.parameters["Dropout"][0])))
        self.layer2 = nn.Sequential(
            nn.Linear(12000,11904,device=device),
            self.activation,
            nn.Dropout(int(Config.parameters["Dropout"][0])))





