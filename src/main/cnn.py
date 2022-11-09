from torch import nn
import torch
from torch.nn import functional as F


### user defined functions
import Config
from EndLayer import EndLayers

class Conv1DClassifier(nn.Module):
    def __init__(self):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Sequential(
                nn.Conv1d(1, 32, 3),
                nn.ReLU(),
                nn.MaxPool1d(4),
                nn.Dropout(0.5))
            self.layer2 = nn.Sequential(
                nn.Conv1d(32, 64, 3),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(0.5))
                
            self.flatten = nn.Flatten()
            self.dropout = nn.Dropout()
            self.fc1 = nn.Linear(11904, 256)
            self.fc2 = nn.Linear(256, 15)
            n=3 #This is the DOO for COOL, I will need to make some way of easily editing it.
            #self.COOL = nn.Linear(256, 15*n)

            self.end = EndLayers(15,type="Soft",cutoff=Config.parameters["threshold"])
            self.batchnum = 0

        # Specify how the data passes in the neural network
        def forward(self, x: torch.Tensor):
            # x = to_device(x, device)
            x = x.float()
            x = x.unsqueeze(1)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.flatten(x)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            if self.end.type!="COOL":
                x = self.fc2(x)
            else:
                x = self.COOL(x)
            # print("in forward", F.log_softmax(x, dim=1))
            return x
            #return F.log_softmax(x, dim=1)






    def validation_step(self, batch):
        self.eval()
        savePoint(self,"test", phase=phase)
        data, labels = batch
        out = self(data)  # Generate predictions
        out = self.end.endlayer(out, labels) #              <----Here is where it is using Softmax TODO: make this be able to run all of the versions and save the outputs.
        #out = self.end.endlayer(out, labels, type="Open")
        #out = self.end.endlayer(out, labels, type="Energy")
        
        # Y_pred = out
        # Y_test = labels
        # print("y-test from validation",Y_test)
        # print("y-pred from validation", Y_pred)
        unknowns = out[:,15].mean()
        out = to_device(out,device)
        loss = F.cross_entropy(out, labels)  # Calculate loss
        plots.write_batch_to_file(loss,self.batchnum,self.end.type,"test")
        self.batchnum += 1
        acc = accuracy(out, labels)  # Calculate accuracy
        self.train()
        print("validation accuracy: ",acc)
        return {'val_loss': loss.detach(), 'val_acc': acc, "val_avgUnknown":unknowns}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        batch_unkn = self.end.Save_score
        self.end.Save_score = []
        epoch_unkn = torch.stack(batch_unkn).mean()  # Combine Unknowns
        
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item(),"val_avgUnknown":epoch_unkn.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['train_loss'],
                                                                                        result['val_loss'],
                                                                                        result['val_acc']))
                                                                                    
class MetricGeneration(model):

    def evaluate(model, validationset):
        model.batchnum = 0
        outputs = [model.validation_step(to_device(batch,device)) for batch in validationset] ### reverted bac
        return model.validation_epoch_end(outputs)

    def accuracy(outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        print("out from accuracy", preds)
        print("labels from accuracy", labels)
        y_pred.append(preds.tolist()[:])
        Y_test.append(labels.tolist()[:])
        # preds = torch.tensor(preds)
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))
        # def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.Adam):
        history = []
        optimizer = opt_func(model.parameters(), lr)
        torch.cuda.empty_cache()
        if epochs > 0:
            for epoch in range(epochs):
                # Training Phase
                model.train()
                train_losses = []
                num = 0
                for batch in train_loader:
                    # batch = to_device(batch,device)
                    batch = DeviceDataLoader(batch, device)
                    loss = model.training_step(batch)
                    
                    plots.write_batch_to_file(loss,num,model.end.type,"train")
                    train_losses.append(loss.detach())
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    num += 1
                    
                # Validation phase
                savePoint(model, f"Saves", epoch, phase)
                result = evaluate(model, val_loader)
                result['train_loss'] = torch.stack(train_losses).mean().item()
                result["epoch"] = epoch
                model.epoch_end(epoch, result)
                print("result", result)
                
                history.append(result)
        else:
            # Validation phase
            loadPoint(model, "Saves")
            result = evaluate(model, val_loader)
            result['train_loss'] = -1
            model.epoch_end(0, result)
            print("result", result)
            history.append(result)
        return history