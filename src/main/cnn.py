from torch import nn
import torch
from torch.nn import functional as F
import os
import plots

### user defined functions
import Config
from EndLayer import EndLayers
import GPU


class ModdedParallel(nn.DataParallel):
    # From https://github.com/pytorch/pytorch/issues/16885#issuecomment-551779897
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class Conv1DClassifier(nn.Module):
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
        n = 3  # This is the DOO for COOL, I will need to make some way of easily editing it.
        # self.COOL = nn.Linear(256, 15*n)

        self.end = EndLayers(15, type="Energy", cutoff=Config.parameters["threshold"][0])
        self.batchnum = 0
        self.device = GPU.get_default_device()
        self.store = GPU.to_device(torch.tensor([]), self.device), GPU.to_device(torch.tensor([]), self.device), GPU.to_device(torch.tensor([]), self.device)

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
        if self.end.type != "COOL":
            x = self.fc2(x)
        else:
            x = self.COOL(x)
        # print("in forward", F.log_softmax(x, dim=1))
        return x
        # return F.log_softmax(x, dim=1)


class FullyConnected:
    def __init__(self):
        pass


class AttackTrainingClassification(Conv1DClassifier):
    def __init__(self):
        super().__init__()
        # self.model = model
        # self.batch = batch
        # self.to_device = to_device
        # self.device = device
        # self.Y_Pred = Y_Pred
        # self.Y_test = Y_test

    def training_step(self, batch):
        data, labels = batch
        labels = labels[:,0]    #Select the data we want not the metadata
        out = self(data)  # Generate predictions
        if self.end.type == "COOL":
            labels = self.end.COOL_Label_Mod(labels)
            out = torch.split(out.unsqueeze(dim=1), 15, dim=2)
            out = torch.cat(out, dim=1)

        # out = DeviceDataLoader(out, device)
        loss = F.cross_entropy(out, labels)  # Calculate loss
        torch.cuda.empty_cache()
        # print("loss from training step ... ", loss)
        return loss

    def evaluate(self, validationset):
        self.batchnum = 0
        outputs = [self.validation_step(batch) for batch in validationset]  ### reverted bac
        return self.validation_epoch_end(outputs)

    def accuracy(self, outputs, labels):
        preds = torch.argmax(outputs, dim=1)
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
        history = []
        optimizer = opt_func(self.parameters(), lr)
        # torch.cuda.empty_cache()
        if epochs > 0:
            for epoch in range(epochs):
                # Training Phase
                self.train()
                train_losses = []
                num = 0
                for batch in train_loader:
                    # batch = to_device(batch,device)
                    # batch = DeviceDataLoader(batch, device)
                    loss = self.training_step(batch)

                    plots.write_batch_to_file(loss, num, self.end.type, "train")
                    train_losses.append(loss.detach())
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    num += 1

                # Validation phase
                self.savePoint(f"Saves", epoch, Config.helper_variables["phase"])
                result = self.evaluate(val_loader)
                result['train_loss'] = torch.stack(train_losses).mean().item()
                result["epoch"] = epoch
                self.epoch_end(epoch, result)
                print("result", result)

                history.append(result)
        else:
            # Validation phase
            self.loadPoint("Saves")
            result = self.evaluate(val_loader)
            result['train_loss'] = -1
            self.epoch_end(0, result)
            print("result", result)
            history.append(result)
        return history

    def validation_step(self, batch):
        self.eval()
        self.savePoint("test", phase=Config.helper_variables["phase"])
        data, labels_extended = batch
        labels = labels_extended[:,0]
        out = self(data)  # Generate predictions
        out = self.end.endlayer(out,
                                labels)  # <----Here is where it is using Softmax TODO: make this be able to run all of the versions and save the outputs.
        # out = self.end.endlayer(out, labels, type="Open")
        # out = self.end.endlayer(out, labels, type="Energy")

        # Y_pred = out
        # Y_test = labels
        # print("y-test from validation",Y_test)
        # print("y-pred from validation", Y_pred)
        unknowns = out[:, 15].mean()
        out = GPU.to_device(out, self.device)
        test = torch.argmax(out, dim=1)
        loss = F.cross_entropy(out, labels)  # Calculate loss
        plots.write_batch_to_file(loss, self.batchnum, self.end.type, "test")
        self.batchnum += 1
        acc = self.accuracy(out, labels_extended)  # Calculate accuracy
        print("validation accuracy: ", acc)
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

    def savePoint(net: Conv1DClassifier, path: str, epoch=0, phase=None):
        if not os.path.exists(path):
            os.mkdir(path)
        torch.save(net.state_dict(), path + f"/Epoch{epoch:03d}.pth")
        if phase is not None:
            file = open("Saves/phase", "w")
            file.write(str(phase))
            file.close()

    def loadPoint(net: Conv1DClassifier, path: str):
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




