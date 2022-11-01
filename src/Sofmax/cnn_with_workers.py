
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import Dataload
from torch.utils.data import DataLoader
import plots
from EndLayer import EndLayers
import os
from sklearn.metrics import (precision_score, recall_score)
import warnings



def main():
    #warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"
    os.environ['TORCH'] = torch.__version__
    print(torch.__version__)
    unknownVals = [2,3,13,14]
    knownVals = list(range(15))
    for un in unknownVals:
        knownVals.remove(un)

    # get the data and create a test set and train set
    train = Dataload.Dataset(r"C:\Users\bgaspard\Desktop\OpenSetPerf\datasets\Payload_data_CICIDS2017",use=knownVals)
    train, test = torch.utils.data.random_split(train, [len(train) - len(train)//4,len(train)//4])  # randomly takes 4000 lines to use as a testing dataset
    unknowns = Dataload.Dataset(r"C:\Users\bgaspard\Desktop\OpenSetPerf\datasets\Payload_data_CICIDS2017",use=unknownVals,unknownData=True)
    test = torch.utils.data.ConcatDataset([test,unknowns])
    #test = unknowns

    attemptLoad = True
    batch_size = 1000

    trainset = DataLoader(train, batch_size, num_workers=6,shuffle=True,
                          pin_memory=True)  # for faster processing enable pin memory to true and num_workers=4
    validationset = DataLoader(test, batch_size, shuffle=True, num_workers=6,pin_memory=True)
    testset = DataLoader(test, batch_size, shuffle=True, num_workers=6, pin_memory=True)


    print(len(train))
    print(len(test))

    # print(next(iter(testset)))
    # test_features, testset_labels = next(iter(testset))
    # print(f"Feature batch shape: {test_features.size()}")
    # print(f"Labels batch shape: {testset_labels.size()}")
    # img = test_features[0].squeeze()
    # label = testset_labels[:]
    # print("label sss", label)

    Y_test = []
    y_pred =[]




    def accuracy(outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        print("out from accuracy", preds)
        print("labels from accuracy", labels)
        y_pred.append(preds.tolist()[:])
        Y_test.append(labels.tolist()[:])
        # preds = torch.tensor(preds)
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))


    class AttackClassification(nn.Module):
        def training_step(self, batch):
            data, labels = batch
            # data = to_device(data, device)
            # labels = to_device(labels, device)
            out = self(data)  # Generate predictions
            if self.end.type == "COOL":
                labels = self.end.COOL_Label_Mod(labels)
                out = torch.split(out.unsqueeze(dim=1),15, dim=2)
                out = torch.cat(out,dim=1)
                labels = to_device(labels, device)
            loss = F.cross_entropy(out, labels)  # Calculate loss
            return loss

        def validation_step(self, batch):
            self.eval()
            savePoint(self,"test")
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


            loss = F.cross_entropy(out, labels)  # Calculate loss
            acc = accuracy(out, labels)  # Calculate accuracy
            self.train()
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


    def get_default_device():
        """Pick GPU if available, else CPU"""
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')


    device = get_default_device()


    def to_device(data, device):
        """Move tensor(s) to chosen device"""
        if isinstance(data, (list, tuple)):
            return [to_device(x, device) for x in data]
        return data.to(device, non_blocking=True)


    class DeviceDataLoader():
        """Wrap a dataloader to move data to a device"""

        def __init__(self, dl, device):
            self.dl = dl
            self.device = device

        def __iter__(self):
            """Yield a batch of data after moving it to device"""
            for b in self.dl:
                yield to_device(b, self.device)

        def __len__(self):
            """Number of batches"""
            return len(self.dl)


    train_loader = DeviceDataLoader(trainset, device)
    val_loader = DeviceDataLoader(validationset, device)
    test_loader = DeviceDataLoader(testset, device)


    def evaluate(model, val_loader):
        outputs = [model.validation_step(batch) for batch in DeviceDataLoader(validationset, device)]
        return model.validation_epoch_end(outputs)


    # def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.Adam):
        history = []
        optimizer = opt_func(model.parameters(), lr)
        if epochs > 0:
            for epoch in range(epochs):
                # Training Phase
                model.train()
                train_losses = []
                for batch in trainset:
                    # batch = to_device(batch,device)
                    batch = DeviceDataLoader(batch, device)
                    loss = model.training_step(batch)
                    train_losses.append(loss)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                # Validation phase
                savePoint(model, f"Saves", epoch)
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


    # Building the neural network with 28x28 as input and 10 as output passing via series of relus
    class Net(AttackClassification):

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
            self.COOL = nn.Linear(256, 15*n)

            self.end = EndLayers(15,type="Soft")
            

        # Specify how the data passes in the neural network
        def forward(self, x: torch.Tensor):
            x = to_device(x, device)
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
            return F.log_softmax(x, dim=1)

    def savePoint(net:AttackClassification, path:str, epoch=0):
        if not os.path.exists(path):
            os.mkdir(path)
        torch.save(net.state_dict(),path+f"/Epoch{epoch:03d}.pth")
    def loadPoint(net:AttackClassification, path:str):
        if not os.path.exists(path):
            os.mkdir(path)
        i = 999
        while i >= 0:
            if os.path.exists(path+f"/Epoch{i:03d}.pth"):
                net.load_state_dict(torch.load(path+f"/Epoch{i:03d}.pth"))
                print(f"Loaded  model /Epoch{i:03d}.pth")
                i = -1
            i = i-1
        if i != -2:
            print("No model to load found.")

# initialize the neural network
# net = Net().float()
# # print(Net)
# # calculating gradients using Adam optimizer
# optimizer = optim.Adam(net.parameters(), lr=0.001)

# lr tells the optimizer to optimize(learn) towards less errors using a certain number of steps.
# Adjust lr to an optimal value because too small will get stock even way before it moves to the local minimum and large steps will get jumping without reaching the local minimum.
# A smart learning rate is required

# EPOCHS = 1
# print(device)
# h = 10

# for epoch in range(EPOCHS):
#     for data in trainset:
#         #data is a batch of featuresets and labels
#         X,y = data # data has labels and data, X is data and y is labels
#         net.zero_grad()
#         output = net(X) # -1 is to tell pytorch it can have any value, 28*28 is the size of our images
#         loss = F.nll_loss(output, y) #if you have one hot vector [0,1,0,0] use root mean square error. if it is integer , use nll
#         loss.backward()
#         optimizer.step()
#         h=+1
#         if h==10:
#             break
#     print(loss)
#         # print(X[0])
#         # print(y[0])
#         # break
# correct = 0
# total = 0

# with torch.no_grad():
#     for data in trainset:
#         X, y = data
#         output = net(X)
#         for idx , i in enumerate(output):
#             if torch.argmax(i) == y[idx]:
#                 correct+=1
#             total+=1


# print("Accuracy: ", round(correct/total,3))

# plt.imshow(X[3].view(47,32))
# plt.show()
    
    num_epochs = 5
    opt_func = torch.optim.Adam
    lr = 0.001

    for x in ["COOL","Soft","Open","Energy"]:
        model = Net()
        model.to(device)
        model.end.type=x
        Y_test = []
        y_pred =[]
        history_finaltyped = []
        history_finaltyped += fit(num_epochs, lr, model, train_loader, val_loader, opt_func)
        y_test, y_pred = plots.convert_to_1d(Y_test,y_pred)
        recall = recall_score(y_test,y_pred,average='weighted',zero_division=0)
        precision = precision_score(y_test,y_pred,average='weighted',zero_division=0)
        f1 = 2 * (precision * recall) / (precision + recall)
        # auprc = average_precision_score(y_test, y_pred, average='samples')
        score_list = [recall,precision,f1]
        plots.write_hist_to_file(history_finaltyped,num_epochs,x)
        plots.write_scores_to_file(score_list,num_epochs,x)  
    if attemptLoad:
        loadPoint(model,"Saves")
    
    # model = Net()
    # model = model.to(device)
    
    Y_test = []
    y_pred =[]
    history_final = []
    history_final += fit(num_epochs, lr, model, train_loader, val_loader, opt_func)


    print("all history", history_final)

    print("y test outside",Y_test)
    print("y pred outside",y_pred)

    plots.plot_all_losses(history_final)
    plots.plot_losses(history_final)
    plots.plot_accuracies(history_final)

    y_test, y_pred = plots.convert_to_1d(Y_test,y_pred)
    #plots.plot_confusion_matrix(y_test,y_pred)

    import itertools
    import numpy as np
    from sklearn.metrics import confusion_matrix,average_precision_score
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)
    class_names = Dataload.get_class_names(knownVals)
    class_names.append("unknown")
    plots.plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=False,
                          title='Confusion matrix')
    plt.show()

    recall = recall_score(y_test,y_pred,average='weighted',zero_division=0)
    precision = precision_score(y_test,y_pred,average='weighted',zero_division=0)
    f1 = 2 * (precision * recall) / (precision + recall)
    # auprc = average_precision_score(y_test, y_pred, average='samples')
    score_list = [recall,precision,f1]
    plots.write_hist_to_file(history_final,num_epochs,model.end.type)
    plots.write_scores_to_file(score_list,num_epochs,model.end.type)
    print("F-Score : ", f1*100)
    print("Precision : " ,precision*100)
    print("Recall : ", recall*100)
    # print("AUPRC : ", auprc * 100)

    model.end.prepWeibull(train_loader,device,model)

    


if __name__ == '__main__':
    main()




