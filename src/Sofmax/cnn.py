import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import Dataload
from torch.utils.data import DataLoader
import plots




# get the data and create a test set and train set
train = Dataload.Dataset("/home/designa/Desktop/OSR_updated/Payload_data_CICIDS2017",use=[0,1,4,5,6,7,8,9,10,11,12])
train, test = torch.utils.data.random_split(train, [len(train) - 20,20])  # randomly takes 4000 lines to use as a testing dataset

batch_size = 10

trainset = DataLoader(test, batch_size, shuffle=True,
                      pin_memory=False)  # for faster processing enable pin memory to true and num_workers=4
validationset = DataLoader(test, batch_size, shuffle=True, pin_memory=False)
testset = DataLoader(test, batch_size, shuffle=True, pin_memory=False)

print(len(train))
print(len(test))

print(next(iter(testset)))
test_features, testset_labels = next(iter(testset))
print(f"Feature batch shape: {test_features.size()}")
print(f"Labels batch shape: {testset_labels.size()}")
img = test_features[0].squeeze()
label = testset_labels[:]
print("label sss", label)

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
        loss = F.cross_entropy(out, labels)  # Calculate loss
        return loss

    def validation_step(self, batch):
        data, labels = batch
        out = self(data)  # Generate predictions
        # Y_pred = out
        # Y_test = labels
        # print("y-test from validation",Y_test)
        # print("y-pred from validation", Y_pred)
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)  # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

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
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


# def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.Adam):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_losses = []
        for batch in train_loader:
            # batch = to_device(batch,device)
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
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

    # Specify how the data passes in the neural network
    def forward(self, x: torch.Tensor):
        x = to_device(x, device)
        x = x.float()
        x = x.unsqueeze(1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(self.dropout(x))
        # print("in forward", F.log_softmax(x, dim=1))
        return F.log_softmax(x, dim=1)


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

model = Net()
model.to(device)

num_epochs = 3
opt_func = torch.optim.Adam
lr = 0.001

history_final = []
history_final += fit(num_epochs, lr, model, train_loader, val_loader, opt_func)


print("all history", history_final)

print("y test outside",Y_test)
print("y pred outside",y_pred)

plots.plot_all_losses(history_final)
plots.plot_losses(history_final)
plots.plot_accuracies(history_final)

y_test, y_pred = plots.convert_to_1d(Y_test,y_pred)
plots.plot_confusion_matrix(y_test,y_pred)

recall = plots.recall_score(y_test, y_pred)
precision = plots.precision_score(y_test, y_pred)
f1 = plots.f1_score(y_test, y_pred)

print("F-Score : ", f1*100)
print("Precision : " ,precision*100)
print("Recall : ", recall*100)

