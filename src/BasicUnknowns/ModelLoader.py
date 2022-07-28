import torch.nn as nn
import torch

class Network(nn.Module):
    def __init__(self, CLASSES):
        super().__init__()

        self.RL = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.fc1 = nn.Linear(78, 100)
        self.fc2 = nn.Linear(100,50)
        self.fc3 = nn.Linear(50,CLASSES)

        self.fc = nn.Linear(78,CLASSES)
        self.dropout = nn.Dropout(0.3)

        self.soft = nn.Softmax(dim=1)
        self.double()
        self.openMax = True

    def forward(self, input):
        X = input.double()
        X = torch.flatten(X, start_dim=1)

        X = self.tanh(self.fc1(self.dropout(X)))
        X = self.RL(self.fc2(self.dropout(X)))

        #the OpenMax implementation wants two values here but does not use the first one
        if self.openMax:
            return 0, self.fc3(X)
        else:
            return self.fc3(X)
