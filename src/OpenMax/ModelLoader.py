import torch.nn as nn
import torch

class Network(nn.Module):
    def __init__(self,CLASSES):
        super().__init__()

        self.cv1 = nn.Conv2d(1,4,5)
        self.pool1 = nn.MaxPool2d((2,2),(2,2))
        self.RL = nn.LeakyReLU()
        self.cv2 = nn.Conv2d(4,16,5)
        self.pool2 = nn.MaxPool2d(2,2)
        self.cv3 = nn.Conv2d(16,64,5)
        self.pool3 = nn.MaxPool2d(2,2)

        self.fc1 = nn.Linear(5184, 1028)
        self.fc2 = nn.Linear(1028,240)
        self.fc3 = nn.Linear(240,CLASSES)

        self.fc = nn.Linear(5184,CLASSES)
        self.dropout = nn.Dropout(0.3)

        self.soft = nn.Softmax(dim=1)
        self.double()

    def forward(self, input):
        X = input.double()
        X = self.pool1(self.RL(self.cv1(self.dropout(X))))
        X = self.pool2(self.RL(self.cv2(self.dropout(X))))
        X = self.pool3(self.RL(self.cv3(self.dropout(X))))

        X = torch.flatten(X, start_dim=1)

        X = self.RL(self.fc1(self.dropout(X)))
        X = self.RL(self.fc2(self.dropout(X)))
        
        #the OpenMax implementation wants two values here but does not use the first one
        return 0, self.fc3(X)