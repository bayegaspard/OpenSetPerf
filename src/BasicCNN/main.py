
import torch
import torch.utils.data
import torch.optim as optim
import glob


#four lines from https://xxx-cook-book.gitbooks.io/python-cook-book/content/Import/import-from-parent-folder.html
import os
import sys
root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)

#this seems really messy
from HelperFunctions.LoadPackets import NetworkDataset
from HelperFunctions.Evaluation import correctValCounter
from HelperFunctions.ModelLoader import Network


#pick a device
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")

torch.manual_seed(0)
BATCH = 1000
CUTOFF = 0
NAME = "BasicCNN"

path_to_dataset = "datasets" #put the absolute path to your dataset , type "pwd" within your dataset folder from your teminal to know this path.

def getListOfCSV(path):
    return glob.glob(path+"/*.csv")

data_total = NetworkDataset(getListOfCSV(path_to_dataset))

CLASSES = len(data_total.classes)

data_train, data_test = torch.utils.data.random_split(data_total, [len(data_total)-1000,1000])

training =  torch.utils.data.DataLoader(dataset=data_train, batch_size=BATCH, shuffle=True)
testing = torch.utils.data.DataLoader(dataset=data_test, batch_size=BATCH, shuffle=False)

model = Network(CLASSES).to(device)


evaluative = correctValCounter(CLASSES)

epochs = 50
criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

#single batch training just to see that it can fit to something
# single_batch = next(iter(training))
# X, y = single_batch

#test = testing._get_iterator()

#print(torch.argmax(model(test.next()[0]), dim=1))

#plt.imshow(test.next()[0][0][0])
#plt.show()

#making the items that appear less frequently have a higher penalty so that they are not missed.
# magnification = torch.tensor([1.0000e+00, 2.8636e+02, 3.8547e+02, 3.9218e+02, 4.1337e+02, 9.8373e+00,
#         2.2084e+02, 2.0665e+05, 1.5084e+03, 3.4863e+03, 1.0824e+05, 6.3142e+04,
#         1.1562e+03, 1.4303e+01, 1.7754e+01])[:CLASSES]

for e in range(epochs):
    lost_amount = 0

    for batch, (X, y) in enumerate(training):
    #The below is for single batch training
    #for batch in range(1):
        X = X.to(device)
        y = y.to(device)

        _, output = model(X)
        lost_points = criterion(output, y)
        optimizer.zero_grad()
        lost_points.backward()

        #printing paramiters to check if they are moving
        #for para in model.parameters():
            #print(para.grad)

        optimizer.step()

        lost_amount += lost_points.item()

    


    with torch.no_grad():
        model.eval()
        for batch,(X,y) in enumerate(testing):
            X = X.to(device)
            y = y.to("cpu")

            _,output = model(X)
            output = output.to("cpu")

            evaluative.evalN(output,y)

            #Show what it got correct
            #if e % 25 == 0 and matches.sum().item() > 0:
                #print("Batch: "+str(batch)+" got "+str(matches.sum().item())+" Correct")
                #print(output_val)
                #print(model(X)[torch.argmax(torch.argmax(matches, dim=1))])
                #plt.imshow(X[torch.argmax(torch.argmax(matches, dim=1))][0])
                #print(torch.argmax(matches, dim=1))
                #plt.show()

        print(f"-----------------------------Epoc: {e}-----------------------------")
        print(f"lost: {100*lost_amount/len(data_train)}")
        evaluative.PrintEval()
        evaluative.PrintUnknownEval()
        print(f"There were: {evaluative.count_by_class}")
        evaluative.zero()
        

        model.train()
