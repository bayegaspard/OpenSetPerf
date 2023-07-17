#From https://github.com/leishu02/EMNLP2017_DOC
#   That is from the paper: https://arxiv.org/pdf/1709.08716
import numpy as np
import torch
import src.main.Config as Config
import src.main.helperFunctions as helperFunctions





#my code to link the other code to the same frame

class modelstruct():
    def __init__(self,model):
        self.model = model
    
    def predict(self,numbers:np.ndarray):
        #Just putting things in the form of tensorflow.
        num = torch.tensor(numbers)
        return self.model(num).detach().numpy()

def muStandardsFromDataloader(seen,Dataloader,model):
    """
    The rest of the code stores the input data in a dataloader, 
    the DOC code cannot read a dataloader so this transforms the data into a numpy array.
    """
    #labelArray = np.zeros(shape=(len(Dataloader),1))
    labelArray = None
    with torch.no_grad():
        for inputs,labels in Dataloader:
            outputs, labels = helperFunctions.renameClassesLabeled(torch.sigmoid(model(inputs)),labels)
            outputs = outputs.cpu()
            labels = labels.cpu()
            if labelArray is None:
                outputArray = outputs.cpu().numpy()
                labelArray = labels.cpu().numpy()
            else:
                outputArray = np.vstack((outputArray,outputs.cpu().numpy()))
                labelArray = np.vstack((labelArray,labels.cpu().numpy()))

    return muStandards(seen,outputArray,labelArray)

#End linking code

    
def runDOC(test_X_pred_true, mu_stds, seen, saved_scores=[]):
    #test_X_pred = test_X_pred_true[:,seen]
    test_X_pred = test_X_pred_true
    #Only code by Leishu
    

    # In[20]:

    #predict on test examples
    #NEXT three LINES HAVE BEEN COMMENTED OUT
    #test_X_pred = model.predict(np.concatenate([seen_test_X,unseen_test_X], axis = 0))
    #test_y_gt = np.concatenate([seen_test_y,unseen_test_y], axis = 0)
    #print(test_X_pred.shape, test_y_gt.shape)


    # In[23]:

    #get prediction based on threshold
    test_y_pred = []
    scale = 1.
    for p in test_X_pred:# loop every test prediction
        max_class = np.argmax(p)# predicted class
        max_value = np.max(p)# predicted probability
        saved_scores.append(max_value)# ADDED LINE
        threshold = max(0.5, 1. - scale * mu_stds[max_class][1])#find threshold for the predicted class
        if max_value > threshold:
             test_y_pred.append(max_class)#predicted probability is greater than threshold, accept
        else:
            #THE NEXT LINE HAS BEEN MODIFIED (to not hardcode the rejection value)
            test_y_pred.append(Config.parameters["CLASSES"][0])#otherwise, reject

    
    # In[]:
    return test_y_pred

def muStandards(seen, predictions, labels):
    seen_train_X_pred = predictions #Two lines that are getting things out of the format they are in from main.
    seen_train_y = labels[:,0] #I really hope I am doing this correctly, the paper is difficult to read.

    #Start code by Leishu

    #fit a gaussian model
    from scipy.stats import norm as dist_model
    def fit(prob_pos_X):
        prob_pos = [p for p in prob_pos_X]+[2-p for p in prob_pos_X]
        pos_mu, pos_std = dist_model.fit(prob_pos)
        return pos_mu, pos_std
    
    # In[18]:

    #calculate mu, std of each seen class
    mu_stds = []
    for i in range(len(predictions[0])):
        pos_mu, pos_std = fit(seen_train_X_pred[seen_train_y==i, i])
        mu_stds.append([pos_mu, pos_std])

    #print(mu_stds)

    return mu_stds
