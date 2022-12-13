#From https://github.com/leishu02/EMNLP2017_DOC/blob/2b870170ab20cdc9d6f0ec85631a9ddd199a2b18/DOC_emnlp17.py#L225
import numpy as np

#my code
import torch
class modelstruct():
    def __init__(self,model):
        self.model = model
    
    def predict(self,numbers:np.ndarray):
        num = torch.tensor(numbers)
        return self.model(num).detach().numpy()

def muStandardsFromDataloader(seen,Dataloader,model):
    labelArray = []
    outputArray = []
    with torch.no_grad():
        for inputs,labels in Dataloader:
            outputArray.append(model(inputs).numpy())
            labelArray.append(labels.numpy())
    outputArray = np.array(outputArray).flatten()
    labelArray = np.array(labelArray).flatten()
    return muStandards(seen,outputArray,labelArray)


# def runDOC(net,batch, mu_stds):
#     model = modelstruct(net)

#     seen_test_X,seen_test_y = batch
#     seen_test_y = seen_test_y[:,1].detach()
#     seen_test_X = seen_test_X.detach()
#     unseen_test_X = seen_test_X[seen_test_y==15].numpy()
#     unseen_test_y = seen_test_y[seen_test_y==15].numpy()
#     seen_test_X = seen_test_X[seen_test_y!=15].numpy()
#     seen_test_y = seen_test_y[seen_test_y!=15].numpy()
    
def runDOC(test_X_pred, mu_stds):

    
    #End added code


    


    

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
        threshold = max(0.5, 1. - scale * mu_stds[max_class][1])#find threshold for the predicted class
        if max_value > threshold:
            test_y_pred.append(max_class)#predicted probability is greater than threshold, accept
        else:
            #THE NEXT LINE HAS BEEN MODIFIED
            test_y_pred.append(15)#otherwise, reject

    
    # In[]:
    return test_y_pred

def muStandards(seen, predictions, labels):
    seen_train_X_pred = predictions.detach()
    seen_train_y = labels[:,1].detach()

    #fit a gaussian model
    from scipy.stats import norm as dist_model
    def fit(prob_pos_X):
        prob_pos = [p for p in prob_pos_X]+[2-p for p in prob_pos_X]
        pos_mu, pos_std = dist_model.fit(prob_pos)
        return pos_mu, pos_std
    
    # In[18]:

    #calculate mu, std of each seen class
    mu_stds = []
    #THIS NEXT LINE HAS BEEN REPLACED
    #for i in range(len(seen)):
    for i in seen:
        pos_mu, pos_std = fit(seen_train_X_pred[seen_train_y==i, i])
        mu_stds.append([pos_mu, pos_std])

    print(mu_stds)

    return mu_stds
