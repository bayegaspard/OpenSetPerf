#From https://github.com/leishu02/EMNLP2017_DOC/blob/2b870170ab20cdc9d6f0ec85631a9ddd199a2b18/DOC_emnlp17.py#L225
import numpy as np
import torch
import Config

#ADDED FOR USE
def renameClasses(modelOut:torch.Tensor, labels:torch.Tensor):
    labels = labels.clone()
    lastval = -1
    label = list(range(15))
    newout = []
    for val in Config.helper_variables["unknowns_clss"]:
        label.remove(val)
        if val > lastval+1:
            if modelOut.dim() == 2:
                newout.append(modelOut[:,lastval+1:val])
            else:
                newout.append(modelOut[lastval+1:val])
        lastval = val
    if modelOut.dim() == 2:
        newout.append(modelOut[:,lastval+1:])
    else:
        newout.append(modelOut[lastval+1:])

    newout = torch.cat(newout, dim=-1)

    i = 0
    for l in label:
        labels[labels==l] = i
        i+=1
    return newout, labels



#my code

class modelstruct():
    def __init__(self,model):
        self.model = model
    
    def predict(self,numbers:np.ndarray):
        num = torch.tensor(numbers)
        return self.model(num).detach().numpy()

def muStandardsFromDataloader(seen,Dataloader,model):
    #labelArray = np.zeros(shape=(len(Dataloader),1))
    labelArray = None
    with torch.no_grad():
        for inputs,labels in Dataloader:
            outputs, labels = renameClasses(model(inputs),labels)
            if labelArray is None:
                outputArray = outputs.cpu().numpy()
                labelArray = labels.cpu().numpy()
            else:
                outputArray = np.vstack((outputArray,outputs.numpy()))
                labelArray = np.vstack((labelArray,labels.numpy()))

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
    
def runDOC(test_X_pred_true, mu_stds, seen):
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
        threshold = max(0.5, 1. - scale * mu_stds[max_class][1])#find threshold for the predicted class
        if max_value > threshold:
             test_y_pred.append(max_class)#predicted probability is greater than threshold, accept
        else:
            #THE NEXT LINE HAS BEEN MODIFIED
            test_y_pred.append(15)#otherwise, reject

    
    # In[]:
    return test_y_pred

def muStandards(seen, predictions, labels):
    seen_train_X_pred = predictions
    seen_train_y = labels[:,0]

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
    #THIS HAS BEEN CHANGED
    for i in range(len(predictions[0])):
        if i in seen or True: #this was modified again
            pos_mu, pos_std = fit(seen_train_X_pred[seen_train_y==i, i])
            mu_stds.append([pos_mu, pos_std])
        else:
            #I have no idea if this is what I am supposed to be doing for the unknowns 
            mu_stds.append([0, 0])

    print(mu_stds)

    return mu_stds
