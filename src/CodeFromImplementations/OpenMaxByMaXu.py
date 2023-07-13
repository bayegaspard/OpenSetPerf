#From https://github.com/ma-xu/Open-Set-Recognition/blob/26a8a1cca199f4e23df98abca6893e3eef3307da/OSR/OpenMax/openmax.py
#this code is not mine, it was created by someone called ma-xu, and it reproduces the results of the paper "Towards Open Set Deep Networks" https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Bendale_Towards_Open_Set_CVPR_2016_paper.pdf
#Note: is this enough recognition that this is not my code?
#Note: I added np.unique() to line 44 because it was having trouble generating weibull models when the model outputs a lot of identical responses. NOTE: I am not sure if this is allowed or not but it gets it to work better?
#Note: I added an error message when it fails.

import numpy as np
import scipy.spatial.distance as spd
import torch


#three lines from https://xxx-cook-book.gitbooks.io/python-cook-book/content/Import/import-from-parent-folder.html
import os
import sys
root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(root_folder,"CodeFromImplementations"))

import src.main.Config as Config
import src.main.helperFunctions as helperFunctions
from src.main.helperFunctions import NoExamples



import libmr
def calc_distance(query_score, mcv, eu_weight, distance_type='eucos'):
    if distance_type == 'eucos':
        query_distance = spd.euclidean(mcv, query_score) * eu_weight + \
            spd.cosine(mcv, query_score)
    elif distance_type == 'euclidean':
        query_distance = spd.euclidean(mcv, query_score)
    elif distance_type == 'cosine':
        query_distance = spd.cosine(mcv, query_score)
    else:
        print("distance type not known: enter either of eucos, euclidean or cosine")
    return query_distance


def fit_weibull(means, dists, categories, tailsize=20, distance_type='eucos'):
    """
    Input:
        means (C, channel, C)
        dists (N_c, channel, C) * C
    Output:
        weibull_model : Perform EVT based analysis using tails of distances and save
                        weibull model parameters for re-adjusting softmax scores
    """
    weibull_model = {}
    for mean, dist, category_name in zip(means, dists, categories):
        weibull_model[category_name] = {}
        weibull_model[category_name]['distances_{}'.format(distance_type)] = dist[distance_type]
        weibull_model[category_name]['mean_vec'] = mean
        weibull_model[category_name]['weibull_model'] = []
        for channel in range(mean.shape[0]):
            mr = libmr.MR()
            tailtofit = np.sort(np.unique(dist[distance_type][channel, :]))[-tailsize:]
            mr.fit_high(tailtofit, len(tailtofit))
            weibull_model[category_name]['weibull_model'].append(mr)

    return weibull_model


def query_weibull(category_name, weibull_model, distance_type='eucos'):
    return [weibull_model[category_name]['mean_vec'],
            weibull_model[category_name]['distances_{}'.format(distance_type)],
            weibull_model[category_name]['weibull_model']]


def compute_openmax_prob(scores, scores_u):
    prob_scores, prob_unknowns = [], []
    for s, su in zip(scores, scores_u):
        channel_scores = np.exp(s)
        channel_unknown = np.exp(np.sum(su))

        total_denom = np.sum(channel_scores) + channel_unknown
        prob_scores.append(channel_scores / total_denom)
        prob_unknowns.append(channel_unknown / total_denom)

    # Take channel mean
    scores = np.mean(prob_scores, axis=0)
    unknowns = np.mean(prob_unknowns, axis=0)
    modified_scores = scores.tolist() + [unknowns]
    return modified_scores


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def openmax(weibull_model, categories, input_score, eu_weight, alpha=10, distance_type='eucos'):
    """Re-calibrate scores via OpenMax layer
    Output:
        openmax probability and softmax probability
    """
    nb_classes = len(categories)

    ranked_list = input_score.argsort().ravel()[::-1][:alpha]
    alpha_weights = [((alpha + 1) - i) / float(alpha) for i in range(1, alpha + 1)]
    omega = np.zeros(nb_classes)
    # print(f"Omega = {omega}")
    # print(f"Ranked List = {ranked_list}")
    # print(f"Alpha Weights = {alpha_weights}")
    omega[ranked_list] = alpha_weights

    scores, scores_u = [], []
    for channel, input_score_channel in enumerate(input_score):
        score_channel, score_channel_u = [], []
        for c, category_name in enumerate(categories):
            mav, dist, model = query_weibull(category_name, weibull_model, distance_type)
            channel_dist = calc_distance(input_score_channel, mav[channel], eu_weight, distance_type)
            wscore = model[channel].w_score(channel_dist)
            modified_score = input_score_channel[c] * (1 - wscore * omega[c])
            score_channel.append(modified_score)
            score_channel_u.append(input_score_channel[c] - modified_score)

        scores.append(score_channel)
        scores_u.append(score_channel_u)

    scores = np.asarray(scores)
    scores_u = np.asarray(scores_u)

    openmax_prob = np.array(compute_openmax_prob(scores, scores_u))
    softmax_prob = softmax(np.array(input_score.ravel()))
    return openmax_prob, softmax_prob


def compute_channel_distances(mavs, features, eu_weight=0.5):
    """
    Input:
        mavs (channel, C)
        features: (N, channel, C)
    Output:
        channel_distances: dict of distance distribution from MAV for each channel.
    """
    eucos_dists, eu_dists, cos_dists = [], [], []
    for channel, mcv in enumerate(mavs):  # Compute channel specific distances
        eu_dists.append([spd.euclidean(mcv, feat[channel]) for feat in features])
        cos_dists.append([spd.cosine(mcv, feat[channel]) for feat in features])
        eucos_dists.append([spd.euclidean(mcv, feat[channel]) * eu_weight +
                            spd.cosine(mcv, feat[channel]) for feat in features])

    return {'eucos': np.array(eucos_dists), 'cosine': np.array(cos_dists), 'euclidean': np.array(eu_dists)}


def compute_train_score_and_mavs_and_dists(train_class_num,trainloader,device,net):
    net.eval()#LINE ADDED
    scores = [[] for _ in range(train_class_num)]
    #print("train class in open",train_class_num)
    #print("scores from open",scores)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets[:,1].to(device)
            
            # this must cause error for cifar
            #_, outputs = net(inputs)                   <--this was from the orignial OpenMax implementation
            outputs = net(inputs)   



            #print("output from open",outputs)                    #<-this was a replacement
            for score, t in zip(outputs, targets):
                score, t = helperFunctions.renameClassesLabeled(score,t)   
                #print ("t at time of crash",t)
                     #This line has been added so that all of the knowns are sequental
                # print(f"torch.argmax(score) is {torch.argmax(score)}, t is {t}")
                if torch.argmax(score) == t:
                    scores[t].append(score.unsqueeze(dim=0).unsqueeze(dim=0))
    #LINES ADDED HERE
    a = 0
    for x in scores:
        if len(x) == 0:
            print(f"Class{a} has no examples")
            raise NoExamples()
        a+=1
    scores = [torch.cat(x).cpu().numpy() for x in scores]  # (N_c, 1, C) * C
    mavs = np.array([np.mean(x, axis=0) for x in scores])  # (C, 1, C)
    dists = [compute_channel_distances(mcv, score) for mcv, score in zip(mavs, scores)]
    return scores, mavs, dists


#This was not a function before!
def openmaxevaluation(scores,args,dict,weibull=None):
    trainloader = dict["loader"]
    device = dict["device"]
    net = dict["net"]
    scores = helperFunctions.renameClasses(scores)
    scores = [scores]

    #The following is from lines 186 to 207 from https://github.com/ma-xu/Open-Set-Recognition/blob/master/OSR/OpenMax/cifar100.py
    # Get the prdict results.
    scores = torch.cat(scores,dim=0).cpu().numpy()
    #labels = torch.cat(labels,dim=0).cpu().numpy()
    scores = np.array(scores)[:, np.newaxis, :]
    #labels = np.array(labels)


    #ADDED: SAVE THE WEIBULL MODEL
    if weibull == None:
        # Fit the weibull distribution from training data.
        print("Fittting Weibull distribution...")
        _, mavs, dists = compute_train_score_and_mavs_and_dists(args.train_class_num, trainloader, device, net)
        categories = list(range(0, args.train_class_num))
        weibull_model = fit_weibull(mavs, dists, categories, args.weibull_tail, "euclidean")
    else:
        #THIS SECTION WAS ADDED
        categories = list(range(0, args.train_class_num))
        weibull_model = weibull

    pred_softmax, pred_softmax_threshold, pred_openmax = [], [], []
    score_softmax, score_openmax = [], []
    for score in scores:
        so, ss = openmax(weibull_model, categories, score,
                        0.5, args.weibull_alpha, "euclidean")  # openmax_prob, softmax_prob
        pred_softmax.append(np.argmax(ss))
        pred_softmax_threshold.append(np.argmax(ss) if np.max(ss) >= args.weibull_threshold else args.train_class_num)
        pred_openmax.append(np.argmax(so) if np.max(so) >= args.weibull_threshold else args.train_class_num)
        score_softmax.append(ss)
        score_openmax.append(so)
    #end copied code
    return score_openmax

#This was also not specifically a function before
def weibull_fittting(args,dict):
    trainloader = dict["loader"]
    device = dict["device"]
    net = dict["net"]
    # Fit the weibull distribution from training data.
    print("Fittting Weibull distribution...")
    _, mavs, dists = compute_train_score_and_mavs_and_dists(args.train_class_num, trainloader, device, net)
    categories = list(range(0, args.train_class_num))
    return fit_weibull(mavs, dists, categories, args.weibull_tail, "euclidean")



