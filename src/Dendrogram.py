import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import numpy as np
import itertools


CLASSLIST = {0: 'BENIGN', 1: 'Infiltration', 2: 'Bot', 3: 'PortScan', 4: 'DDoS', 5: 'FTP-Patator', 6: 'SSH-Patator', 7: 'DoS slowloris', 8: 'DoS Slowhttptest', 9: 'DoS Hulk', 10: 'DoS GoldenEye', 11: 'Heartbleed', 12: 'Web Attack – Brute Force', 13: 'Web Attack – XSS', 14:'Web Attack – Sql Injection'}
LISTCLASS = {CLASSLIST[x]:x for x in range(15)}
PROTOCOLS = {"udp":0,"tcp":1}
def classConvert(x):
    return LISTCLASS[x]
def protocalConvert(x):
    return PROTOCOLS[x]


def plot_confusion_matrix(cm:np.ndarray, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    row_sum = cm.sum(0)
    col_sum = cm.sum(1)
    col_sum = np.append(col_sum,row_sum.sum())
    col_sum = np.expand_dims(col_sum,1)
    classes.append("Sum")
    groups = list(range(len(classes)-1))
    groups.append("Sum")

    cm = cm/row_sum

    cm = np.vstack([cm,row_sum])
    cm = np.hstack([cm,col_sum])

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, groups, rotation=90)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis].clip(0.000001,1)
        # cm = "{:.2f}".format(float)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print("cm", cm.shape)
    #cm = cm.astype("int")

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        # print("cm ij ",str(round(cm[i, j], 2)))
        plt.text(j, i, str(round(cm[i, j], 2)),
                horizontalalignment="center",verticalalignment="center_baseline",
                color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Dendrogram label')
    plt.savefig("Saves/confusion_matrix.png")

def plot_confusion_matrix_2(cm:np.ndarray, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    groups = list(range(len(classes)))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, groups, rotation=90)
    plt.yticks(tick_marks, classes)


    print("cm", cm.shape)
    #cm = cm.astype("int")

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        # print("cm ij ",str(round(cm[i, j], 2)))
        plt.text(j, i, str(round(cm[i, j], 2)),
                horizontalalignment="center",verticalalignment="center_baseline",
                color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Dendrogram label')



#From https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import confusion_matrix, rand_score, adjusted_rand_score

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


print("\n\nLoading data")
data = pd.read_csv("datasets/Payload_data_CICIDS2017.csv",converters={"protocol":protocalConvert,"label":classConvert})
X = data
X = X.sample(n=40000)

label = X["label"].to_numpy()
X2 = X.iloc[:,:len(X)-1].to_numpy()

# setting distance_threshold=0 ensures we compute the full tree.
model = AgglomerativeClustering(distance_threshold=None, n_clusters=15, compute_distances=True)

print("Fitting model")
model = model.fit(X2)
lst = model.labels_

class_list = []
for i in range(15):
    class_list.append(CLASSLIST[i])

cm = confusion_matrix(label,lst,labels=list(range(15)))

#Metric information https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering
print(f"Random cluster evaluation: {rand_score(label,lst)}")
print(f"Adjusted random cluster evaluation: {adjusted_rand_score(label,lst)}")


bincount = np.bincount(lst,minlength=15)
plt.figure(1)
plot_confusion_matrix_2(cm,class_list)

plt.figure(2)
plot_confusion_matrix(cm,class_list)

plt.figure(3)
plt.title("Hierarchical Clustering Dendrogram")
# plot the top three levels of the dendrogram
plot_dendrogram(model, truncate_mode="level", p=5)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.savefig("Saves/Dendrogram.png")
plt.show()



#links = sp.cluster.hierarchy.linkage(data)

#print("done")