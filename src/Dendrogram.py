import pandas as pd
import scipy as sp

CLASSLIST = {0: 'BENIGN', 1: 'Infiltration', 2: 'Bot', 3: 'PortScan', 4: 'DDoS', 5: 'FTP-Patator', 6: 'SSH-Patator', 7: 'DoS slowloris', 8: 'DoS Slowhttptest', 9: 'DoS Hulk', 10: 'DoS GoldenEye', 11: 'Heartbleed', 12: 'Web Attack – Brute Force', 13: 'Web Attack – XSS', 14:'Web Attack – Sql Injection'}
LISTCLASS = {CLASSLIST[x]:x for x in range(15)}
PROTOCOLS = {"udp":0,"tcp":1}
def classConvert(x):
    return LISTCLASS[x]
def protocalConvert(x):
    return PROTOCOLS[x]

#From https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html
import numpy as np

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
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

iris = load_iris()
X = iris.data

counts = pd.DataFrame(0, index=range(15),columns=range(32))
data = pd.read_csv("datasets/Payload_data_CICIDS2017.csv",converters={"protocol":protocalConvert,"label":classConvert})
for x in range(15):
    X = data
    X = X[X["label"]==14-x]
    #X = X.sample(n=100 if 100<len(X) else len(X))
    X2 = X.to_numpy()

    # setting distance_threshold=0 ensures we compute the full tree.
    model = AgglomerativeClustering(distance_threshold=None, n_clusters=32 if 32<len(X) else 1, compute_distances=True)

    model = model.fit(X2)
    lst = model.labels_
    bincount = np.bincount(lst,minlength=32)
    counts.iloc[x] = bincount
    for i in range(32):
        X3 = X[lst==i]
        X3.to_csv("datasets/Payload_data_CICIDS2017_Clustered"+f"/chunk{CLASSLIST[14-x]}-type{i:03d}.csv",index_label=False,index=False)
counts.to_csv("datasets/Payload_data_CICIDS2017_Clustered/counts.csv",index_label=False,index=False)

plt.title("Hierarchical Clustering Dendrogram")
# plot the top three levels of the dendrogram
plot_dendrogram(model, truncate_mode="level", p=5)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()
plt.savefig(fname="Dendrogram.png")



#links = sp.cluster.hierarchy.linkage(data)

#print("done")