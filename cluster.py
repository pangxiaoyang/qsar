"""
Usage:
    DeepCoy.py [options]

Options:
    -h --help                Show this screen
    --path                   the path of 3018 CDK4 inhibitors
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from scipy.cluster import hierarchy
from sklearn import metrics
import matplotlib.pyplot as plt
from descriptors_calculation import cal_ecfp4,smiles_to_mol
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.cluster import AgglomerativeClustering
import umap
from docopt import docopt

def evaluation(data,pred,method):
    # Metrics
    s1 = silhouette_score(data, pred, metric='euclidean')
    c1 = calinski_harabasz_score(data, pred)
    d1 = davies_bouldin_score(data, pred)
    df_metrics = {'method':method,'Silhouette':s1, 'CH score':c1, 'DB score':d1}
    return df_metrics

def cluster_kmeans(data,n_clusters = 11):
    cluster = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, max_iter=300, random_state=100).fit(data)
    pred = cluster.predict(data)
    method = 'k_means'
    return pred,method

def cluster_Agglomerative(data,n_clusters = 11):
    clustering = AgglomerativeClustering(linkage='ward', n_clusters=n_clusters)
    clustering.fit(data)
    pred = clustering.labels_
    method = 'HC'
    return pred,method

def DR_tsne(data):
    tsne = TSNE(n_components=2, perplexity=50, learning_rate=1000, n_iter=1000, early_exaggeration = 12, init='pca')
    tsne.fit(data)
    emb = tsne.embedding_
    return emb

def DR_umap(data):
    emb = umap.UMAP(n_neighbors=100, min_dist=0.0,n_components=2, metric='jaccard',random_state=42).fit_transform(data)
    return emb


def cluster_all(data,n_clusters = 11):
    result = []
    pred_all = []
    
    pred,closest = cluster_kmeans(data,n_clusters)
    result.append(evaluation(data,pred))
    pred_all.append(pred)
    
    pred = cluster_Agglomerative(data,n_clusters)
    result.append(evaluation(data,pred))
    pred_all.append(pred)
    
    emb_tsne = DR_tsne(data)
    
    pred,closest = cluster_kmeans(emb_tsne,n_clusters)
    result.append(evaluation(emb_tsne,pred))
    pred_all.append(pred)
    
    pred = cluster_Agglomerative(emb_tsne,n_clusters)
    result.append(evaluation(emb_tsne,pred))
    pred_all.append(pred)
    
    emb_umap = DR_umap(data)
    
    pred,closest = cluster_kmeans(emb_umap,n_clusters)
    result.append(evaluation(emb_umap,pred))
    pred_all.append(pred)
    
    pred = cluster_Agglomerative(emb_umap,n_clusters)
    result.append(evaluation(emb_umap,pred))
    pred_all.append(pred)

    return emb_tsne,emb_umap,result,pred_all

def draw_plot(pred,emb,n_clusters,plot_path):
    color = ['yellow', 'red', 'blue', 'orange', 'fuchsia', 'black', 'cyan', 'lime', 'darkgoldenrod', 'green', 'slategray','darkred']
    marker = ['o', '^', 'x', '8', 's', '*', 'p', '2', 'h', 'x', 'D','|']
    plt.clf()
    for i in range(n_clusters):
        plt.plot(emb[:, 0][np.where(pred==i)], emb[:, 1][np.where(pred==i)], marker= marker[i], c=color[i], markerfacecolor='none',alpha=0.4, linestyle='', label='subset %d'%(i+1))
    plt.legend(bbox_to_anchor=(1,1), ncol=1)
    plt.savefig(plot_path, dpi=600, bbox_inches="tight")
    plt.show()






if __name__ == "__main__":
    args = docopt(__doc__)
    path = args.get('--path')
    ecfp = cal_ecfp4(smiles_to_mol(path))
    emb_tsne,emb_umap,result,pred_all = cluster_all(ecfp,n_clusters = 11)
    a = ['None','None','TSNE','TSNE','UMAP','UMAP']
    result = pd.DataFrame(result)
    result['DR_method'] = a
    print(result)
