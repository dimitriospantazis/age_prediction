from hyperbolic_clustering.hyperbolic_kmeans.hkmeans import HyperbolicKMeans
from hyperbolic_clustering.utils.utils import poincare_dist, poincare_distances
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

NUM_SUBNETS = 22

def fit_clusters(embeddings_df, num_updates=10, is_tree=False):
    emb_coords = np.array(embeddings_df[['x', 'y']])
    hkmeans = HyperbolicKMeans(n_clusters = embeddings_df.label.nunique(), is_tree=is_tree)
    hkmeans.n_samples = emb_coords.shape[0]
    hkmeans.init_centroids(radius=0.1)
    hkmeans.init_assign(embeddings_df.label.values)
    for i in range(num_updates):
        hkmeans.update_centroids(emb_coords)
    return {'model': hkmeans, 'embedding': embeddings_df}

def distance_between(A, B, metric='average'):
    # methods for intercluster distances
    distances = []
    for a in A:
        distances += [poincare_dist(a, b) for b in B]
    if metric == 'average':
        return np.mean(distances)
    elif metric == 'max':
        return np.max(distances)
    elif metric == 'min':
        return np.min(distances)
    else:
        print('Invalid metric specified')
        return
        
def distance_within(A, centroid, metric='variance'):
    # methods to compute cohesion within cluster
    centroid_distances = np.array([poincare_dist(x, centroid) for x in A])
    pairwise_distances = poincare_distances(A)
    if metric == 'variance':
        return np.mean(centroid_distances ** 2)
    elif metric == 'diameter':
        return np.max(pairwise_distances)
    elif metric == 'pairwise':
        return np.sum(pairwise_distances) / len(A)
    else:
        print('Invalid metric specified')
        return

def cluster_features(embeddings_df, centroids, wc_metric='pairwise', bc_metric='average'):
    emb_coords = np.array(embeddings_df[['x', 'y']])
    within_cluster = []
    between_cluster = []
    for i in range(len(np.unique(embeddings_df.label))):
        within_cluster.append(distance_within(emb_coords[embeddings_df.label == i], centroid=centroids[i], metric=wc_metric))
        for j in range(i + 1, len(np.unique(embeddings_df.label))):
            between_cluster.append(distance_between(emb_coords[embeddings_df.label == i], emb_coords[embeddings_df.label == j], metric=bc_metric))
    return {'within': np.array(within_cluster), 'between': np.array(between_cluster)}


def get_cortices_to_hyperbolic_cluster_radii_left_right(embeddings_df, cortex_ids_to_cortices):
    embeddings_df_left = embeddings_df[embeddings_df.LR == "L"]
    embeddings_df_right = embeddings_df[embeddings_df.LR == "R"]
    cortices_to_hyperbolic_radii_left = get_cortices_to_hyperbolic_cluster_radii(embeddings_df_left, cortex_ids_to_cortices)
    cortices_to_hyperbolic_radii_right = get_cortices_to_hyperbolic_cluster_radii(embeddings_df_right, cortex_ids_to_cortices)
    return cortices_to_hyperbolic_radii_left, cortices_to_hyperbolic_radii_right

def get_cortices_to_hyperbolic_cluster_radii(embeddings_df, cortex_ids_to_cortices, clustering=None):
    if not clustering: clustering = fit_clusters(embeddings_df)
    embeddings_df = clustering['embedding']
    hkmeans = clustering['model']
    origin = np.array([0, 0])
    cortices_to_hyperbolic_radii = dict()
    for index, centroid in enumerate(hkmeans.centroids):
        hyperbolic_radius = poincare_dist(centroid, origin)
        cortices_to_hyperbolic_radii[cortex_ids_to_cortices[embeddings_df.label.unique()[index]]] = hyperbolic_radius
    return cortices_to_hyperbolic_radii

def get_cortices_to_avg_hyperbolic_cluster_radii(embeddings_df, cortex_ids_to_cortices):
    origin = np.array([0, 0])
    cortices_to_hyperbolic_radii = dict()
    avg_radius = 0
    for cortex_id in embeddings_df.label.unique():
        embeddings_from_id = embeddings_df[embeddings_df.label == cortex_id][['x', 'y']].values[0]
        avg_radius = 0
        for embedding_coords in embeddings_from_id:
            hyperbolic_radius = poincare_dist(embedding_coords, origin)
            avg_radius += hyperbolic_radius
        avg_radius /= len(embeddings_from_id)
        cortices_to_hyperbolic_radii[cortex_ids_to_cortices[cortex_id]] = avg_radius
    return cortices_to_hyperbolic_radii
    

def get_depths_to_hyperbolic_cluster_radii(embeddings_df, cortex_ids_to_cortices, clustering=None, is_tree=False):
    if not clustering: clustering = fit_clusters(embeddings_df, is_tree=is_tree)
    embeddings_df = clustering['embedding']
    hkmeans = clustering['model']
    origin = np.array([0, 0])
    depths_to_hyperbolic_radii = dict()
    for index, centroid in enumerate(hkmeans.centroids):
        hyperbolic_radius = poincare_dist(centroid, origin)
        depths_to_hyperbolic_radii[index] = hyperbolic_radius
    return depths_to_hyperbolic_radii

def get_depths_to_avg_hyperbolic_cluster_radii(embeddings_df):
    origin = np.array([0, 0])
    depths_to_avg_hyperbolic_radii = dict()
    avg_radius = 0
    for depth in embeddings_df.label.unique():
        embeddings_from_depth = embeddings_df[embeddings_df.label == depth][['x', 'y']].values[0]
        avg_radius = 0
        for embedding_coords in embeddings_from_depth:
            hyperbolic_radius = poincare_dist(embedding_coords, origin)
            avg_radius += hyperbolic_radius
        avg_radius /= len(embeddings_from_depth)
        depths_to_avg_hyperbolic_radii[depth] = avg_radius
    return depths_to_avg_hyperbolic_radii

def get_cortices_to_hyperbolic_cluster_cohesion(embeddings_df, cortex_ids_to_cortices, clustering=None):
    if not clustering: clustering = fit_clusters(embeddings_df)
    embeddings_df = clustering['embedding']
    hkmeans = clustering['model']
    cohesions = cluster_features(embeddings_df, hkmeans.centroids)['within']
    cortices_to_hyperbolic_cohesions = dict()
    for index, cohesion in enumerate(cohesions):
        cortices_to_hyperbolic_cohesions[cortex_ids_to_cortices[embeddings_df.label.unique()[index]]] = cohesion
    return cortices_to_hyperbolic_cohesions

def get_hyperbolic_cluster_metrics(embeddings_df, cortices, clustering=None):
    hyp_cluster_metrics = dict()
    within_cluster_features = []

    # NOTE: Can be adapted into for loop to iterate over all subject embeddings...
    if not clustering: clustering = fit_clusters(embeddings_df)
    embeddings_df = clustering['embedding']
    hkmeans = clustering['model']
    within_cluster_features.append(cluster_features(embeddings_df, hkmeans.centroids)['within'])

    eval_cluster = pd.DataFrame(np.ravel(np.array(within_cluster_features)), columns=['within_cluster_cohesion'])
    # TODO: Can use to give Age Labels!
    graph_labels = np.array([0])
    eval_cluster['label'] = np.repeat(graph_labels, NUM_SUBNETS)
    eval_cluster['label'] = eval_cluster.label.apply(lambda x: ['healthy', 'diagnosed'][int(x)])
    eval_cluster['network'] = np.tile(cortices.unique(), 1)
    sns.set(style="whitegrid")
    ax = sns.boxplot(x="network", y="within_cluster_cohesion", hue="label", data=eval_cluster, palette="pastel")
    plt.title('Intra-Cluster Analysis: Brain Networks', size=16)
    plt.show(); 

def plot_between_cluster_metrics(embeddings_df, cortices, clustering=None):
    if not clustering: clustering = fit_clusters(embeddings_df)
    hkmeans = clustering['model']
    embeddings_df = clustering['embedding']
    plt.figure(figsize=(15,5))
    plt.subplot(121)
    # NUM_SUBNETS = cortices.nunique()
    subnet_dist = np.zeros((NUM_SUBNETS, NUM_SUBNETS))
    for i in range(NUM_SUBNETS):
        for j in range(i + 1, NUM_SUBNETS):
            subnet_dist[i, j] = cluster_features(embeddings_df, hkmeans.centroids)['between'][i + j - 1]
            subnet_dist[j, i] = subnet_dist[i, j]

    sns.heatmap(subnet_dist, annot=True);
    ax = plt.gca()
    
    
    ax.set_xticklabels(cortices.unique())
    ax.set_yticklabels(cortices.unique())
    plt.title('Cam_Can_Average_276')

    # sns.heatmap(subnet_dist, annot=True);
    # ax = plt.gca()
    # ax.set_xticklabels(['MOT', 'VIS', 'DMN', 'ATN'])
    # ax.set_yticklabels(['MOT', 'VIS', 'DMN', 'ATN'])
    # plt.title('Schizophrenia Patients')
    plt.suptitle('Hyperbolic Distances: Between Network Clusters', size=16)
    plt.show();