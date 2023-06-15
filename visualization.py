import os 
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Circle 
from matplotlib.colors import LinearSegmentedColormap
from hyperbolic_clustering.hyperbolic_cluster_metrics import get_hyperbolic_cluster_metrics, \
                    plot_between_cluster_metrics, get_cortices_to_hyperbolic_cluster_radii, \
                    get_cortices_to_hyperbolic_cluster_cohesion, get_cortices_to_hyperbolic_cluster_radii_left_right, \
                    get_cortices_to_avg_hyperbolic_cluster_radii, get_depths_to_avg_hyperbolic_cluster_radii, \
                    fit_clusters, get_depths_to_hyperbolic_cluster_radii 
os.environ['DATAPATH'] = os.path.join(os.getcwd(), 'data')
os.environ['LOG_DIR'] = os.path.join(os.getcwd(), 'logs')
import matplotlib.colors as mcolors
from utils.data_utils import TREE_DEPTH


NUM_ROIS = 360
CORTEX_TO_ABBREVIATION = {
    'Primary_Visual': 'PV',
    'MT+_Complex_and_Neighboring_Visual_Areas': 'MT+CNVS',
    'Dorsal_Stream_Visual': 'DSV',
    'Early_Visual': 'EV',
    'Ventral_Stream_Visual': 'VSV',
    'Somatosensory_and_Motor': 'SM',
    'Premotor': 'Pre',
    'Posterior_Cingulate': 'PC',
    'Early_Auditory': 'EA',
    'Temporo-Parieto-Occipital_Junction': 'TPOJ',
    'Dorsolateral_Prefrontal': 'DP',
    'Superior_Parietal': 'SP',
    'Paracentral_Lobular_and_Mid_Cingulate': 'PLMC',
    'Anterior_Cingulate_and_Medial_Prefrontal': 'ACMP',
    'Orbital_and_Polar_Frontal': 'OPF',
    'Inferior_Frontal': 'IF',
    'Posterior_Opercular': 'PO',
    'Insular_and_Frontal_Opercular': 'IFO',
    'Auditory_Association': 'AA',
    'Inferior_Parietal': 'IP',
    'Medial_Temporal': 'MT',
    'Lateral_Temporal': 'LT'
}
COLORS = ['#ff0000', '#00ff00', '#0000ff', '#ffff00', '#ff00ff', 
            '#00ffff', '#800000', '#008000', '#000080', '#808000', 
            '#800080', '#008080', '#ff8080', '#80ff80', '#8080ff', 
            '#ffff80', '#ff80ff', '#80ffff', '#c00000', '#00c000', 
            '#0000c0', '#c0c000', '#c000c0']
    
def viz_metrics(log_path, use_save=False, prefix= ""):
    def collect_and_plot_curvatures(log_path, use_save, prefix):
        curvatures = []
        with open(log_path) as f:
            for line in f.readlines():
                if "INFO:root:Model" in line:
                    words = line.split()
                    curvatures.append(float(words[3]))    
        if not curvatures: return
        plot_values(curvatures, "Model Curvature", use_save, prefix)
    def collect_loss_roc_ap_for_train_val_test_from_log(log_path):
        log_path
        train_losses = []
        train_rocs = []
        train_aps = []
        val_losses = []
        val_rocs = []
        val_aps = []
        test_losses = []
        test_rocs = []
        test_aps = []
        test_epochs = []
        with open(log_path) as f:
            for line in f.readlines():

                if "INFO:root:Epoch:" in line:
                    words = line.split()
                    if "train_loss" in line:
                        train_losses.append(float(words[6]))
                        train_rocs.append(float(words[8]))
                        train_aps.append(float(words[10]))
                    if "val_loss" in line:
                        val_losses.append(float(words[3]))
                        val_rocs.append(float(words[5]))
                        val_aps.append(float(words[7]))
                if "INFO:root:Test Epoch" in line:
                    words = line.split()
                    test_losses.append(float(words[4]))
                    test_rocs.append(float(words[6]))
                    test_aps.append(float(words[8]))
                if "INFO:root:Model Improved;" in line:
                    words = line.split()
                    test_epochs.append(int(words[5]))
                    test_losses.append(float(words[7]))
                    test_rocs.append(float(words[9]))
                    test_aps.append(float(words[11]))
                if "Last Epoch:" in line:
                    words = line.split()
                    test_epochs.append(int(words[2]))
        return train_losses, train_rocs, train_aps, val_losses, val_rocs, val_aps, test_losses, test_rocs, test_aps, test_epochs
    
    metrics = collect_loss_roc_ap_for_train_val_test_from_log(log_path)
    train_losses, train_rocs, train_aps, val_losses, val_rocs, val_aps, \
        test_losses, test_rocs, test_aps, test_epochs = metrics
    def plot_values(values, title, use_save, prefix=""): 
        fig, ax = plt.subplots()
        epochs = [5 * i for i in range(len(values))] # Model Curvature actually is reported every epoch
        plt.title(title)
        plt.xlabel("Epoch")
        plt.plot(epochs, values)
        if use_save: plt.savefig(f"{prefix}_{title}.png" if prefix else f"{title}.png")
    def plot_values_with_epochs(values, epochs, title, use_save, prefix=""): 
        fig, ax = plt.subplots()
        plt.title(title)
        plt.xlabel("Epoch")
        plt.xticks(rotation=45)
        plt.plot(epochs, values)
        if use_save: plt.savefig(f"{prefix}_{title}.png" if prefix else f"{title}.png")
    
    fig, axes = plt.subplots(3, 3, figsize=(20, 10))
    
    datas = [train_losses,
                val_losses,
                test_losses,
                train_aps,
                val_aps,
                test_aps,
                train_rocs,
                val_rocs,
                test_rocs]
    titles = ["Train_Loss",
            "Validation_Loss",
            "Test_Loss",
            "Train_Average_Precision",
            "Validation_Average_Precision",
            "Test_Average_Precision",
            "Train_ROC",
            "Validation_ROC",
            "Test_ROC"]
    
    for i, ax in enumerate(axes.flat):
        
        data = datas[i]
        title = titles[i]
        # epochs = [5 * i for i in range(len(data))] if i % 3 != 2 else test_epochs
        epochs = [5 * i for i in range(len(data))]
        # Plot the data in the current subplot
        ax.plot(epochs, data)
        ax.set_title(title)

    # Adjust spacing between subplots
    fig.tight_layout()

    print(f"Test_Loss: {test_losses[-1]}")
    print(f"Test_ROC: {test_rocs[-1]}")
    print(f"Test_Average_Precision {test_aps[-1]}")

def viz_metrics_multiple(log_path, use_save=False, prefix= ""):
    def collect_and_plot_curvatures(log_path, use_save, prefix):
        curvatures = []
        with open(log_path) as f:
            for line in f.readlines():
                if "INFO:root:Model" in line:
                    words = line.split()
                    curvatures.append(float(words[3]))    
        if not curvatures: return
        plot_values(curvatures, "Model Curvature", use_save, prefix)
    def collect_loss_roc_ap_for_train_val_test_from_log_multiple(log_path):
        log_path
        train_losses = []
        train_rocs = []
        train_aps = []
        val_losses = []
        val_rocs = []
        val_aps = []
        test_losses = []
        test_rocs = []
        test_aps = []
        test_epochs = []
        with open(log_path) as f:
            for line in f.readlines():

                if "INFO:root:Epoch:" in line:
                    words = line.split()
                    if "train_loss" in line:
                        train_losses.append(float(words[6]))
                        train_rocs.append(float(words[8]))
                        train_aps.append(float(words[10]))
                    if "val_loss" in line:
                        val_losses.append(float(words[3]))
                        val_rocs.append(float(words[5]))
                        val_aps.append(float(words[7]))
                if "INFO:root:Val Epoch" in line:
                    words = line.split()
                    # val_losses.append(float(words[12][ : -1]))
                    # val_rocs.append(float(words[10][ : -1]))
                    # val_aps.append(float(words[9][ : -4]))
                    val_losses.append(float(words[13][ : -1]))
                    val_rocs.append(float(words[11][ : -1]))
                    val_aps.append(float(words[9][ : -1]))
                if "INFO:root:Test Epoch" in line:
                    words = line.split()
                    test_losses.append(float(words[13][ : -1]))
                    test_rocs.append(float(words[11][ : -1]))
                    test_aps.append(float(words[9][ : -1]))
                if "INFO:root:Model Improved;" in line:
                    words = line.split()
                    test_epochs.append(int(words[5]))
                    test_losses.append(float(words[7]))
                    test_rocs.append(float(words[9]))
                    test_aps.append(float(words[11]))
                if "Last Epoch:" in line:
                    words = line.split()
                    test_epochs.append(int(words[2]))
        return train_losses, train_rocs, train_aps, val_losses, val_rocs, val_aps, test_losses, test_rocs, test_aps, test_epochs
    
    metrics = collect_loss_roc_ap_for_train_val_test_from_log_multiple(log_path)
    train_losses, train_rocs, train_aps, val_losses, val_rocs, val_aps, \
        test_losses, test_rocs, test_aps, test_epochs = metrics
    def plot_values(values, title, use_save, prefix=""): 
        fig, ax = plt.subplots()
        epochs = [5 * i for i in range(len(values))] # Model Curvature actually is reported every epoch
        plt.title(title)
        plt.xlabel("Epoch")
        plt.plot(epochs, values)
        if use_save: plt.savefig(f"{prefix}_{title}.png" if prefix else f"{title}.png")
    def plot_values_with_epochs(values, epochs, title, use_save, prefix=""): 
        fig, ax = plt.subplots()
        plt.title(title)
        plt.xlabel("Epoch")
        plt.xticks(rotation=45)
        plt.plot(epochs, values)
        if use_save: plt.savefig(f"{prefix}_{title}.png" if prefix else f"{title}.png")
    
    fig, axes = plt.subplots(3, 2, figsize=(20, 10))
    
    datas = [
                val_losses,
                test_losses,
            
                val_aps,
                test_aps,
            
                val_rocs,
                test_rocs]
    titles = [
            "Validation_Loss",
            "Test_Loss",
            
            "Validation_Average_Precision",
            "Test_Average_Precision",
            
            "Validation_ROC",
            "Test_ROC"]
    
    for i, ax in enumerate(axes.flat):
        
        data = datas[i]
        title = titles[i]
        epochs = [index for index in range(len(data))] # if i % 3 != 2 else test_epochs
        
        # Plot the data in the current subplot
        ax.plot(epochs, data)
        ax.set_title(title)

    # Adjust spacing between subplots
    fig.tight_layout()
    print(f"Test_Loss: {test_losses[-1]}")
    print(f"Test_ROC: {test_rocs[-1]}")
    print(f"Test_Average_Precision {test_aps[-1]}")

def to_poincare(x, c):
    K = 1. / c
    sqrtK = K ** 0.5
    d = x.size(-1) - 1
#     return sqrtK * x.narrow(-1, 1, d) / (x[:, 0:1] + sqrtK)
    return sqrtK * x.narrow(-1, 1, d) / (x[0] + sqrtK)

def scale_two_dimensional_embeddings_to_poincare_disk(poincare_embeddings):
    eps = 1e-2
    radii = [torch.sqrt(poincare_embedding[0] ** 2 + poincare_embedding[1] ** 2) 
            for poincare_embedding in poincare_embeddings]
    max_radius = np.max(radii)
    for poincare_embedding in poincare_embeddings:
        poincare_embedding[0] /= (max_radius + eps)
        poincare_embedding[1] /= (max_radius + eps)
    return poincare_embeddings

def get_adjacency_matrix_for_binary_tree(depth):
        adj_matrix = np.array([[0.0 for _ in range(2 ** depth - 1)] for _ in range(2 ** depth - 1)])
        
        for node_index in range(1, (2 ** depth - 1) // 2 + 1):
            adj_matrix[node_index - 1][2 * node_index - 1] = 1.0
            adj_matrix[node_index - 1][2 * node_index] = 1.0
            adj_matrix[2 * node_index - 1][node_index - 1] = 1.0
            adj_matrix[2 * node_index][node_index - 1] = 1.0
            
        return adj_matrix
    
def get_adjacency_matrix_for_binary_cyclic_tree(depth):
    adj_matrix = np.array([[0.0 for _ in range(2 ** depth - 1)] for _ in range(2 ** depth - 1)])

    for node_index in range(1, (2 ** depth - 1) // 2 + 1):
        adj_matrix[node_index - 1][2 * node_index - 1] = 1.0
        adj_matrix[node_index - 1][2 * node_index] = 1.0
        adj_matrix[2 * node_index - 1][node_index - 1] = 1.0
        adj_matrix[2 * node_index][node_index - 1] = 1.0
    for node_index in range(len(adj_matrix)):
        if (node_index + 1) % 2 == 0: 
            adj_matrix[node_index][node_index + 1] = 1.0
            adj_matrix[node_index + 1][node_index] = 1.0
    return adj_matrix
    
def viz_bin_tree(depth, is_cyclic=False):
    if not is_cyclic: adj_matrix = get_adjacency_matrix_for_binary_tree(depth)
    else: adj_matrix = get_adjacency_matrix_for_binary_cyclic_tree(depth)
    G = nx.from_numpy_matrix(adj_matrix)
    nx.draw(G, with_labels=True)
    plt.savefig(f"binary_tree_depth_{depth}.png" if not is_cyclic else f"binary_cyclic_tree_depth_{depth}.png")
    plt.show()

def viz_embeddings(embeddings_path, 
                reduce_dim = False, 
                is_bin_tree= False, 
                is_cyclic= False, 
                use_colors = False,
                use_indices = False,
                show_edges = False,
                use_save = True,
                prefix = ""):
    depth = TREE_DEPTH
    # if is_bin_tree: viz_bin_tree(depth, is_cyclic=is_cyclic)
    embeddings_name = embeddings_path.split('\\')[-1]
    if 'log' in embeddings_name: return 
    hyperboloid_embeddings = np.load(embeddings_path)
    if not reduce_dim:
        c = 1.
        torch_embeddings = torch.from_numpy(hyperboloid_embeddings)
        poincare_embeddings = [to_poincare(torch_embedding, c) for torch_embedding in torch_embeddings]
    else:
        from sklearn.manifold import TSNE
        projected_embeddings = TSNE(n_components=2,
                        init='random', perplexity=3).fit_transform(hyperboloid_embeddings)
        poincare_embeddings = torch.from_numpy(projected_embeddings)

    # Create a figure and axis
    fig, ax = plt.subplots()
    # Create a circle with radius 1 centered at the origin
    circ = plt.Circle((0, 0), 
                radius= 1, 
                edgecolor='black', 
                facecolor='None', 
                linewidth=3, 
                alpha=0.5)
    # Add the circle to the axis
    ax.set_aspect(0.9)
    ax.add_patch(circ)
    if is_bin_tree: ax.set_title(f"{prefix}_Embeddings")
    else: ax.set_title(f"{embeddings_name[:-4]}")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    
    if use_colors: # Coloring for Binary Tree Nodes Onlys
        color_log_indices = np.array([int(np.log2(index + 1)) for index in range(len(poincare_embeddings))])
        cmap = plt.cm.jet   
        poincare_embeddings_x = [poincare_embedding[0] for poincare_embedding in poincare_embeddings]
        poincare_embeddings_y = [poincare_embedding[1] for poincare_embedding in poincare_embeddings]
        plt.scatter(poincare_embeddings_x, poincare_embeddings_y, c=color_log_indices, cmap=cmap)
        if use_indices:
            for index, poincare_embedding in enumerate(poincare_embeddings):
                ax.annotate(index, (poincare_embedding[0], poincare_embedding[1]))

    else:
        for index, poincare_embedding in enumerate(poincare_embeddings):
            plt.scatter(poincare_embedding[0], poincare_embedding[1])

    cbar = plt.colorbar()
    cbar.set_label('Node Depth')
    
    adj_matrix = get_adjacency_matrix_for_binary_tree(TREE_DEPTH)

    embeddings_df = pd.DataFrame({'x': [poincare_embeddings[i][0] for i in range(len(poincare_embeddings))], 
                                        'y': [poincare_embeddings[i][1] for i in range(len(poincare_embeddings))],
                                        'id': [i for i in range(len(poincare_embeddings))]}
                                    )
    if show_edges:
        edge_list_0 = get_edges(adj_matrix)
        for i in range(len(edge_list_0)):
            x1 = embeddings_df.loc[(embeddings_df.id == edge_list_0[i][0]), ['x', 'y']].values[0]
            x2 = embeddings_df.loc[(embeddings_df.id == edge_list_0[i][1]), ['x', 'y']].values[0]
            _ = plt.plot([x1[0], x2[0]], [x1[1], x2[1]], '--', c='black', linewidth=1, alpha=0.5)
    if use_save: plt.savefig(f"{prefix}_embeddings.png")
    plt.show()

def get_edges(adj_matrix):
        edges = []
        for i in range(len(adj_matrix)):
            for j in range(i, len(adj_matrix)):
                if adj_matrix[i, j] > 0:
                    edges.append([i, j])
        return edges

def get_hcp_atlas_df():
    datapath = os.environ['DATAPATH']
    atlas_path = os.path.join(os.path.join(datapath, "cam_can_avg"), "HCP-MMP1_UniqueRegionList.csv")
    hcp_atlas_df = pd.read_csv(atlas_path)
    cortices = hcp_atlas_df['cortex']
    region_indices_to_cortices = {index : cortex for index, cortex in enumerate(cortices)}
    color_index = -1
    seen_cortices = set()
    region_indices_to_colors = {}

    for index in region_indices_to_cortices:
        cortex = region_indices_to_cortices[index]
        if cortex not in seen_cortices:
            color_index += 1
            region_indices_to_colors[index] = COLORS[color_index]
            seen_cortices.add(region_indices_to_cortices[index])    
        else:
            region_indices_to_colors[index] = COLORS[color_index]
        
    return hcp_atlas_df
def get_cortices_and_cortex_ids_to_cortices():
    hcp_atlas_df = get_hcp_atlas_df()
    cortices = hcp_atlas_df['cortex']
    cortex_ids = hcp_atlas_df['Cortex_ID']
    cortex_ids_to_cortices = {cortex_ids[i] : cortices[i] for i in range(NUM_ROIS)}
    
    return cortices, cortex_ids_to_cortices

def get_embeddings_df(hyperboloid_embeddings):
    hcp_atlas_df = get_hcp_atlas_df()
    
    region_indices_to_cortex_ids = {i : hcp_atlas_df['Cortex_ID'][i] for i in range(NUM_ROIS)}
    c = 1.
    torch_embeddings = torch.from_numpy(hyperboloid_embeddings)
    poincare_embeddings = [to_poincare(torch_embedding, c) for torch_embedding in torch_embeddings]
    
    embeddings_df = pd.DataFrame({'x': [poincare_embeddings[i][0] for i in range(NUM_ROIS)], 
                                        'y': [poincare_embeddings[i][1] for i in range(NUM_ROIS)],
                                        'label': [region_indices_to_cortex_ids[i] for i in range(NUM_ROIS)],
                                        'id': [i for i in range(NUM_ROIS)],
                                        'LR': hcp_atlas_df['LR']}
                                )
    return embeddings_df   

def get_tree_embeddings_df(hyperboloid_embeddings):
    c = 1.
    torch_embeddings = torch.from_numpy(hyperboloid_embeddings)
    poincare_embeddings = [to_poincare(torch_embedding, c) for torch_embedding in torch_embeddings]
    
    embeddings_df = pd.DataFrame({'x': [poincare_embeddings[i][0] for i in range(len(poincare_embeddings))], 
                                        'y': [poincare_embeddings[i][1] for i in range(len(poincare_embeddings))],
                                        'label': [int(np.log2(index + 1)) for index in range(len(poincare_embeddings))],
                                        'id': [i for i in range(len(poincare_embeddings))],
                                        }
                                )
    return embeddings_df   

def scale_embeddings_df_to_poincare_disk(embeddings_df):
    eps = 1e-2
    embeddings_df['r'] = torch.sqrt(torch.Tensor(embeddings_df['x']) ** 2 + torch.Tensor(embeddings_df['y']) ** 2)
    max_radius = np.max(embeddings_df.r)
    embeddings_df['x'] /= (max_radius + eps)
    embeddings_df['y'] /= (max_radius + eps)
    embeddings_df['r'] /= (max_radius + eps)
    return embeddings_df


def plot_embeddings(embeddings, title=None, use_scale=False, use_cluster_metrics=False, use_centroids=False):
    # Create a list of labels for the legend
    cortices, cortex_ids_to_cortices = get_cortices_and_cortex_ids_to_cortices()

    labels = list(set(cortices))
    embeddings_df = get_embeddings_df(embeddings) 
    # Create the legend
    if use_scale: embeddings_df = scale_embeddings_df_to_poincare_disk(embeddings_df)
    
    for i in embeddings_df.label.unique():
        emb_L = embeddings_df.loc[(embeddings_df.LR == "L")]
        plt.scatter(emb_L.loc[(emb_L.label == i), 'x'], 
                    emb_L.loc[(emb_L.label == i), 'y'], 
                    c = COLORS[i],
                    s = 50, 
                    marker = "v",)
                    # label = cortex_ids_to_cortices[i]) avoid repeating same labels but with differnet shape
        emb_R = embeddings_df.loc[(embeddings_df.LR == "R")]
        plt.scatter(emb_R.loc[(emb_R.label == i), 'x'], 
                    emb_R.loc[(emb_R.label == i), 'y'], 
                    c = COLORS[i], 
                    s = 50,
                    marker = "s",
                    label = cortex_ids_to_cortices[i])
    # Shrink current axis by 20%
    ax = plt.gca()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    circ = plt.Circle((0, 0), 
                    radius=1, 
                    edgecolor='black', 
                    facecolor='None', 
                    linewidth=3, 
                    alpha=0.5)
    ax.add_patch(circ)
    # plot_edges = False
    # TODO: Need to include cam_can adjacency matrix as input to make plotting edges possible  
    # if plot_edges:
    #     edge_list_0 = get_edges(adj)
    #     for i in range(len(edge_list_0)):
    #         x1 = embeddings_df.loc[(embeddings_df.id == edge_list_0[i][0]), ['x', 'y']].values[0]
    #         x2 = embeddings_df.loc[(embeddings_df.id == edge_list_0[i][1]), ['x', 'y']].values[0]
    #         _ = plt.plot([x1[0], x2[0]], [x1[1], x2[1]], '--', c='black', linewidth=1, alpha=0.25)
    if title != None:
        plt.title(title, size=16)
    plt.savefig("fhnn_embedding_for_average_276_plv.png")
    
    permuted_colors = [COLORS[i] for i in embeddings_df.label.unique()]
    if use_centroids:
        embeddings_df_left = embeddings_df[embeddings_df.LR == "L"]
        embeddings_df_right = embeddings_df[embeddings_df.LR == "R"]
        embeddings_df_left_detensored = detensorify_embeddings_df(embeddings_df_left)
        embeddings_df_right_detensored = detensorify_embeddings_df(embeddings_df_right)
        # HK MEANS CENTROIDS
        # clustering_left = plot_centroids(embeddings_df_left_detensored, is_left=True)
        # clustering_right = plot_centroids(embeddings_df_right_detensored, is_left=False)

        # embeddings_df_detensored = detensorify_embeddings_df(embeddings_df)
        # plot_hyperbolic_radii(embeddings_df_detensored, 
        #                     "Hyperbolic_Cluster_Radius_CamCan_Avg_276", 
        #                     clustering, 
        #                     permuted_colors=permuted_colors)
        # plot_hyperbolic_cohesion(embeddings_df_detensored, 
        #                         "Hyperbolic_Cluster_Cohesion_CamCan_Avg_276", 
        #                         clustering,
        #                         permuted_colors=permuted_colors)
        # HK MEANS CENTROID RADII 
        # plot_hyperbolic_radii_left_right(embeddings_df_left_detensored,
        #                                 embeddings_df_right_detensored,  
        #                                 clustering_left=clustering_left,
        #                                 clustering_right=clustering_right,
        #                                 use_diff_plot=True, 
        #                                 permuted_colors=permuted_colors)
        plot_avg_hyperbolic_radii_left_right(embeddings_df_left_detensored,
                                        embeddings_df_right_detensored,  
                                        use_diff_plot=True, 
                                        permuted_colors=permuted_colors)

def plot_tree_embeddings(embeddings, title=None, use_scale=False, use_cluster_metrics=False, use_centroids=False):
    # Create a list of labels for the legend
    embeddings_df = get_tree_embeddings_df(embeddings) 
    # Create the legend
    if use_scale: embeddings_df = scale_embeddings_df_to_poincare_disk(embeddings_df)
    
    color_log_indices = np.array([int(np.log2(index + 1)) for index in range(len(embeddings_df))])
    cmap = plt.cm.jet   
    
    plt.scatter(embeddings_df['x'], embeddings_df['y'], c=color_log_indices, cmap=cmap)
    cbar = plt.colorbar()
    cbar.set_label('Node Depth')
    # Shrink current axis by 20%
    ax = plt.gca()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.set_aspect(0.9)
    
    circ = plt.Circle((0, 0), 
                    radius=1, 
                    edgecolor='black', 
                    facecolor='None', 
                    linewidth=3, 
                    alpha=0.5)
    ax.add_patch(circ)
    if title != None:
        plt.title(title, size=16)
    plt.savefig("tree_embeddings.png")
    
    # permuted_colors = [color_log_indices[i] for i in embeddings_df.label.unique()]
    if use_centroids:
        embeddings_df_detensored = detensorify_embeddings_df(embeddings_df)
        cmap = plt.cm.jet
        color_numbers = [i for i in embeddings_df.label.unique()]
        # Normalize the array of numbers to the range [0, 1]
        normalized_numbers = (color_numbers - np.min(color_numbers)) / (np.max(color_numbers) - np.min(color_numbers))

        # Apply the colormap to the normalized array
        permuted_colors = cmap(normalized_numbers)
        clustering = plot_tree_centroids(embeddings_df_detensored, is_left=True, permuted_colors=permuted_colors)
        
        plot_hyperbolic_radii_tree(embeddings_df_detensored,
                            clustering=clustering, 
                            permuted_colors=permuted_colors)
        plot_avg_hyperbolic_radii_tree(embeddings_df_detensored,
                            clustering=clustering, 
                            permuted_colors=permuted_colors)



def detensorify_embeddings_df(embeddings_df):
    embeddings_df_detensored = embeddings_df.copy()
    embeddings_df_detensored['x'] = [x.item() for x in embeddings_df_detensored['x']]
    embeddings_df_detensored['y'] = [y.item() for y in embeddings_df_detensored['y']]
    return embeddings_df_detensored

def plot_tree_centroids(embeddings_df_detensored, clustering=None, permuted_colors=None, is_left=False) -> dict:
    # permuted_colors = np.array([int(np.log2(index + 1)) for index in range(len(embeddings_df_detensored))])
    if permuted_colors is None: permuted_colors = [COLORS[i] for i in embeddings_df_detensored.label.unique()]
    if not clustering: clustering = fit_clusters(embeddings_df_detensored, is_tree=True)
    hkmeans = clustering['model']
    if is_left: left_right_marker = "v"
    else: left_right_marker = "s"
    MARKER_SIZE = 100
    for index, centroid in enumerate(hkmeans.centroids):
        # if index == len(hkmeans.centroids) - 1: break
        plt.scatter(centroid[0], 
                    centroid[1], 
                color = permuted_colors[index],
                s = MARKER_SIZE, 
                marker = left_right_marker,
                edgecolors='black')
    return clustering


def plot_centroids(embeddings_df_detensored, clustering=None, permuted_colors=None, is_left=False) -> dict:
    permuted_colors = [COLORS[i] for i in embeddings_df_detensored.label.unique()]
    if not clustering: clustering = fit_clusters(embeddings_df_detensored)
    hkmeans = clustering['model']
    if is_left: left_right_marker = "v"
    else: left_right_marker = "s"
    MARKER_SIZE = 100
    for index, centroid in enumerate(hkmeans.centroids):
        plt.scatter(centroid[0], 
                    centroid[1], 
                c = permuted_colors[index],
                s = MARKER_SIZE, 
                marker = left_right_marker,
                edgecolors='black')
    return clustering

def plot_cluster_metrics(embeddings_df, title=None, clustering=None):
    cortices, cortex_ids_to_cortices = get_cortices_and_cortex_ids_to_cortices()
    plot_between_cluster_metrics(embeddings_df, cortices, clustering)

def plot_hyperbolic_radii(embeddings_df, title=None, clustering=None, permuted_colors=None):
    cortices, cortex_ids_to_cortices = get_cortices_and_cortex_ids_to_cortices()
    cortices_to_hyperbolic_radii = get_cortices_to_hyperbolic_cluster_radii(embeddings_df, cortex_ids_to_cortices, clustering)
    if not permuted_colors: permuted_colors = [COLORS[i] for i in embeddings_df.label.unique()]
    
    # Plotting Radii
    plt.figure(figsize=(10, 6))
    plt.bar([*map(lambda x : CORTEX_TO_ABBREVIATION[x], cortices_to_hyperbolic_radii.keys())], 
            cortices_to_hyperbolic_radii.values(), color=permuted_colors)
    plt.xlabel('Cortices')
    plt.ylabel('Hyperbolic Cluster Radius')
    plt.title('Hyperbolic Radius of Cortices')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("hyperbolic_cluster_radius.png")
    plt.show()

def plot_hyperbolic_radii_tree(embeddings_df, title=None, clustering=None, permuted_colors=None):
    depths_to_hyperbolic_radii = get_depths_to_hyperbolic_cluster_radii(embeddings_df, clustering, is_tree=True)
    if permuted_colors is None: permuted_colors = [COLORS[i] for i in embeddings_df.label.unique()]
    
    # Plotting Radii
    plt.figure(figsize=(10, 6))
    plt.bar(depths_to_hyperbolic_radii.keys(), 
            depths_to_hyperbolic_radii.values(), color=permuted_colors)
    plt.xlabel('Node Tree Depth')
    plt.ylabel('Hyperbolic Cluster Radius')
    plt.title('Hyperbolic Radius of Depth Node Clusters')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("hyperbolic_cluster_radius.png")
    plt.show()

def plot_avg_hyperbolic_radii_tree(embeddings_df, title=None, clustering=None, permuted_colors=None):
    
    depths_to_hyperbolic_radii = get_depths_to_avg_hyperbolic_cluster_radii(embeddings_df)
    if permuted_colors is None: permuted_colors = [COLORS[i] for i in embeddings_df.label.unique()]
    
    # Plotting Radii
    plt.figure(figsize=(10, 6))
    plt.bar(depths_to_hyperbolic_radii.keys(), 
            depths_to_hyperbolic_radii.values(), color=permuted_colors)
    plt.xlabel('Node Tree Depth')
    plt.ylabel('Hyperbolic Average Radius')
    plt.title('Hyperbolic Average Radius of Depth Node Clusters')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("hyperbolic_average_radius.png")
    plt.show()

def plot_avg_hyperbolic_radii_left_right(embeddings_df_left, embeddings_df_right, use_diff_plot=False, permuted_colors=None):
    cortices, cortex_ids_to_cortices = get_cortices_and_cortex_ids_to_cortices()    
    cortices_to_hyperbolic_radii_left = get_cortices_to_avg_hyperbolic_cluster_radii(embeddings_df_left, cortex_ids_to_cortices)
    cortices_to_hyperbolic_radii_right = get_cortices_to_avg_hyperbolic_cluster_radii(embeddings_df_right, cortex_ids_to_cortices)
    
    if not permuted_colors: permuted_colors = [COLORS[i] for i in embeddings_df_left.label.unique()]
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 5))
    NUM_SUBNETS = 22
    num_subnets = len(cortices_to_hyperbolic_radii_right.keys())
    axes[0].set_xticks(np.arange(num_subnets), rotation=45)
    axes[0].set_xticklabels([*map(lambda x : CORTEX_TO_ABBREVIATION[x], cortices_to_hyperbolic_radii_right.keys())], rotation=45)
    axes[0].bar([*map(lambda x : CORTEX_TO_ABBREVIATION[x], cortices_to_hyperbolic_radii_left.keys())], 
            cortices_to_hyperbolic_radii_left.values(), color=permuted_colors)
    axes[0].set_xlabel('Cortices')
    axes[0].set_ylabel('Hyperbolic Average Radius')
    axes[0].set_title('Hyperbolic Average Radius of Cortices LEFT')
    # axes[0].set_ylim([0, 3])

    axes[1].set_xticks(np.arange(num_subnets))
    axes[1].set_xticklabels([*map(lambda x : CORTEX_TO_ABBREVIATION[x], cortices_to_hyperbolic_radii_right.keys())], rotation=45)
    axes[1].bar([*map(lambda x : CORTEX_TO_ABBREVIATION[x], cortices_to_hyperbolic_radii_right.keys())], 
            cortices_to_hyperbolic_radii_right.values(), color=permuted_colors)    
    axes[1].set_xlabel('Cortices')
    axes[1].set_ylabel('Hyperbolic Average Radius')
    axes[1].set_title('Hyperbolic Average Radius of Cortices RIGHT')
    # axes[1].set_ylim([0, 3])

    # Adjust the spacing between subplots
    plt.tight_layout()
    plt.savefig("hyperbolic_average_radius_left_right.png")
    plt.show()

    if use_diff_plot:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar([*map(lambda x : CORTEX_TO_ABBREVIATION[x], cortices_to_hyperbolic_radii_left.keys())], 
                np.array([*cortices_to_hyperbolic_radii_left.values()]) - np.array([*cortices_to_hyperbolic_radii_right.values()]),
                color = permuted_colors, 
                label  = 'Left - Right Hemisphere')
        ax.set_xlabel('Cortex')
        ax.set_ylabel('Hyperbolic Average Radius Difference')
        ax.set_title("Hyperbolic Average Difference of Left Minus Right Hemisphere Cortices")
        # ax.set_ylim(-1, 1)
        plt.xticks(rotation=45)
        plt.savefig("hyperbolic_average_radius_left_right_diff.png")
        plt.show()


def plot_hyperbolic_radii_left_right(embeddings_df_left, embeddings_df_right, clustering_left=None, clustering_right=None, use_diff_plot=False, permuted_colors=None):
    cortices, cortex_ids_to_cortices = get_cortices_and_cortex_ids_to_cortices()
    # cortices_to_hyperbolic_radii_left, cortices_to_hyperbolic_radii_right = \
        # get_cortices_to_hyperbolic_cluster_radii_left_right(embeddings_df, cortex_ids_to_cortices)
    cortices_to_hyperbolic_radii_left = get_cortices_to_hyperbolic_cluster_radii(embeddings_df_left, cortex_ids_to_cortices, clustering_left)
    cortices_to_hyperbolic_radii_right = get_cortices_to_hyperbolic_cluster_radii(embeddings_df_right, cortex_ids_to_cortices, clustering_right)
    
    if not permuted_colors: permuted_colors = [COLORS[i] for i in embeddings_df_left.label.unique()]
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 5))
    NUM_SUBNETS = 22
    num_subnets = len(cortices_to_hyperbolic_radii_right.keys())
    axes[0].set_xticks(np.arange(num_subnets), rotation=45)
    axes[0].set_xticklabels([*map(lambda x : CORTEX_TO_ABBREVIATION[x], cortices_to_hyperbolic_radii_right.keys())], rotation=45)
    axes[0].bar([*map(lambda x : CORTEX_TO_ABBREVIATION[x], cortices_to_hyperbolic_radii_left.keys())], 
            cortices_to_hyperbolic_radii_left.values(), color=permuted_colors)
    axes[0].set_xlabel('Cortices')
    axes[0].set_ylabel('Hyperbolic Cluster Radius')
    axes[0].set_title('Hyperbolic Radius of Cortices LEFT')
    # axes[0].set_ylim([0, 3])

    axes[1].set_xticks(np.arange(num_subnets))
    axes[1].set_xticklabels([*map(lambda x : CORTEX_TO_ABBREVIATION[x], cortices_to_hyperbolic_radii_right.keys())], rotation=45)
    axes[1].bar([*map(lambda x : CORTEX_TO_ABBREVIATION[x], cortices_to_hyperbolic_radii_right.keys())], 
            cortices_to_hyperbolic_radii_right.values(), color=permuted_colors)    
    axes[1].set_xlabel('Cortices')
    axes[1].set_ylabel('Hyperbolic Cluster Radius')
    axes[1].set_title('Hyperbolic Radius of Cortices RIGHT')
    # axes[1].set_ylim([0, 3])

    # Adjust the spacing between subplots
    plt.tight_layout()
    plt.savefig("hyperbolic_cluster_radius_left_right.png")
    plt.show()

    if use_diff_plot:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar([*map(lambda x : CORTEX_TO_ABBREVIATION[x], cortices_to_hyperbolic_radii_left.keys())], 
                np.array([*cortices_to_hyperbolic_radii_left.values()]) - np.array([*cortices_to_hyperbolic_radii_right.values()]),
                color = permuted_colors, 
                label  = 'Left - Right Hemisphere')
        ax.set_xlabel('Cortex')
        ax.set_ylabel('Hyperbolic Cluster Radius Difference')
        ax.set_title("Hyperbolic Radius Difference of Left Minus Right Hemisphere Cortices")
        # ax.set_ylim(-1, 1)
        plt.xticks(rotation=45)
        plt.savefig("hyperbolic_cluster_radius_left_right_diff.png")
        plt.show()

def plot_hyperbolic_cohesion(embeddings_df, title=None, clustering=None, permuted_colors=None):
    cortices, cortex_ids_to_cortices = get_cortices_and_cortex_ids_to_cortices()
    cortices_to_hyperbolic_cohesions = get_cortices_to_hyperbolic_cluster_cohesion(embeddings_df, cortex_ids_to_cortices, clustering)
    if not permuted_colors: permuted_colors = [COLORS[i] for i in embeddings_df.label.unique()]
    # Plotting Radii
    plt.figure(figsize=(10, 6))
    plt.bar([*map(lambda x : CORTEX_TO_ABBREVIATION[x], cortices_to_hyperbolic_cohesions.keys())], 
            cortices_to_hyperbolic_cohesions.values(), color=permuted_colors)
    plt.xlabel('Cortices')
    plt.ylabel('Hyperbolic Cluster Cohesion')
    plt.title('Hyperbolic Cohesion of Cortices')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("hyperbolic_cluster_cohesion.png")
    plt.show()

def plot_embeddings_by_parts(embeddings, title=None, use_scale=False, use_cluster_metrics=False, use_centroids=False):
    # Create a list of labels for the legend
    cortices, cortex_ids_to_cortices = get_cortices_and_cortex_ids_to_cortices()

    labels = list(set(cortices))
    embeddings_df = get_embeddings_df(embeddings) 
    # Create the legend
    partial_labels = np.array_split(embeddings_df.label.unique(), 5)
    for split_labels in partial_labels:
        embeddings_partial_df = pd.concat([embeddings_df.loc[(embeddings_df.label == i)] for i in split_labels])

        if use_scale: embeddings_partial_df = scale_embeddings_df_to_poincare_disk(embeddings_partial_df)
        
        for i in embeddings_partial_df.label.unique():
            emb_L = embeddings_partial_df.loc[(embeddings_partial_df.LR == "L")]
            plt.scatter(emb_L.loc[(emb_L.label == i), 'x'], 
                        emb_L.loc[(emb_L.label == i), 'y'], 
                        c = COLORS[i],
                        s = 50, 
                        marker = "v",)
            emb_R = embeddings_partial_df.loc[(embeddings_partial_df.LR == "R")]
            plt.scatter(emb_R.loc[(emb_R.label == i), 'x'], 
                        emb_R.loc[(emb_R.label == i), 'y'], 
                        c = COLORS[i], 
                        s = 50,
                        marker = "s",
                        label = cortex_ids_to_cortices[i])
        # Shrink current axis by 20%
        ax = plt.gca()
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        circ = plt.Circle((0, 0), 
                        radius=1, 
                        edgecolor='black', 
                        facecolor='None', 
                        linewidth=3, 
                        alpha=0.5)
        ax.add_patch(circ)
        # plot_edges = False
        # TODO: Need to include cam_can adjacency matrix as input to make plotting edges possible  
        # if plot_edges:
        #     edge_list_0 = get_edges(adj)
        #     for i in range(len(edge_list_0)):
        #         x1 = embeddings_df.loc[(embeddings_df.id == edge_list_0[i][0]), ['x', 'y']].values[0]
        #         x2 = embeddings_df.loc[(embeddings_df.id == edge_list_0[i][1]), ['x', 'y']].values[0]
        #         _ = plt.plot([x1[0], x2[0]], [x1[1], x2[1]], '--', c='black', linewidth=1, alpha=0.25)
        if title != None:
            plt.title(title, size=16)
        plt.savefig("fhnn_embedding_for_average_276_plv.png")
        
        permuted_colors = [COLORS[i] for i in embeddings_partial_df.label.unique()]
        if use_centroids:
            embeddings_df_left = embeddings_df[embeddings_df.LR == "L"]
            embeddings_df_right = embeddings_df[embeddings_df.LR == "R"]
            embeddings_df_left_detensored = detensorify_embeddings_df(embeddings_df_left)
            embeddings_df_right_detensored = detensorify_embeddings_df(embeddings_df_right)
            plot_centroids(embeddings_df_left_detensored)
            plot_centroids(embeddings_df_right_detensored)

            embeddings_df_detensored = detensorify_embeddings_df(embeddings_df)
            # plot_hyperbolic_radii(embeddings_df_detensored, 
            #                     "Hyperbolic_Cluster_Radius_CamCan_Avg_276", 
            #                     clustering, 
            #                     permuted_colors=permuted_colors)
            # plot_hyperbolic_cohesion(embeddings_df_detensored, 
            #                         "Hyperbolic_Cluster_Cohesion_CamCan_Avg_276", 
            #                         clustering,
            #                         permuted_colors=permuted_colors)
            plot_hyperbolic_radii_left_right(embeddings_df_detensored,
                                            "Hyperbolic_Cluster_Radius_LR_CamCan_Avg_276",
                                            use_diff_plot=True,
                                            permuted_colors=permuted_colors)
