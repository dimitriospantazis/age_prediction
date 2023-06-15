"""Data utils functions for pre-processing and data loading."""
import os
import pickle as pkl
import sys
from typing import List

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch

THICK_INDEX = 0
MYELIN_INDEX = 1
TREE_DEPTH = 10
THRESHOLD = 0.07
NUM_ROIS = 360
THRESHOLD_NEW = 0.23
TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT = .6, .2, .2

TOTAL_SBJS = 592
NUM_SBJS = TOTAL_SBJS - 5
# Remember that 5 subjects excluded due to filtering out Numerically Unstable PLV Matrices

def load_data(args, datapath):
    if args.dataset == 'cam_can_multiple' or args.dataset == 'binary_cyclic_tree_multiple':
        data = load_data_lp(args.dataset, args.use_feats, datapath, args.use_super_node)

        for graph_data_dicts in data.values():
            for graph_data in graph_data_dicts:
                adj = graph_data['adj_train']
                
                adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = mask_edges(
                        adj, args.val_prop, args.test_prop, args.split_seed
                )
                graph_data['adj_train'] = adj_train
                graph_data['train_edges'], graph_data['train_edges_false'] = train_edges, train_edges_false
                graph_data['val_edges'], graph_data['val_edges_false'] = val_edges, val_edges_false
                graph_data['test_edges'], graph_data['test_edges_false'] = test_edges, test_edges_false
                
                graph_data['edges'] = train_edges 
                graph_data['edges_false'] = train_edges_false
                graph_data['adj_train_norm'], graph_data['features'] = process(
                    graph_data['adj_train'], graph_data['features'], args.normalize_adj, args.normalize_feats
                )
        return data

    if args.task == 'nc':
        data = load_data_nc(args.dataset, args.use_feats, datapath, args.split_seed)
    
    else:
        data = load_data_lp(args.dataset, args.use_feats, datapath, args.use_super_node)
        
        adj = data['adj_train']
        if args.task == 'lp':
            adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = mask_edges(
                    adj, args.val_prop, args.test_prop, args.split_seed
            )
            data['adj_train'] = adj_train
            data['train_edges'], data['train_edges_false'] = train_edges, train_edges_false
            data['val_edges'], data['val_edges_false'] = val_edges, val_edges_false
            data['test_edges'], data['test_edges_false'] = test_edges, test_edges_false
    data['adj_train_norm'], data['features'] = process(
            data['adj_train'], data['features'], args.normalize_adj, args.normalize_feats
    )
    if args.dataset == 'airport':
        data['features'] = augment(data['adj_train'], data['features'])
    return data


# ############### FEATURES PROCESSING ####################################


def process(adj, features, normalize_adj, normalize_feats):
    if sp.isspmatrix(features):
        features = np.array(features.todense())
    if normalize_feats:
        features = normalize(features)
    features = torch.Tensor(features)
    if normalize_adj:
        adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj, features


def normalize(mx):
    """Row-normalize sparse matrix."""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def min_max_normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))
def z_score_normalize(x):
    return (x - np.mean(x)) / np.std(x)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.Tensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def augment(adj, features, normalize_feats=True):
    deg = np.squeeze(np.sum(adj, axis=0).astype(int))
    deg[deg > 5] = 5
    deg_onehot = torch.tensor(np.eye(6)[deg], dtype=torch.float).squeeze()
    const_f = torch.ones(features.size(0), 1)
    features = torch.cat((features, deg_onehot, const_f), dim=1)
    return features


# ############### DATA SPLITS #####################################################


def mask_edges(adj, val_prop, test_prop, seed):
    
    np.random.seed(seed)  # get tp edges
    x, y = sp.triu(adj).nonzero()
    pos_edges = np.array(list(zip(x, y)))
    np.random.shuffle(pos_edges)
    # get tn edges
    x, y = sp.triu(sp.csr_matrix(1. - adj.toarray())).nonzero()
    neg_edges = np.array(list(zip(x, y)))
    np.random.shuffle(neg_edges)

    m_pos = len(pos_edges)
    n_val = int(m_pos * val_prop)
    n_test = int(m_pos * test_prop)
    val_edges, test_edges, train_edges = pos_edges[:n_val], pos_edges[n_val:n_test + n_val], pos_edges[n_test + n_val:]
    val_edges_false, test_edges_false = neg_edges[:n_val], neg_edges[n_val:n_test + n_val]
    train_edges_false = np.concatenate([neg_edges, val_edges, test_edges], axis=0)
    adj_train = sp.csr_matrix((np.ones(train_edges.shape[0]), (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # Training with 100 % edges visible since will do graph iteration and train, val, test splits will come from graphs
    train_edges = pos_edges
    train_edges_false = neg_edges

    return adj_train, torch.LongTensor(train_edges), torch.LongTensor(train_edges_false), torch.LongTensor(val_edges), \
           torch.LongTensor(val_edges_false), torch.LongTensor(test_edges), torch.LongTensor(
            test_edges_false)  


def split_data(labels, val_prop, test_prop, seed):
    np.random.seed(seed)
    nb_nodes = labels.shape[0]
    all_idx = np.arange(nb_nodes)
    pos_idx = labels.nonzero()[0]
    neg_idx = (1. - labels).nonzero()[0]
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)
    pos_idx = pos_idx.tolist()
    neg_idx = neg_idx.tolist()
    nb_pos_neg = min(len(pos_idx), len(neg_idx))
    nb_val = round(val_prop * nb_pos_neg)
    nb_test = round(test_prop * nb_pos_neg)
    idx_val_pos, idx_test_pos, idx_train_pos = pos_idx[:nb_val], pos_idx[nb_val:nb_val + nb_test], pos_idx[
                                                                                                   nb_val + nb_test:]
    idx_val_neg, idx_test_neg, idx_train_neg = neg_idx[:nb_val], neg_idx[nb_val:nb_val + nb_test], neg_idx[
                                                                                                   nb_val + nb_test:]
    return idx_val_pos + idx_val_neg, idx_test_pos + idx_test_neg, idx_train_pos + idx_train_neg


def bin_feat(feat, bins):
    digitized = np.digitize(feat, bins)
    return digitized - digitized.min()


# ############### LINK PREDICTION DATA LOADERS ####################################
def add_super_node_to_adjacency_matrix(adj):
    adj_matrix = np.zeros((adj.shape[0] + 1, adj.shape[1] + 1))
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            adj_matrix[i][j] = adj[i][j]
    for i in range(len(adj_matrix)):
        adj_matrix[i][-1] = 1
    for j in range(len(adj_matrix[0])):
        adj_matrix[-1][j] = 1
    adj_matrix[-1][-1] = 0
    return adj_matrix

def load_data_lp(dataset, use_feats, data_path, use_super_node=False):
    if dataset in ['cora', 'pubmed']:
        adj, features = load_citation_data(dataset, use_feats, data_path)[:2]
    elif dataset == 'disease_lp':
        adj, features = load_synthetic_data(dataset, use_feats, data_path)[:2]
    elif dataset == 'airport':
        adj, features = load_data_airport(dataset, data_path, return_label=False)
    elif dataset == 'binary_tree':
        def get_adjacency_matrix_for_binary_tree(depth):
            adj_matrix = np.array([[0.0 for _ in range(2 ** depth - 1)] for _ in range(2 ** depth - 1)])
            
            for node_index in range(1, (2 ** depth - 1) // 2 + 1):
                adj_matrix[node_index - 1][2 * node_index - 1] = 1.0
                adj_matrix[node_index - 1][2 * node_index] = 1.0
                adj_matrix[2 * node_index - 1][node_index - 1] = 1.0
                adj_matrix[2 * node_index][node_index - 1] = 1.0
                
            return adj_matrix
        
        adj_mat = get_adjacency_matrix_for_binary_tree(TREE_DEPTH)
        adj, features = sp.csr_matrix(adj_mat), np.eye(len(adj_mat))
    elif dataset == 'binary_cyclic_tree':
        if use_super_node: adj_mat = get_adjacency_matrix_for_binary_cyclic_tree_with_super_node(TREE_DEPTH)
        else: adj_mat = get_adjacency_matrix_for_binary_cyclic_tree(TREE_DEPTH) 
        adj, features = sp.csr_matrix(adj_mat), np.eye(len(adj_mat))
        features = np.zeros((len(adj_mat), len(adj_mat)))
        for i in range(len(features)):
            features[i][0] = 1
        print("CONVERTING INTO ZEROS", features)
    elif dataset == 'cam_can_avg':
        # ADJ MAT
        plv_tensor = np.load(os.path.join(data_path, "plv_tensor_276_sbj.npy"))
        plv_average = sum(plv_tensor) / len(plv_tensor)
        plv_node_features = plv_average.copy()
        plv_matrix = plv_average.copy()
        plv_matrix [plv_matrix >= THRESHOLD] = 1
        plv_matrix [plv_matrix < THRESHOLD] = 0
        
        adj = plv_matrix.copy()
        
        if use_super_node: adj_mat = add_super_node_to_adjacency_matrix(adj)
        else: adj_mat = adj
        # Thicks Myelins Features is 2 x 276 x 360
        # FEATURES 276 x 2 x 360
        if use_feats:
            thicks_myelins_tensor = np.load(os.path.join(data_path, "cam_can_thicks_myelins_tensor.npy")) # 276 as well  
            
            average_thicks_myelins = np.sum(thicks_myelins_tensor, axis = 1) / len(thicks_myelins_tensor[0])
            
            thicks_myelins_features = average_thicks_myelins.T
            use_normalize_thicks_myelins = False
            
            if use_normalize_thicks_myelins: 
                thicks_avg = thicks_myelins_features[:, THICK_INDEX]
                myelins_avg = thicks_myelins_features[:, MYELIN_INDEX]
                thicks_avg_mean = np.mean(thicks_avg)
                thicks_avg_sd = np.std(thicks_avg)
                myelins_avg_mean = np.mean(myelins_avg)
                myelins_avg_sd = np.std(myelins_avg)
                normalized_thicks_avg = [(thick - thicks_avg_mean) / thicks_avg_sd for thick in thicks_avg]
                normalized_myelins_avg = [(myelin - myelins_avg_mean) / myelins_avg_sd for myelin in myelins_avg]
                thicks_myelins_features = np.vstack((normalized_thicks_avg, normalized_myelins_avg)), (NUM_ROIS, 2).T
            features = np.hstack((thicks_myelins_features, plv_node_features))
            if use_super_node: 
                one_hot_vec_for_super_node = np.zeros(len(features[0]))
                one_hot_vec_for_super_node[-1] = 1
                features = np.vstack((features, one_hot_vec_for_super_node))
                
        else:
            if use_super_node: 
                features = np.eye(NUM_ROIS + 1)
            else: 
                features = np.eye(NUM_ROIS)
        
        
        adj = sp.csr_matrix(adj_mat)
    elif dataset == 'cam_can_avg_new':
        use_noise = False
        # (592 -5) x 360 x 360
        plv_tensor = np.load(os.path.join(data_path, "plv_tensor_592_sbj_filtered.npy"))
        plv_tensor = remove_self_loops(plv_tensor)
        plv_average = np.sum(plv_tensor, axis=0) / len(plv_tensor)
        plv_node_features = plv_average.copy()
        # ADJ MAT
        plv_matrix = plv_average.copy()
        if use_noise: plv_matrix = plv_matrix + np.random.randn(*plv_matrix.shape)

        plv_matrix [plv_matrix >= THRESHOLD_NEW] = 1
        plv_matrix [plv_matrix < THRESHOLD_NEW] = 0
        
        adj = plv_matrix.copy()
        if use_super_node: adj_mat = add_super_node_to_adjacency_matrix(adj)
        else: adj_mat = adj
        # Thicks Myelins Features is 2 x 592 x 360
        # FEATURES 592 x 2 x 360
        if use_feats:
            thicks_myelins_tensor = np.load(os.path.join(data_path, "cam_can_thicks_myelins_tensor_592_filtered.npy")) # 592 as well  
            
            
            average_thicks_myelins = np.sum(thicks_myelins_tensor, axis = 1) / len(thicks_myelins_tensor[0])
            
            thicks_myelins_features = average_thicks_myelins.T
            
            use_normalize_thicks_myelins = False
            if use_normalize_thicks_myelins: 
                thicks_avg = thicks_myelins_features[:, THICK_INDEX]
                myelins_avg = thicks_myelins_features[:, MYELIN_INDEX]
                thicks_avg_mean = np.mean(thicks_avg)
                thicks_avg_sd = np.std(thicks_avg)
                myelins_avg_mean = np.mean(myelins_avg)
                myelins_avg_sd = np.std(myelins_avg)
                normalized_thicks_avg = [(thick - thicks_avg_mean) / thicks_avg_sd for thick in thicks_avg]
                normalized_myelins_avg = [(myelin - myelins_avg_mean) / myelins_avg_sd for myelin in myelins_avg]

                thicks_myelins_features = np.vstack((normalized_thicks_avg, normalized_myelins_avg)).T
            
            features = np.hstack((thicks_myelins_features, plv_node_features))
            if use_super_node: 
                one_hot_vec_for_super_node = np.zeros(len(features[0]))
                one_hot_vec_for_super_node[-1] = 1
                features = np.vstack((features, one_hot_vec_for_super_node))
        else:
            if use_super_node: features = np.eye(NUM_ROIS + 1)
            else: features = np.eye(NUM_ROIS)
        adj = sp.csr_matrix(adj_mat)
        
        if use_noise: features = features + np.random.randn(*features.shape) * 1 / 100

    elif dataset == 'cam_can_single':
        # ADJ MAT
        subject_index = 275
        age_label = np.load(os.path.join(data_path, "age_labels_276_sbj.npy"))[subject_index]
        print(f"Subject Index : {subject_index}")
        print(f"Subject Age : {age_label}")
        plv_tensor = np.load(os.path.join(data_path, "plv_tensor_276_sbj.npy"))
        plv_sbj_matrix = plv_tensor[subject_index] 
        plv_node_features = plv_sbj_matrix.copy()
        plv_matrix = plv_sbj_matrix.copy()
        plv_matrix [plv_matrix >= THRESHOLD] = 1
        plv_matrix [plv_matrix < THRESHOLD] = 0
        
        adj = plv_matrix.copy()
        if use_super_node: adj_mat = add_super_node_to_adjacency_matrix(adj)
        else: adj_mat = adj
        # Thicks Myelins is 2 x 276 x 360
        # FEATURES 276 x 2 x 360
        if use_feats:
            thicks_myelins_tensor = np.load(os.path.join(data_path, "cam_can_thicks_myelins_tensor.npy")) # 276 as well  
            
            
            thicks_sbj = thicks_myelins_tensor[THICK_INDEX][subject_index]
            myelins_sbj = thicks_myelins_tensor[MYELIN_INDEX][subject_index]
            use_normalize_thicks_myelins = False
            if use_normalize_thicks_myelins: 
                thicks_sbj_mean = np.mean(thicks_sbj)
                thicks_sbj_sd = np.std(thicks_sbj)
                myelins_sbj_mean = np.mean(myelins_sbj)
                myelins_sbj_sd = np.std(myelins_sbj)
                normalized_thicks_sbj = [(thick - thicks_sbj_mean) / thicks_sbj_sd for thick in thicks_sbj]
                normalized_myelins_sbj = [(myelin - myelins_sbj_mean) / myelins_sbj_sd for myelin in myelins_sbj]
                thicks_sbj, myelins_sbj = normalized_thicks_sbj, normalized_myelins_sbj    

            thicks_myelins_features = np.vstack((thicks_sbj, myelins_sbj)).T
            features = np.hstack((thicks_myelins_features, plv_node_features))
            if use_super_node: 
                one_hot_vec_for_super_node = np.zeros(len(features[0]))
                one_hot_vec_for_super_node[-1] = 1
                features = np.vstack((features, one_hot_vec_for_super_node))
        else:
            if use_super_node: features = np.eye(NUM_ROIS + 1) 
            else: features = np.eye(NUM_ROIS)
        adj = sp.csr_matrix(adj_mat)
        
    elif dataset == 'cam_can_single_new':
        # ADJ MAT
        subject_index = 0
        age_label = np.load(os.path.join(data_path, "age_labels_592_sbj_filtered.npy"))[subject_index]
        print(f"Subject Index : {subject_index}")
        print(f"Subject Age : {age_label}")
        plv_tensor = np.load(os.path.join(data_path, "plv_tensor_592_sbj_filtered.npy"))
        plv_tensor = remove_self_loops(plv_tensor)
        plv_sbj_matrix = plv_tensor[subject_index] 
        plv_node_features = plv_sbj_matrix.copy()
        plv_matrix_thresholded = plv_sbj_matrix.copy()
        plv_matrix_thresholded [plv_matrix_thresholded >= THRESHOLD_NEW] = 1
        plv_matrix_thresholded [plv_matrix_thresholded < THRESHOLD_NEW] = 0
        
        adj = plv_matrix_thresholded.copy()
        if use_super_node: adj_mat = add_super_node_to_adjacency_matrix(adj)
        else: adj_mat = adj
        
        # Thicks Myelins is 2 x 276 x 360
        # FEATURES 276 x 2 x 360
        if use_feats:
            thicks_myelins_tensor = np.load(os.path.join(data_path, "cam_can_thicks_myelins_tensor_592_filtered.npy")) # 276 as well  
            
            thicks_sbj = thicks_myelins_tensor[THICK_INDEX][subject_index]
            myelins_sbj = thicks_myelins_tensor[MYELIN_INDEX][subject_index]
            
            use_min_max_scaler = True
            use_z_score_normalization = False
            if use_min_max_scaler:
                thicks_sbj = min_max_normalize(thicks_sbj)
                myelins_sbj = min_max_normalize(myelins_sbj)    
                plv_node_features = min_max_normalize(plv_node_features)
            if use_z_score_normalization:
                thicks_sbj = z_score_normalize(thicks_sbj)
                myelins_sbj = z_score_normalize(myelins_sbj)
            
            thicks_myelins_features = np.vstack((thicks_sbj, myelins_sbj)).T
            features = np.hstack((thicks_myelins_features, plv_node_features))
            if use_super_node: 
                one_hot_vec_for_super_node = np.zeros(len(features[0]))
                one_hot_vec_for_super_node[-1] = 1
                features = np.vstack((features, one_hot_vec_for_super_node))
        else:
            if use_super_node: features = np.eye(NUM_ROIS + 1) 
            else: features = np.eye(NUM_ROIS)
        adj = sp.csr_matrix(adj_mat)
    elif dataset == 'cam_can_multiple':
        train_graph_data_dicts, val_graph_data_dicts, test_graph_data_dicts = load_data_cam_can_new(NUM_SBJS, data_path)
    elif dataset == 'binary_cyclic_tree_multiple':
        train_graph_data_dicts, val_graph_data_dicts, test_graph_data_dicts = load_data_binary_cyclic_tree_multiple(NUM_SBJS, data_path)

    else:
        raise FileNotFoundError('Dataset {} is not supported.'.format(dataset))
    if dataset =='cam_can_multiple' or dataset == "binary_cyclic_tree_multiple":
        data = {'train_graph_data_dicts' : train_graph_data_dicts, 
                'val_graph_data_dicts' : val_graph_data_dicts,
                'test_graph_data_dicts'  : test_graph_data_dicts,
        }

    else:
        data = {'adj_train': adj, 'features': features}
    
    return data

def load_data_cam_can_new(num_sbjs, data_path):
    # ADJ MAT
    # 360 x 360
    # FEATURES 592 x 2 x 360
    # Thicks Myelins is 2 x 592 x 360

    # 360 x 2 n_nodes = 360 feat_dim = 2
    # 360 x 362 n_nodes = 360 feat_dim = 362

    [train_adj_matrices, 
        train_feats, 
        train_age_labels,
        train_split_indices,
        val_adj_matrices, 
        val_feats,
        val_age_labels,
        val_split_indices, 
        test_adj_matrices, 
        test_feats,
        test_age_labels,
        test_split_indices] = get_adj_mat_dataset_splits_with_features_and_age_labels_and_indices(num_sbjs, data_path)
    
    train_graph_data_dicts = [{'adj_train': sp.csr_matrix(train_adj), 
                                'features': train_feat,
                                'age_label': train_age_label,
                                'index': train_split_index} 
                                for train_adj, train_feat, train_age_label, train_split_index 
                                    in zip(train_adj_matrices, train_feats, train_age_labels, train_split_indices)]
    val_graph_data_dicts = [{'adj_train': sp.csr_matrix(val_adj), 
                                'features': val_feat,
                                'age_label': val_age_label,
                                'index': val_split_index} 
                                for val_adj, val_feat, val_age_label, val_split_index 
                                    in zip(val_adj_matrices, val_feats, val_age_labels, val_split_indices)]
    test_graph_data_dicts = [{'adj_train': sp.csr_matrix(test_adj), 
                                'features': test_feat,
                                'age_label': test_age_label,
                                'index': test_split_index} 
                                for test_adj, test_feat, test_age_label, test_split_index 
                                    in zip(test_adj_matrices, test_feats, test_age_labels, test_split_indices)]

    return train_graph_data_dicts, val_graph_data_dicts, test_graph_data_dicts

def load_data_binary_cyclic_tree_multiple(num_sbjs, data_path):
    # ADJ MAT
    # 360 x 360
    # FEATURES 592 x 2 x 360
    # Thicks Myelins is 2 x 592 x 360

    # 360 x 2 n_nodes = 360 feat_dim = 2
    # 360 x 362 n_nodes = 360 feat_dim = 362

    [train_adj_matrices, 
        train_feats, 
        train_age_labels,
        train_split_indices,
        val_adj_matrices, 
        val_feats,
        val_age_labels,
        val_split_indices, 
        test_adj_matrices, 
        test_feats,
        test_age_labels,
        test_split_indices] = get_adj_mat_dataset_splits_with_features_and_age_labels_and_indices_for_binary_cyclic_tree_multiple(num_sbjs, data_path)
    
    train_graph_data_dicts = [{'adj_train': sp.csr_matrix(train_adj), 
                                'features': train_feat,
                                'age_label': train_age_label,
                                'index': train_split_index} 
                                for train_adj, train_feat, train_age_label, train_split_index 
                                    in zip(train_adj_matrices, train_feats, train_age_labels, train_split_indices)]
    val_graph_data_dicts = [{'adj_train': sp.csr_matrix(val_adj), 
                                'features': val_feat,
                                'age_label': val_age_label,
                                'index': val_split_index} 
                                for val_adj, val_feat, val_age_label, val_split_index 
                                    in zip(val_adj_matrices, val_feats, val_age_labels, val_split_indices)]
    test_graph_data_dicts = [{'adj_train': sp.csr_matrix(test_adj), 
                                'features': test_feat,
                                'age_label': test_age_label,
                                'index': test_split_index} 
                                for test_adj, test_feat, test_age_label, test_split_index 
                                    in zip(test_adj_matrices, test_feats, test_age_labels, test_split_indices)]

    return train_graph_data_dicts, val_graph_data_dicts, test_graph_data_dicts

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

def get_adjacency_matrix_for_binary_cyclic_tree_with_super_node(depth):
    adj_matrix = np.array([[0.0 for _ in range(2 ** depth)] for _ in range(2 ** depth)])

    for node_index in range(1, (2 ** depth - 1) // 2 + 1):
        adj_matrix[node_index - 1][2 * node_index - 1] = 1.0
        adj_matrix[node_index - 1][2 * node_index] = 1.0
        adj_matrix[2 * node_index - 1][node_index - 1] = 1.0
        adj_matrix[2 * node_index][node_index - 1] = 1.0
    for node_index in range(len(adj_matrix) - 1):
        if (node_index + 1) % 2 == 0: 
            adj_matrix[node_index][node_index + 1] = 1.0
            adj_matrix[node_index + 1][node_index] = 1.0
    adj_matrix[-1] = [1] * len(adj_matrix)
    for i in range(len(adj_matrix)):
        adj_matrix[i][-1] = 1
    return adj_matrix


def get_dataset_split_indices(num_sbjs) -> List[List[int]]:
    
    train_split_indices, val_split_indices, test_split_indices = [], [], []
    train_num, val_num = int(num_sbjs * TRAIN_SPLIT), int(num_sbjs * VAL_SPLIT)
    test_num = num_sbjs - train_num - val_num 
    
    seen = set()
    for split_indices, split_num in zip([train_split_indices, val_split_indices, test_split_indices],
                                    [train_num, val_num, test_num]):
        num_indices = 0
        while num_indices < split_num:
            index = np.random.randint(0, num_sbjs) # Do not forget filtering! (subtract 5)        
            if index in seen: continue
            split_indices.append(index)
            num_indices += 1
            seen.add(index)
    assert not set(train_split_indices).intersection(val_split_indices), "Train and Val Sets should not overlap"
    assert not set(train_split_indices).intersection(test_split_indices), "Train and Test Sets should not overlap"
    assert not set(val_split_indices).intersection(test_split_indices), "Val and Test Sets should not overlap"
    return train_split_indices, val_split_indices, test_split_indices


def get_adjacency_matrix_cam_can(graph_index, data_path) -> List[List[int]]:
    """
    Create adjacency matrix from threshold number by binarizing the float PLV matrix
    into a 1 and 0 matrix 
    """
    plv_tensor = np.load(os.path.join(data_path, "plv_tensor_592_sbj_filtered.npy"))
    plv_tensor = remove_self_loops(plv_tensor)
    plv_matrix = plv_tensor[graph_index].copy()
    plv_matrix [plv_matrix < THRESHOLD_NEW] = 0
    plv_matrix [plv_matrix >= THRESHOLD_NEW] = 1
    
    return plv_matrix

def get_adj_mat_dataset_splits_with_features_and_age_labels_and_indices_for_binary_cyclic_tree_multiple(num_sbjs : int, data_path):
    
    train_num, val_num = int(num_sbjs * TRAIN_SPLIT), int(num_sbjs * VAL_SPLIT)
    test_num = num_sbjs - train_num - val_num 
    assert train_num + val_num + test_num == num_sbjs, "Must add to total number of subjects"
    
    train_split_indices, val_split_indices, test_split_indices = [0, 100, 200], [250, 450], [125, 525]

    train_split_adj_matrices = [get_adjacency_matrix_for_binary_cyclic_tree(TREE_DEPTH) for i in train_split_indices]
    val_split_adj_matrices = [get_adjacency_matrix_for_binary_cyclic_tree(TREE_DEPTH) for i in val_split_indices]
    test_split_adj_matrices = [get_adjacency_matrix_for_binary_cyclic_tree(TREE_DEPTH) for i in test_split_indices]
    
    train_feats = [np.eye(len(train_split_adj_matrices[0])) + np.random.randn(len(train_split_adj_matrices[0]), len(train_split_adj_matrices[0])) for i in train_split_indices]
    val_feats = [np.eye(len(train_split_adj_matrices[0])) + np.random.randn(len(train_split_adj_matrices[0]), len(train_split_adj_matrices[0])) for i in val_split_indices]
    test_feats = [np.eye(len(train_split_adj_matrices[0])) + np.random.randn(len(train_split_adj_matrices[0]), len(train_split_adj_matrices[0])) for i in test_split_indices]

    age_labels = np.load(os.path.join(data_path, "age_labels_592_sbj_filtered.npy"))
    train_age_labels = [age_labels[i] for i in train_split_indices]
    val_age_labels = [age_labels[i] for i in val_split_indices]
    test_age_labels = [age_labels[i] for i in test_split_indices]

    return [train_split_adj_matrices, 
            train_feats, 
            train_age_labels,
            train_split_indices, 
            val_split_adj_matrices, 
            val_feats, 
            val_age_labels,
            val_split_indices,
            test_split_adj_matrices, 
            test_feats, 
            test_age_labels,
            test_split_indices]


def get_adj_mat_dataset_splits_with_features_and_age_labels_and_indices(num_sbjs : int, data_path):
    
    train_num, val_num = int(num_sbjs * TRAIN_SPLIT), int(num_sbjs * VAL_SPLIT)
    test_num = num_sbjs - train_num - val_num 
    assert train_num + val_num + test_num == num_sbjs, "Must add to total number of subjects"
    # train_split_indices, val_split_indices, test_split_indices = get_dataset_split_indices(num_sbjs)
    # NOTE: Using subset of data for now to test and fix clumping issue, will use full dataset once clumping issue is fixed
    train_split_indices, val_split_indices, test_split_indices = [0, 100, 200, 300, 400, 500], [50, 250, 450], [125, 525]

    train_split_adj_matrices = [get_adjacency_matrix_cam_can(i, data_path) for i in train_split_indices]
    val_split_adj_matrices = [get_adjacency_matrix_cam_can(i, data_path) for i in val_split_indices]
    test_split_adj_matrices = [get_adjacency_matrix_cam_can(i, data_path) for i in test_split_indices]
    
    train_feats = [get_feature_matrix(i, data_path) for i in train_split_indices]
    val_feats = [get_feature_matrix(i, data_path) for i in val_split_indices]
    test_feats = [get_feature_matrix(i, data_path) for i in test_split_indices]

    age_labels = np.load(os.path.join(data_path, "age_labels_592_sbj_filtered.npy"))
    train_age_labels = [age_labels[i] for i in train_split_indices]
    val_age_labels = [age_labels[i] for i in val_split_indices]
    test_age_labels = [age_labels[i] for i in test_split_indices]

    return [train_split_adj_matrices, 
            train_feats, 
            train_age_labels,
            train_split_indices, 
            val_split_adj_matrices, 
            val_feats, 
            val_age_labels,
            val_split_indices,
            test_split_adj_matrices, 
            test_feats, 
            test_age_labels,
            test_split_indices]

def remove_self_loops(plv_tensor):
    for sbj_index in range(len(plv_tensor)):
        for i in range(NUM_ROIS):
            plv_tensor[sbj_index][i][i] = 0
    return plv_tensor

def get_feature_matrix(graph_index, data_path):
    """
    Feature Matrix for specific subject if 360 x 362 since PLV_vectors are 360
    [CT_0   Myelin_0      PLV_vector_0.T      ]
    [CT_1   Myelin_1      PLV_vector_1.T      ]
    ...
    [CT_359 Myelin_359    PLV_vector_359.T    ]

    """
    plv_tensor = np.load(os.path.join(data_path, "plv_tensor_592_sbj_filtered.npy"))
    
    plv_tensor = remove_self_loops(plv_tensor)
    thicks_myelins_tensor = np.load(os.path.join(data_path, "cam_can_thicks_myelins_tensor_592_filtered.npy")) # 592 as well  
    # Thicks Myelins is 2 x 592 x 360
    # TODO: Make PLV Tensor and Thickness + Myelins Tensor global to avoid 
    # constant loading
    
    plv_features = plv_tensor[graph_index].copy()
    
    thick_features = thicks_myelins_tensor[THICK_INDEX][graph_index]
    thick_features = np.reshape(thick_features, (len(thick_features), 1))

    myelin_features = thicks_myelins_tensor[MYELIN_INDEX][graph_index] 
    myelin_features = np.reshape(myelin_features, (len(myelin_features), 1))

    use_min_max_scaler = True
    use_z_score_normalization = False
    if use_min_max_scaler:
        thick_features = min_max_normalize(thick_features)
        myelin_features = min_max_normalize(myelin_features)    
        plv_features = min_max_normalize(plv_features)
    if use_z_score_normalization:
        thick_features = z_score_normalize(thick_features)
        myelin_features = z_score_normalize(myelin_features)
    use_thicks_myelins = True
    if use_thicks_myelins:
        feature_matrix = np.hstack((thick_features, myelin_features, plv_features))
    else:
        feature_matrix = plv_features
    import logging
    logging.getLogger().setLevel(logging.INFO)
    logging.info(f"Feature Matrix: {feature_matrix}")

    return feature_matrix


# ############### NODE CLASSIFICATION DATA LOADERS ####################################


def load_data_nc(dataset, use_feats, data_path, split_seed):
    if dataset in ['cora', 'pubmed']:
        adj, features, labels, idx_train, idx_val, idx_test = load_citation_data(
            dataset, use_feats, data_path, split_seed
        )
    else:
        if dataset == 'disease_nc':
            adj, features, labels = load_synthetic_data(dataset, use_feats, data_path)
            val_prop, test_prop = 0.10, 0.60
        elif dataset == 'airport':
            adj, features, labels = load_data_airport(dataset, data_path, return_label=True)
            val_prop, test_prop = 0.15, 0.15
        else:
            raise FileNotFoundError('Dataset {} is not supported.'.format(dataset))
        idx_val, idx_test, idx_train = split_data(labels, val_prop, test_prop, seed=split_seed)

    labels = torch.LongTensor(labels)
    data = {'adj_train': adj, 'features': features, 'labels': labels, 'idx_train': idx_train, 'idx_val': idx_val, 'idx_test': idx_test}
    return data


# ############### DATASETS ####################################


def load_citation_data(dataset_str, use_feats, data_path, split_seed=None):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open(os.path.join(data_path, "ind.{}.{}".format(dataset_str, names[i])), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(os.path.join(data_path, "ind.{}.test.index".format(dataset_str)))
    test_idx_range = np.sort(test_idx_reorder)

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    labels = np.argmax(labels, 1)

    idx_test = test_idx_range.tolist()
    idx_train = list(range(len(y)))
    idx_val = range(len(y), len(y) + min(1000, len(labels) - len(y) - len(idx_test)))

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    if not use_feats:
        features = sp.eye(adj.shape[0])
    return adj, features, labels, idx_train, idx_val, idx_test


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_synthetic_data(dataset_str, use_feats, data_path):
    object_to_idx = {}
    idx_counter = 0
    edges = []
    with open(os.path.join(data_path, "{}.edges.csv".format(dataset_str)), 'r') as f:
        all_edges = f.readlines()
    for line in all_edges:
        n1, n2 = line.rstrip().split(',')
        if n1 in object_to_idx:
            i = object_to_idx[n1]
        else:
            i = idx_counter
            object_to_idx[n1] = i
            idx_counter += 1
        if n2 in object_to_idx:
            j = object_to_idx[n2]
        else:
            j = idx_counter
            object_to_idx[n2] = j
            idx_counter += 1
        edges.append((i, j))
    adj = np.zeros((len(object_to_idx), len(object_to_idx)))
    for i, j in edges:
        adj[i, j] = 1.  # comment this line for directed adjacency matrix
        adj[j, i] = 1.
    if use_feats:
        features = sp.load_npz(os.path.join(data_path, "{}.feats.npz".format(dataset_str)))
    else:
        features = sp.eye(adj.shape[0])
    labels = np.load(os.path.join(data_path, "{}.labels.npy".format(dataset_str)))
    return sp.csr_matrix(adj), features, labels


def load_data_airport(dataset_str, data_path, return_label=False):
    graph = pkl.load(open(os.path.join(data_path, dataset_str + '.p'), 'rb'))
    adj = nx.adjacency_matrix(graph)
    
    features = np.array([graph.nodes[u]['feat'] for u in graph.nodes()])
    if return_label:
        label_idx = 4
        labels = features[:, label_idx]
        features = features[:, :label_idx]
        labels = bin_feat(labels, bins=[7.0/7, 8.0/7, 9.0/7])
        return sp.csr_matrix(adj), features, labels
    else:
        return sp.csr_matrix(adj), features

