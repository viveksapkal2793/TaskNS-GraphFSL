import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.datasets import Coauthor
from scipy.sparse import csr_matrix
import torch_geometric
import scipy.io as sio
import random
import json
import pickle
import networkx as nx
from sklearn import preprocessing
from sklearn.metrics import f1_score
from collections import defaultdict

def load_npz_to_sparse_graph(file_name):
    """Load a SparseGraph from a Numpy binary file.
    Parameters
    ----------
    file_name : str
        Name of the file to load.
    Returns
    -------
    sparse_graph : SparseGraph
        Graph in sparse matrix format.
    """
    with np.load(file_name) as loader:
        loader = dict(loader)
        adj_matrix = sp.csr_matrix((loader['adj_data'], loader['adj_indices'], loader['adj_indptr']),
                                   shape=loader['adj_shape'])

        if 'attr_data' in loader:
            # Attributes are stored as a sparse CSR matrix
            attr_matrix = sp.csr_matrix((loader['attr_data'], loader['attr_indices'], loader['attr_indptr']),
                                        shape=loader['attr_shape'])
        elif 'attr_matrix' in loader:
            # Attributes are stored as a (dense) np.ndarray
            attr_matrix = loader['attr_matrix']
        else:
            attr_matrix = None

        if 'labels_data' in loader:
            # Labels are stored as a CSR matrix
            labels = sp.csr_matrix((loader['labels_data'], loader['labels_indices'], loader['labels_indptr']),
                                   shape=loader['labels_shape'])
        elif 'labels' in loader:
            # Labels are stored as a numpy array
            labels = loader['labels']
        else:
            labels = None

        node_names = loader.get('node_names')
        attr_names = loader.get('attr_names')
        class_names = loader.get('class_names')
        metadata = loader.get('metadata')

    return adj_matrix, attr_matrix, labels, node_names, attr_names, class_names, metadata


valid_num_dic = {'Amazon_clothing': 17, 'Amazon_eletronics': 36, 'dblp': 27}

def load_data(dataset_source):
    # adj, features, labels, degree = None, None, None, None
    # class_list_train, class_list_valid, class_list_test, id_by_class = None, None, None, None
    
    if dataset_source in ['Amazon_clothing', 'Amazon_electronics', 'dblp']:
        n1s = []
        n2s = []
        for line in open("few_shot_data/{}_network".format(dataset_source)):
            n1, n2 = line.strip().split('\t')
            n1s.append(int(n1))
            n2s.append(int(n2))

        num_nodes = max(max(n1s),max(n2s)) + 1
        adj = sp.coo_matrix((np.ones(len(n1s)), (n1s, n2s)),
                                    shape=(num_nodes, num_nodes))


        data_train = sio.loadmat("few_shot_data/{}_train.mat".format(dataset_source))
        train_class = list(set(data_train["Label"].reshape((1,len(data_train["Label"])))[0]))
        

        data_test = sio.loadmat("few_shot_data/{}_test.mat".format(dataset_source))
        class_list_test = list(set(data_test["Label"].reshape((1,len(data_test["Label"])))[0]))


        labels = np.zeros((num_nodes,1))
        labels[data_train['Index']] = data_train["Label"]
        labels[data_test['Index']] = data_test["Label"]

        features = np.zeros((num_nodes,data_train["Attributes"].shape[1]))
        features[data_train['Index']] = data_train["Attributes"].toarray()
        features[data_test['Index']] = data_test["Attributes"].toarray()

        class_list = []
        for cla in labels:
            if cla[0] not in class_list:
                class_list.append(cla[0])  # unsorted

        id_by_class = {}
        for i in class_list:
            id_by_class[i] = []
        for id, cla in enumerate(labels):
            id_by_class[cla[0]].append(id)

        lb = preprocessing.LabelBinarizer()
        labels = lb.fit_transform(labels)

        degree = np.sum(adj, axis=1)
        degree = torch.FloatTensor(degree)

        adj = normalize_adj(adj + sp.eye(adj.shape[0]))
        features = torch.FloatTensor(features)
        labels = torch.LongTensor(np.where(labels)[1])

        adj = sparse_mx_to_torch_sparse_tensor(adj)
        
        class_list_valid = random.sample(train_class, valid_num_dic[dataset_source])

        class_list_train = list(set(train_class).difference(set(class_list_valid)))

    elif dataset_source == 'corafull':
        cora_full = torch_geometric.datasets.CitationFull(
            './dataset', 'cora')

        edges = cora_full.data.edge_index

        num_nodes = max(max(edges[0]), max(edges[1])) + 1

        adj = sp.coo_matrix((np.ones(len(edges[0])), (edges[0], edges[1])),
                            shape=(num_nodes, num_nodes))

        features = cora_full.data.x
        labels = cora_full.data.y

        class_list = cora_full.data.y.unique().tolist()

        with open(file='./dataset/cora/cls_split.pkl', mode='rb') as f:
            class_list_train, class_list_valid, class_list_test = pickle.load(f)

        id_by_class = {}
        for i in class_list:
            id_by_class[i] = []
        for id, cla in enumerate(labels.tolist()):
            id_by_class[cla].append(id)

        lb = preprocessing.LabelBinarizer()
        labels = lb.fit_transform(labels)

        degree = np.sum(adj, axis=1)
        degree = torch.FloatTensor(degree)

        adj = normalize_adj(adj + sp.eye(adj.shape[0]))
        features = torch.FloatTensor(features)
        labels = torch.LongTensor(np.where(labels)[1])

        adj = sparse_mx_to_torch_sparse_tensor(adj)
        
    return adj, features, labels, degree, class_list_train, class_list_valid, class_list_test, id_by_class 

def load_data_more(dataset_source):
    if dataset_source in ['Amazon_clothing', 'Amazon_electronics', 'dblp']:
        n1s = []
        n2s = []
        for line in open("few_shot_data/{}_network".format(dataset_source)):
            n1, n2 = line.strip().split('\t')
            n1s.append(int(n1))
            n2s.append(int(n2))

        num_nodes = max(max(n1s),max(n2s)) + 1
        adj = sp.coo_matrix((np.ones(len(n1s)), (n1s, n2s)),
                                    shape=(num_nodes, num_nodes))


        data_train = sio.loadmat("few_shot_data/{}_train.mat".format(dataset_source))
        train_class = list(set(data_train["Label"].reshape((1,len(data_train["Label"])))[0]))
        

        data_test = sio.loadmat("few_shot_data/{}_test.mat".format(dataset_source))
        class_list_test = list(set(data_test["Label"].reshape((1,len(data_test["Label"])))[0]))


        labels = np.zeros((num_nodes,1))
        labels[data_train['Index']] = data_train["Label"]
        labels[data_test['Index']] = data_test["Label"]

        features = np.zeros((num_nodes,data_train["Attributes"].shape[1]))
        features[data_train['Index']] = data_train["Attributes"].toarray()
        features[data_test['Index']] = data_test["Attributes"].toarray()

        class_list = []
        for cla in labels:
            if cla[0] not in class_list:
                class_list.append(cla[0])  # unsorted

        id_by_class = {}
        for i in class_list:
            id_by_class[i] = []
        for id, cla in enumerate(labels):
            id_by_class[cla[0]].append(id)

        lb = preprocessing.LabelBinarizer()
        labels = lb.fit_transform(labels)

        degree = np.sum(adj, axis=1)
        degree = torch.FloatTensor(degree)

        adj = normalize_adj(adj + sp.eye(adj.shape[0]))
        features = torch.FloatTensor(features)
        labels = torch.LongTensor(np.where(labels)[1])

        adj = sparse_mx_to_torch_sparse_tensor(adj)
        
        class_list_valid = random.sample(train_class, valid_num_dic[dataset_source])

        class_list_train = list(set(train_class).difference(set(class_list_valid)))

    elif dataset_source == 'corafull':
        cora_full = torch_geometric.datasets.CitationFull(
            './dataset', 'cora')

        edges = cora_full.data.edge_index

        num_nodes = max(max(edges[0]), max(edges[1])) + 1

        adj = sp.coo_matrix((np.ones(len(edges[0])), (edges[0], edges[1])),
                            shape=(num_nodes, num_nodes))

        features = cora_full.data.x
        labels = cora_full.data.y

        class_list = cora_full.data.y.unique().tolist()

        with open(file='./dataset/cora/cls_split.pkl', mode='rb') as f:
            class_list_train, class_list_valid, class_list_test = pickle.load(f)

        id_by_class = {}
        for i in class_list:
            id_by_class[i] = []
        for id, cla in enumerate(labels.tolist()):
            id_by_class[cla].append(id)

        lb = preprocessing.LabelBinarizer()
        labels = lb.fit_transform(labels)

        degree = np.sum(adj, axis=1)
        degree = torch.FloatTensor(degree)

        adj = normalize_adj(adj + sp.eye(adj.shape[0]))
        features = torch.FloatTensor(features)
        labels = torch.LongTensor(np.where(labels)[1])

        adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj, features, labels, degree, class_list_train, class_list_valid, class_list_test, id_by_class 


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def f1(output, labels):
    preds = output.max(1)[1].type_as(labels)
    f1 = f1_score(labels, preds, average='weighted')
    return f1


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def find_neighbors(adj, id_by_class, sample_id, k_hop):
    result_id_by_class = {}
    for key, value in id_by_class.items():
        for num in value:
            result_id_by_class[num] = key
    neighbors = {sample_id}
    for _ in range(k_hop):
        new_neighbors = set()
        for node in neighbors:
            for i in range(len(adj)):
                if adj[node][i] == 1:
                    new_neighbors.add(i)
        neighbors = neighbors.union(new_neighbors)
    aux_id_class = {node: result_id_by_class[node] for node in neighbors}

    result = defaultdict(list)
    for key, value in aux_id_class.items():
        result[value].append(key)
    aux_id_class_result = dict(result)

    return aux_id_class_result

def task_generator(id_by_class, class_list, n_way, k_shot, m_query):
    # sample class indices
    class_selected = random.sample(class_list, n_way)
    id_support = []
    id_query = []
    for cla in class_selected:
        temp = random.sample(id_by_class[cla], k_shot + m_query)
        id_support.extend(temp[:k_shot])
        id_query.extend(temp[k_shot:])

    return np.array(id_support), np.array(id_query), class_selected


def G_task_generator(adj, id_by_class, class_list, n_way, k_shot, m_query, hop_num, Outlier_num):
    ID_class_selected = random.sample(class_list, n_way)
    id_support = []
    id_query = []
    for cla in ID_class_selected:
        temp = random.sample(id_by_class[cla], k_shot + m_query)
        id_support.extend(temp[:k_shot]) 
        id_query.extend(temp[k_shot:]) 
    neighbors_id_class = {}
    for id in id_support:
        each_result = find_neighbors(adj, id_by_class, id, hop_num)
        for key, value in each_result.items():
            if key in neighbors_id_class:
                neighbors_id_class[key].extend([v for v in value if v not in neighbors_id_class[key]])
            else:
                neighbors_id_class[key] = value
    aux_query_list = []
    for class_name, ids in neighbors_id_class.items():
        if class_name not in ID_class_selected:
            aux_query_list.extend(ids)
    aux_query = random.sample(aux_query_list, Outlier_num)
    return np.array(id_support), np.array(id_query), np.array(aux_query), ID_class_selected

def get_shortest_path_lengths(adj, id_support, aux_query):
    id_to_index = {id:i for i, id in enumerate(id_support)}
    adj = np.array(adj.to_dense())
    graph = nx.from_numpy_array(adj)
    shortest_path_lengths = {}
    for query_node in aux_query:
        shortest_paths = nx.shortest_path_length(graph, source=query_node)
        lengths = [shortest_paths.get(node, float('inf')) for node in id_support]
        shortest_path_lengths[query_node] = lengths

    return shortest_path_lengths

def task_generator_o(adj, id_by_class, class_list_train, n_way, k_shot, m_query, r, Outlier_num):
    G = adj
    support_set = set()
    query_set = set()
    selected_classes = random.sample(class_list_train, n_way)   
    for cla in selected_classes:
        temp = random.sample(id_by_class[cla], k_shot + m_query)
        support_set.update(temp[:k_shot])
        query_set.update(temp[k_shot:])
    if G.is_sparse:
        G = G.to_dense().numpy() 
        G = nx.from_numpy_array(G) 
    r_hop_neighbors = set()
    for node in support_set:
        neighbors = nx.single_source_shortest_path_length(G, node, cutoff=r)
        for neighbor, dist in neighbors.items():
            if dist > 0: 
                r_hop_neighbors.add(neighbor)
    filtered_neighbors = set()
    for neighbor in r_hop_neighbors:
        for train_class in class_list_train:
            if neighbor in id_by_class[train_class]:
                if neighbor not in support_set and train_class not in selected_classes:
                    filtered_neighbors.add(neighbor)

    if len(filtered_neighbors) < Outlier_num:
        print("Not enough neighbors to select {} outliers. Only {} neighbors available.".format(Outlier_num,len(filtered_neighbors)))
        return np.array(list(support_set)), np.array(list(query_set)),np.array(list(filtered_neighbors)), selected_classes 
    outlier_set = random.sample(filtered_neighbors, Outlier_num)
    return np.array(list(support_set)), np.array(list(query_set)), np.array(outlier_set), selected_classes 
    

def select_task_generator(adj, id_by_class, class_list, n_way, k_shot, m_query, aux_way, aux_num_per_way, Outlier_num):
    selected_classes = random.sample(class_list, n_way + aux_way)   
    id_support = []
    id_query = []
    aux_query = []
    i = 0
    for cla in selected_classes:
        if i < n_way: 
            temp = random.sample(id_by_class[cla], k_shot + m_query)
            id_support.extend(temp[:k_shot])
            id_query.extend(temp[k_shot:])
        else: 
            auxtemp = random.sample(id_by_class[cla], aux_num_per_way) 
            aux_query.extend(auxtemp[:aux_num_per_way])
        i = i + 1
    shortest_path_lengths = get_shortest_path_lengths(adj, id_support, aux_query)
    node_lengths = {node: min(lengths) for node, lengths in shortest_path_lengths.items()}
    final_outlier_list = sorted(aux_query, key=lambda x: node_lengths[x])[:Outlier_num]
    return np.array(id_support), np.array(id_query), np.array(final_outlier_list), selected_classes 

def euclidean_dist(x, y):
    # x: N x D query
    # y: M x D prototype
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)  # N x M


