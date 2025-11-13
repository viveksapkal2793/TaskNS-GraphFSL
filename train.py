from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.optim as optim

from utils import *
from models import *
import math

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--use_cuda', action='store_true', help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=1234, help='Random seed.')
parser.add_argument('--episodes', type=int, default=2001,
                    help='Number of episodes to train.') 
parser.add_argument('--lr', type=float, default=0.0005,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.') 
parser.add_argument('--dropout', type=float, default=0.4,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--way', type=int, default=5, help='way.')
parser.add_argument('--shot', type=int, default=3,help='shot.')
parser.add_argument('--qry', type=int, help='k shot for query set', default=15)
parser.add_argument('--dataset', default='Amazon_clothing', help='Dataset:Amazon_clothing/Amazon_electronics/dblp/corafull')
parser.add_argument('--aux_way', type=int, default=8, help='')
parser.add_argument('--aux_num_per_way', type=int, default=20, help='') 
parser.add_argument('--outlier_num', type=int, default=10, help='outlier_num')
parser.add_argument('--alpha', type=float, default=0.7, help='NK_loss')
parser.add_argument('--rhop', type=int, default=2, help='r-hop')
parser.add_argument('--external_ood_path', type=str, default=None, help='Dataset name for external OOD (e.g., Amazon_electronics when training on Amazon_clothing)')
parser.add_argument('--external_ood_ratio', type=float, default=0.5, help='Ratio of external OOD samples (0.5 means 50% external, 50% internal)')

args = parser.parse_args()
args.cuda = args.use_cuda and torch.cuda.is_available()

random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
print("Job started with args:", args, flush=True)
dataset = args.dataset
print('Loading {} dataset...'.format(dataset), flush=True)
adj, features, labels, degrees, class_list_train, class_list_valid, class_list_test, id_by_class = load_data(dataset)
print('Dataset has {} nodes, {} features.'.format(features.shape[0], features.shape[1]), flush=True)

# Load external OOD graph if specified
ext_features = None
ext_adj = None
ext_embeddings = None
feature_adapter = None

if args.external_ood_path is not None:
    print(f'Loading external OOD dataset: {args.external_ood_path}...', flush=True)
    ext_adj, ext_features, ext_labels, ext_degrees, _, _, _, _ = load_data_more(args.external_ood_path)
    print(f'External dataset has {ext_features.shape[0]} nodes, {ext_features.shape[1]} features.', flush=True)
    
    # Handle feature dimension mismatch
    src_dim = features.shape[1]
    ext_dim = ext_features.shape[1]
    
    if ext_dim != src_dim:
        print(f'Feature dimension mismatch: source={src_dim}, external={ext_dim}. Creating adapter...', flush=True)
        import torch.nn as nn
        feature_adapter = nn.Linear(ext_dim, src_dim)
        if args.cuda:
            feature_adapter.cuda()
            ext_features = ext_features.cuda()
            ext_adj = ext_adj.cuda()
        # Apply adapter
        ext_features = feature_adapter(ext_features)
    else:
        if args.cuda:
            ext_features = ext_features.cuda()
            ext_adj = ext_adj.cuda()

# Model and optimizer
encoder = GPN_Encoder(nfeat=features.shape[1],
            nhid=args.hidden,
            dropout=args.dropout)

scorer = GPN_Valuator(nfeat=features.shape[1],
            nhid=args.hidden,
            dropout=args.dropout) #initial score


optimizer_encoder = optim.Adam(encoder.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

optimizer_scorer = optim.Adam(scorer.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

# Add adapter to optimizer if it exists
if feature_adapter is not None:
    optimizer_adapter = optim.Adam(feature_adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay)
else:
    optimizer_adapter = None

if args.cuda:
    encoder.cuda()
    scorer.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    degrees = degrees.cuda()

def train(ID_class_selected, id_support, id_query, OOD_sample, OOD_embeddings_ext, n_way, k_shot, num_OOD, num_OOD_ext):
    encoder.train()
    scorer.train()
    if feature_adapter is not None:
        feature_adapter.train()
    
    optimizer_encoder.zero_grad()
    optimizer_scorer.zero_grad()
    if optimizer_adapter is not None:
        optimizer_adapter.zero_grad()
    
    embeddings = encoder(features, adj)
    z_dim = embeddings.size()[1]
    scores = scorer(features, adj)
    
    support_embeddings = embeddings[id_support]
    support_embeddings = support_embeddings.view([n_way, k_shot, z_dim])
    query_embeddings = embeddings[id_query]
    
    # Internal OOD embeddings
    OOD_embeddings_internal = embeddings[OOD_sample] if len(OOD_sample) > 0 else None
    
    support_degrees = torch.log(degrees[id_support].view([n_way, k_shot]))
    support_scores = scores[id_support].view([n_way, k_shot])
    support_scores = torch.sigmoid(support_degrees * support_scores).unsqueeze(-1)
    support_scores = support_scores / torch.sum(support_scores, dim=1, keepdim=True)
    support_embeddings = support_embeddings * support_scores 
    prototype_embeddings = support_embeddings.sum(1)
    
    # Query loss
    dists = euclidean_dist(query_embeddings, prototype_embeddings)
    output = F.log_softmax(-dists, dim=1)
    labels_new = torch.LongTensor([ID_class_selected.index(i) for i in labels[id_query]])
    if args.cuda:
        labels_new = labels_new.cuda()
    loss_train = F.nll_loss(output, labels_new)
    
    # OOD loss (combined internal + external)
    loss_OOD = torch.tensor(0., requires_grad=True)
    if args.cuda:
        loss_OOD = loss_OOD.cuda()
    
    # Internal OOD loss
    if OOD_embeddings_internal is not None and num_OOD > 0:
        dists_OOD_internal = euclidean_dist(OOD_embeddings_internal, prototype_embeddings)
        output_OOD_internal = F.softmax(-dists_OOD_internal, dim=1)
        for i in range(num_OOD):
            output_OOD_each = output_OOD_internal[i]
            output_OOD_max = torch.max(output_OOD_each)
            loss_OOD = loss_OOD + output_OOD_max
    
    # External OOD loss
    if OOD_embeddings_ext is not None and num_OOD_ext > 0:
        dists_OOD_ext = euclidean_dist(OOD_embeddings_ext, prototype_embeddings)
        output_OOD_ext = F.softmax(-dists_OOD_ext, dim=1)
        for i in range(num_OOD_ext):
            output_OOD_each = output_OOD_ext[i]
            output_OOD_max = torch.max(output_OOD_each)
            loss_OOD = loss_OOD + output_OOD_max

    # Average OOD loss
    total_ood = num_OOD + num_OOD_ext
    if total_ood > 0:
        loss_OOD = loss_OOD / total_ood

    loss_total = alpha * loss_train + (1 - alpha) * loss_OOD
    loss_total.backward()
    
    optimizer_encoder.step()
    optimizer_scorer.step()
    if optimizer_adapter is not None:
        optimizer_adapter.step()
    
    if args.cuda:
        output = output.cpu().detach()
        labels_new = labels_new.cpu().detach()
    acc_train = accuracy(output, labels_new)
    f1_train = f1(output, labels_new)
    return acc_train, f1_train

def test(class_selected, id_support, id_query, n_way, k_shot, episode):
    encoder.eval()
    scorer.eval()
    embeddings = encoder(features, adj)
    z_dim = embeddings.size()[1]
    scores = scorer(features, adj)
    support_embeddings_ini = embeddings[id_support]
    support_embeddings = support_embeddings_ini.view([n_way, k_shot, z_dim])
    query_embeddings = embeddings[id_query]
    support_degrees = torch.log(degrees[id_support].view([n_way, k_shot]))
    support_scores = scores[id_support].view([n_way, k_shot])
    support_scores = torch.sigmoid(support_degrees * support_scores).unsqueeze(-1)
    support_scores = support_scores / torch.sum(support_scores, dim=1, keepdim=True)
    support_embeddings = support_embeddings * support_scores
    prototype_embeddings = support_embeddings.sum(1) 
    dists = euclidean_dist(query_embeddings, prototype_embeddings)
    output = F.log_softmax(-dists, dim=1)
    support_dists = euclidean_dist(support_embeddings_ini, prototype_embeddings)
    support_prob= F.softmax(-support_dists, dim=1)
    s_prob = support_prob.tolist()
    query_prob= F.softmax(-dists, dim=1)
    q_prob = query_prob.tolist()
    radius_pre = [[0 for j in range(k_shot)] for i in range(n_way)]
    for j in range(n_way):
        for i in range(k_shot):
            radius_pre[n_way-j-1][i] = s_prob[-1][n_way-j-1]
            s_prob.pop()
    labels_new = torch.LongTensor([class_selected.index(i) for i in labels[id_query]])
    if args.cuda:
        labels_new = labels_new.cuda()
    loss_test = F.nll_loss(output, labels_new)
    if args.cuda:
        output = output.cpu().detach()
        labels_new = labels_new.cpu().detach()
    acc_test = accuracy(output, labels_new)
    f1_test = f1(output, labels_new)
    return acc_test, f1_test

if __name__ == '__main__':
    n_query = args.qry
    r = args.rhop
    meta_test_num = 50 
    meta_valid_num = 50
    alpha = args.alpha
    Outlier_num = args.outlier_num 
    o_way= args.aux_way
    o_num_way = args.aux_num_per_way
    settings = [(args.way, args.shot)]

    for n_way, k_shot in settings:
        valid_pool = [task_generator(id_by_class, class_list_valid, n_way, k_shot, n_query) for i in range(meta_valid_num)]
        test_pool = [task_generator(id_by_class, class_list_test, n_way, k_shot, n_query) for i in range(meta_test_num)]
        t_total = time.time()
        meta_train_acc = []
        meta_train_f1 = []
        for episode in range(args.episodes):
            print('--- Episode %d ---' % episode, flush=True)
            id_support, id_query, OOD_sample, ID_class_selected = \
                select_task_generator(adj, id_by_class, class_list_train, n_way, k_shot, n_query, o_way, o_num_way, Outlier_num)
            
            # Determine split between internal and external OOD
            num_OOD_internal = len(OOD_sample)
            num_OOD_ext = 0
            OOD_embeddings_ext = None
            
            if ext_features is not None and args.external_ood_ratio > 0:
                # Calculate number of external OOD samples
                total_ood = Outlier_num
                num_OOD_ext = int(total_ood * args.external_ood_ratio)
                num_OOD_internal = total_ood - num_OOD_ext
                
                # Adjust internal OOD sample
                if num_OOD_internal < len(OOD_sample):
                    OOD_sample = OOD_sample[:num_OOD_internal]
                
                # Sample external OOD nodes
                ext_node_indices = random.sample(range(ext_features.shape[0]), num_OOD_ext)
                
                # Compute external embeddings
                with torch.no_grad():
                    ext_embeddings_batch = encoder(ext_features, ext_adj)
                OOD_embeddings_ext = ext_embeddings_batch[ext_node_indices]
            
            aux_num = num_OOD_internal
            acc_train, f1_train = train(ID_class_selected, id_support, id_query, OOD_sample, 
                                        OOD_embeddings_ext, n_way, k_shot, aux_num, num_OOD_ext)
            meta_train_acc.append(acc_train)
            meta_train_f1.append(f1_train)

            if episode % 1 == 0:
                print(f'Episode {episode}: Train Acc: {acc_train:.4f}, Train F1: {f1_train:.4f}', flush=True)
            
            if episode > 0 and episode % 100 == 0:
                meta_test_acc = []
                meta_test_f1 = []
                for idx in range(meta_test_num):
                    id_support, id_query, ID_class_selected = test_pool[idx]
                    acc_test, f1_test= test(ID_class_selected, id_support, id_query, n_way, k_shot, episode)
                    meta_test_acc.append(acc_test)
                    meta_test_f1.append(f1_test)

                avg_test_acc = np.mean(meta_test_acc)
                avg_test_f1 = np.mean(meta_test_f1)
                print(f'Episode {episode}: Test Acc: {avg_test_acc:.4f} ± {np.std(meta_test_acc):.4f}')
                print(f'Episode {episode}: Test F1: {avg_test_f1:.4f} ± {np.std(meta_test_f1):.4f}')
                print('-' * 50)
