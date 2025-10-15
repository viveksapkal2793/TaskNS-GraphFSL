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
parser.add_argument('--dataset', default='Amazon_clothing', help='Dataset:Amazon_clothing/Amazon_eletronics/dblp/corafull')
parser.add_argument('--aux_way', type=int, default=8, help='')
parser.add_argument('--aux_num_per_way', type=int, default=20, help='') 
parser.add_argument('--outlier_num', type=int, default=10, help='outlier_num')
parser.add_argument('--alpha', type=float, default=0.7, help='NK_loss')
parser.add_argument('--rhop', type=int, default=2, help='r-hop')

args = parser.parse_args()
args.cuda = args.use_cuda and torch.cuda.is_available()

random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

dataset = args.dataset
adj, features, labels, degrees, class_list_train, class_list_valid, class_list_test, id_by_class = load_data(dataset)

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

if args.cuda:
    encoder.cuda()
    scorer.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    degrees = degrees.cuda()

def train(ID_class_selected, id_support, id_query, OOD_sample, n_way, k_shot, num_OOD):
    encoder.train()
    scorer.train()
    optimizer_encoder.zero_grad()
    optimizer_scorer.zero_grad()
    embeddings = encoder(features, adj)
    z_dim = embeddings.size()[1]
    scores = scorer(features, adj)
    support_embeddings = embeddings[id_support]
    support_embeddings = support_embeddings.view([n_way, k_shot, z_dim])
    query_embeddings = embeddings[id_query]
    OOD_embeddings = embeddings[OOD_sample]
    support_degrees = torch.log(degrees[id_support].view([n_way, k_shot]))
    support_scores = scores[id_support].view([n_way, k_shot])
    support_scores = torch.sigmoid(support_degrees * support_scores).unsqueeze(-1)
    support_scores = support_scores / torch.sum(support_scores, dim=1, keepdim=True)
    support_embeddings = support_embeddings * support_scores 
    prototype_embeddings = support_embeddings.sum(1)
    dists = euclidean_dist(query_embeddings, prototype_embeddings)
    dists_OOD = euclidean_dist(OOD_embeddings, prototype_embeddings)
    output = F.log_softmax(-dists, dim=1)
    labels_new = torch.LongTensor([ID_class_selected.index(i) for i in labels[id_query]])
    if args.cuda:
        labels_new = labels_new.cuda()
    output_OOD = F.softmax(-dists_OOD, dim=1)
    loss_train = F.nll_loss(output, labels_new)
    loss_OOD = torch.tensor(0.,requires_grad = True)
    for i in range(num_OOD):
        output_OOD_each = output_OOD[i]
        output_OOD_max = torch.tensor(-10.,requires_grad = True)
        for j in range(n_way):
            if output_OOD_each[j] > output_OOD_max:
                output_OOD_max = output_OOD_each[j]
        loss_OOD = loss_OOD + output_OOD_max 
    loss_OOD = loss_OOD / num_OOD 
    loss_total = alpha * loss_train + (1-alpha) * loss_OOD 
    print_loss_total = loss_total
    loss_total.backward()
    optimizer_encoder.step()
    optimizer_scorer.step()
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
        for episode in range(args.episodes):#
            id_support, id_query, OOD_sample, ID_class_selected = \
                select_task_generator(adj, id_by_class, class_list_train, n_way, k_shot, n_query, o_way, o_num_way, Outlier_num)
            aux_num = Outlier_num
            acc_train, f1_train = train(ID_class_selected, id_support, id_query, OOD_sample, n_way, k_shot, aux_num)
            meta_train_acc.append(acc_train)
            meta_train_f1.append(f1_train)
            if episode > 0 and episode % 100 == 0:
                meta_test_acc = []
                meta_test_f1 = []
                for idx in range(meta_test_num):
                    id_support, id_query, ID_class_selected = test_pool[idx]
                    acc_test, f1_test= test(ID_class_selected, id_support, id_query, n_way, k_shot, episode)
                    meta_test_acc.append(acc_test)
                    meta_test_f1.append(f1_test)
