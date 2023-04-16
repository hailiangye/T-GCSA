# -*- coding: utf-8 -*-
import numpy as np
import scipy.sparse as sp
import torch
import random
import argparse
import os
import warnings
warnings.filterwarnings("ignore")
import process
import aug
from gcn import GCNLayer
from GTCLN import GTCLN
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from process import sparse_mx_to_torch_sparse_tensor
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import time


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


parser = argparse.ArgumentParser()

parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--data', type=str, default='cora')
parser.add_argument('--runs', type=int, default=1)            
parser.add_argument('--eval_every', type=int, default=1)
parser.add_argument('--epochs', type=int, default=800)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--sample_size', type=int, default=2000)
parser.add_argument('--patience', type=int, default=10000)
parser.add_argument('--sparse', type=str_to_bool, default=True) 

parser.add_argument('--input_dim', type=int, default=1433) 
parser.add_argument('--gnn_dim', type=int, default=512) 
parser.add_argument('--proj_dim', type=int, default=512)
parser.add_argument('--proj_hid', type=int, default=4096)
parser.add_argument('--pred_dim', type=int, default=512)
parser.add_argument('--pred_hid', type=int, default=4096)
parser.add_argument('--momentum', type=float, default=0.8)
parser.add_argument('--alpha', type=float, default=0.05)
parser.add_argument('--gamma', type=float, default=0.44)
parser.add_argument('--drop_edge_Base', type=float, default=0.2)
parser.add_argument('--drop_edge_Student', type=float, default=0.2)
parser.add_argument('--drop_feat_Base', type=float, default=0.5)
parser.add_argument('--drop_feat_Student', type=float, default=0.5)
parser.add_argument('--drop_feat_Expand', type=float, default=0.5)

args = parser.parse_args()
torch.set_num_threads(4)


def evaluation(epoch, adj, diff, feat, gnn, idx_train, idx_test, sparse):
    clf = LogisticRegression(random_state=0, max_iter=2000) 
    model = GCNLayer(input_size, gnn_output_size)  
    model.load_state_dict(gnn.state_dict()) 
    
    with torch.no_grad():
        embeds1 = model(feat, adj, sparse) 
        embeds2 = model(feat, diff, sparse) 
        train_embs = embeds1[0, idx_train] + embeds2[0, idx_train] 
        test_embs = embeds1[0, idx_test] + embeds2[0, idx_test] 
        train_labels = torch.argmax(labels[0, idx_train], dim=1)
        test_labels = torch.argmax(labels[0, idx_test], dim=1) 

    clf.fit(train_embs, train_labels) 
    pred_test_labels = clf.predict(test_embs) 
    return accuracy_score(test_labels, pred_test_labels) 

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


if __name__ == '__main__':

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    n_runs = args.runs  
    eval_every_epoch = args.eval_every 

    dataset = args.data
    input_size = args.input_dim

    gnn_output_size = args.gnn_dim 
    projection_size = args.proj_dim 
    projection_hidden_size = args.proj_hid 
    prediction_size = args.pred_dim 
    prediction_hidden_size = args.pred_hid 
    momentum = args.momentum 
    alpha = args.alpha 
    gamma = args.gamma

    drop_edge_Base_rate = args.drop_edge_Base
    drop_edge_Student_rate = args.drop_edge_Student 
    drop_feature_rate_Base = args.drop_feat_Base  
    drop_feature_rate_Student = args.drop_feat_Student
    drop_feature_rate_Expand = args.drop_feat_Expand

    epochs = args.epochs 
    lr = args.lr 
    weight_decay = args.weight_decay
    sample_size = args.sample_size 
    batch_size = args.batch_size
    patience = args.patience 

    sparse = args.sparse

    # Loading dataset
    adj, features, labels, idx_train, idx_val, idx_test = process.load_data(dataset)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    if os.path.exists('data/diff_{}_{}.npy'.format(dataset, alpha)): 
        diff = np.load('data/diff_{}_{}.npy'.format(dataset, alpha), allow_pickle=True)
    else: 
        diff = aug.gdc(adj, alpha=alpha, eps=0.0001)
        np.save('data/diff_{}_{}'.format(dataset, alpha), diff)

    features, _ = process.preprocess_features(features)

    nb_nodes = features.shape[0] 
    ft_size = features.shape[1] 
    nb_classes = labels.shape[1] 

    features = torch.FloatTensor(features[np.newaxis])
    labels = torch.FloatTensor(labels[np.newaxis]) 
    
    norm_adj = process.normalize_adj(adj + sp.eye(adj.shape[0])) 
    norm_diff = sp.csr_matrix(diff)
    if sparse:
        eval_adj = process.sparse_mx_to_torch_sparse_tensor(norm_adj)  
        eval_diff = process.sparse_mx_to_torch_sparse_tensor(norm_diff)
    else:
        eval_adj = (norm_adj + sp.eye(norm_adj.shape[0])).todense()
        eval_diff = (norm_diff + sp.eye(norm_diff.shape[0])).todense()
        eval_adj = torch.FloatTensor(eval_adj[np.newaxis])
        eval_diff = torch.FloatTensor(eval_diff[np.newaxis])

    result_over_runs = [] 
    
    # Initiate models
    model = GCNLayer(input_size, gnn_output_size)
    node_score = aug.Node_Score(ft_size)
    feat_score = aug.Feature_Score(sample_size)
    GTCLN = GTCLN(gnn=model,
                  node_score_net=node_score,
                  feat_score_net=feat_score,
                  node_num=sample_size,
                  feat_size=input_size,
                  gamma=gamma,
                  projection_size=projection_size,
                  projection_hidden_size=projection_hidden_size,
                  prediction_size=prediction_size,
                  prediction_hidden_size=prediction_hidden_size,
                  moving_average_decay=momentum).to(device)
    para_num_dict= get_parameter_number(model)


    opt = torch.optim.Adam(GTCLN.parameters(), lr=lr, weight_decay=weight_decay)

    results = []

    
    best = 0
    patience_count = 0
    for epoch in tqdm(range(epochs)):
        for _ in range(batch_size):
            t0 = time.time()
            idx = np.random.randint(0, adj.shape[-1] - sample_size + 1) 
            ba = adj[idx: idx + sample_size, idx: idx + sample_size] 
            features = features.squeeze(0) 
            bf = features[idx: idx + sample_size] 
            bd = diff[idx: idx + sample_size, idx: idx + sample_size] 
            adj_Expand = sp.csr_matrix(np.matrix(bd)) 
            
            adj_oricoo = ba.tocoo()
            Apt = sparse_mx_to_torch_sparse_tensor(adj_oricoo).to(device)           
            Apt = Apt.to_dense()         
            feat_ori = bf.to(device)
            adj_ori = ba

            opt.zero_grad()
            loss = GTCLN(adj_ori, Apt, feat_ori, adj_Expand, drop_edge_Base_rate, drop_edge_Student_rate, drop_feature_rate_Base, 
                drop_feature_rate_Student, drop_feature_rate_Expand, sparse) 
            loss.backward()
            opt.step() 
            GTCLN.teacherB_update_ma() 
            GTCLN.teacherE_update_ma()
            t1 = time.time()
            triantime = t1-t0
            # print(triantime)

        if epoch % eval_every_epoch == 0: 
            acc = evaluation(epoch, eval_adj, eval_diff, features, model, idx_train, idx_test, sparse) 
            if acc > best:
                best = acc
                patience_count = 0
            else:
                patience_count += 1
            results.append(acc) 
            print('\t epoch {:03d} | loss {:.5f} | clf test acc {:.5f}'.format(epoch, loss.item(), acc))
            if patience_count >= patience: 
                print('Early Stopping.')
                break
        
            
    result_over_runs.append(max(results))
    print('\t best acc {:.5f}'.format(max(results))) 