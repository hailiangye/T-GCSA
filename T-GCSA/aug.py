import torch
import torch.nn as nn
import copy
import random
import scipy.sparse as sp
import numpy as np
from torch_geometric.nn import GCNConv
from gcn import feat_self_attention, GCNLayer
from torch_geometric.nn.pool.topk_pool import topk
from process import sparse_mx_to_torch_sparse_tensor

class Node_Score(nn.Module):
    def __init__(self, in_channels, Conv=GCNLayer):
        super(Node_Score, self).__init__()
        self.in_channels = in_channels
        self.score_layer = Conv(in_channels,1)
        
    def forward(self, x, adj, edge_attr=None, batch=None):
        node_score = self.score_layer(x,adj).squeeze()
        return node_score
    
def Edge_Score(node_score, adj):
    
    row_idx, col_idx = adj.nonzero()    
    edge_score = (node_score[row_idx] + node_score[col_idx])/2
    
    return edge_score

class Feature_Score(nn.Module):
    def __init__(self, in_channels, Conv=feat_self_attention):
        super(Feature_Score, self).__init__()
        self.in_channels = in_channels
        self.score_layer = Conv(in_channels,1)
        
    def forward(self, x, adj, edge_attr=None, batch=None):
        feat_score = self.score_layer(x,adj).squeeze()
        return feat_score
    
def aug_student_edge(adj, Aptt, drop_percent, edge_score, topk_ratio=0.5, batch=None):
    if batch is None:
        batch = Aptt.new_zeros(edge_score.size(0))
        
    percent = drop_percent
    row_idx, col_idx = adj.nonzero()
    num_drop = int(len(row_idx)*percent)
    edge_index = [i for i in range(len(row_idx))]
    edges = dict(zip(edge_index, zip(row_idx, col_idx)))
    perm = topk(edge_score, topk_ratio, batch.to(dtype=torch.int64)) 
    low_score_edge_index = list(set(edge_index)-set(perm))
    
    drop_idx = random.sample(low_score_edge_index, k = num_drop)
    list(map(edges.__delitem__, filter(edges.__contains__, drop_idx))) 
    
    
    new_edges = list(zip(*list(edges.values())))
    new_row_idx = new_edges[0]
    new_col_idx = new_edges[1]
    data = np.ones(len(new_row_idx)).tolist()
    
    new_adj = sp.csr_matrix((data, (new_row_idx, new_col_idx)), shape = adj.shape) 

    new_adjcoo = new_adj.tocoo()    
    Apt = sparse_mx_to_torch_sparse_tensor(new_adjcoo)   
    adj_aug = torch.tensor(Apt.to_dense())    
    B = (torch.tensor(adj_aug[row_idx, col_idx]).squeeze(0))
    if edge_score.max() > 2:
        adj_aug[row_idx, col_idx] = B*torch.exp(edge_score.cpu()/torch.norm(edge_score.cpu()))
    else:
        adj_aug[row_idx, col_idx] = B*torch.exp(edge_score.cpu())
        
    return adj_aug
    
def aug_base_edge(adj, Aptt, drop_percent, edge_score, topk_ratio=0.5, batch=None):
    if batch is None:
        batch = Aptt.new_zeros(edge_score.size(0))
        
    percent = drop_percent
    row_idx, col_idx = adj.nonzero()
    num_drop = int(len(row_idx)*percent)
    edge_index = [i for i in range(len(row_idx))]
    edges = dict(zip(edge_index, zip(row_idx, col_idx)))
    perm = topk(edge_score, topk_ratio, batch.to(dtype=torch.int64)) 
    low_score_edge_index = list(set(edge_index)-set(perm))
    
    drop_idx = random.sample(low_score_edge_index, k = num_drop)
    list(map(edges.__delitem__, filter(edges.__contains__, drop_idx))) 
    
    
    new_edges = list(zip(*list(edges.values())))
    new_row_idx = new_edges[0]
    new_col_idx = new_edges[1]
    data = np.ones(len(new_row_idx)).tolist()
    
    new_adj = sp.csr_matrix((data, (new_row_idx, new_col_idx)), shape = adj.shape) 

    return new_adj
    
def aug_student_feat(x, Aptt, adj, drop_percent, feat_score, topk_ratio=0.2, batch=None):
    input_feat = copy.deepcopy((x.squeeze(0))) 
    if batch is None:
        batch = Aptt.new_zeros(input_feat.size(1))
        
    feat_index = [i for i in range(len(feat_score))]
    perm = topk(feat_score, topk_ratio, batch.to(dtype=torch.int64))
    low_score_feat_index = list(set(feat_index)-set(perm))
        

    drop_feat_num = int(input_feat.shape[1] * drop_percent) 
    drop_idx = random.sample(low_score_feat_index, drop_feat_num)  
    input_feat[:, drop_idx] = 0 
    input_feat = input_feat.t() * torch.tanh(feat_score).view(-1, 1)
    input_feat = input_feat.t()
    return input_feat

def aug_feat(x, Aptt, adj, drop_percent, feat_score, topk_ratio=0.2, batch=None):
    input_feat = copy.deepcopy((x.squeeze(0))) 
    if batch is None:
        batch = Aptt.new_zeros(input_feat.size(1))
        
    feat_index = [i for i in range(len(feat_score))]
    perm = topk(feat_score, topk_ratio, batch.to(dtype=torch.int64))
    low_score_feat_index = list(set(feat_index)-set(perm))
        

    drop_feat_num = int(input_feat.shape[1] * drop_percent) 
    drop_idx = random.sample(low_score_feat_index, drop_feat_num) 
    input_feat[:, drop_idx] = 0 
    
    return input_feat

def gdc(A: sp.csr_matrix, alpha: float, eps: float):
    N = A.shape[0]
    A_loop = sp.eye(N) + A
    D_loop_vec = A_loop.sum(0).A1
    D_loop_vec_invsqrt = 1 / np.sqrt(D_loop_vec)
    D_loop_invsqrt = sp.diags(D_loop_vec_invsqrt)
    T_sym = D_loop_invsqrt @ A_loop @ D_loop_invsqrt
    S = alpha * sp.linalg.inv(sp.eye(N) - (1 - alpha) * T_sym)
    S_tilde = S.multiply(S >= eps)
    D_tilde_vec = S_tilde.sum(0).A1
    T_S = S_tilde / D_tilde_vec
    return T_S

