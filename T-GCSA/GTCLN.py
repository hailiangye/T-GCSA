import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
import aug
import process
import scipy.sparse as sp
import numpy as np
import argparse

class MLP(nn.Module): 

    def __init__(self, inp_size, outp_size, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.PReLU(),
            nn.Linear(hidden_size, outp_size)
        )

    def forward(self, x):
        return self.net(x)


class GraphEncoder(nn.Module): 

    def __init__(self, 
                  gnn,
                  projection_hidden_size,
                  projection_size):
        
        super().__init__()
        
        self.gnn = gnn
        self.projector = MLP(512, projection_size, projection_hidden_size)           
        
    def forward(self, adj, in_feats, sparse):
        representations = self.gnn(in_feats, adj, sparse)
        representations = representations.view(-1, representations.size(-1)) 
        projections = self.projector(representations)  
        return projections

    
class EMA():
    
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new 


def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight) 


def set_requires_grad(model, val): 
    for p in model.parameters():
        p.requires_grad = val


def sim(h1, h2): 
    z1 = F.normalize(h1, dim=-1, p=2) 
    z2 = F.normalize(h2, dim=-1, p=2)
    return torch.mm(z1, z2.t()) 


def similar_loss(h1, h2):
    f = lambda x: torch.exp(x)
    cross_sim = f(sim(h1, h2))
    return -torch.log(cross_sim.diag() / cross_sim.sum(dim=-1))
                

class GTCLN(nn.Module):
    
    def __init__(self,
                 gnn,
                 node_score_net,
                 feat_score_net,
                 node_num,
                 feat_size,
                 gamma,
                 projection_size, 
                 projection_hidden_size,
                 prediction_size,
                 prediction_hidden_size,
                 moving_average_decay):
        
        super().__init__()
        
        self.node_score_net = node_score_net
        self.feat_score_net = feat_score_net
        
        self.Student_encoder = GraphEncoder(gnn, projection_hidden_size, projection_size) #student
        self.teacherB_encoder = copy.deepcopy(self.Student_encoder)
        self.teacherE_encoder = copy.deepcopy(self.Student_encoder)
        set_requires_grad(self.teacherB_encoder, False) 
        set_requires_grad(self.teacherE_encoder, False) 
        self.teacherB_ema_updater = EMA(moving_average_decay)
        self.teacherE_ema_updater = EMA(moving_average_decay)
        self.Student_predictor = MLP(projection_size, prediction_size, prediction_hidden_size) 
        
        self.gamma = gamma
        
    def teacherB_reset_moving_average(self):
        del self.teacherB_encoder
        self.teacherB_encoder = None
        
    def teacherE_reset_moving_average(self):
        del self.teacherE_encoder
        self.teacherE_encoder = None
        
    def teacherB_update_ma(self):
        assert self.teacherB_encoder is not None, 'teacherB encoder has not been created yet'
        update_moving_average(self.teacherB_ema_updater, self.teacherB_encoder, self.Student_encoder) 
        
    def teacherE_update_ma(self):
        assert self.teacherE_encoder is not None, 'teacherE encoder has not been created yet'
        update_moving_average(self.teacherE_ema_updater, self.teacherE_encoder, self.Student_encoder) 
        
    def forward(self, adj_ori, Apt, feat_ori, adj_Expand, drop_edge_Base_rate, drop_edge_Student_rate, drop_feature_rate_Base, 
                drop_feature_rate_Student, drop_feature_rate_Expand, sparse):
        
        device = torch.device('cuda')
        
        node_score = self.node_score_net(feat_ori, Apt)
        edge_score = aug.Edge_Score(node_score, adj_ori)
        feat_score = self.feat_score_net(feat_ori, Apt)
        
        #student_input
        adj_Student = aug.aug_student_edge(adj_ori, Apt, drop_edge_Student_rate, edge_score)
        Student_feat = aug.aug_student_feat(feat_ori, Apt, adj_ori, drop_feature_rate_Student, feat_score)
        #teacher_B_input
        adj_Base = aug.aug_base_edge(adj_ori, Apt, drop_edge_Base_rate, edge_score)
        aug_Base_feat = aug.aug_feat(feat_ori, Apt, adj_ori, drop_feature_rate_Base, feat_score)
        #teacher_E_input
        adj_Expand = adj_Expand
        aug_Expand_feat = aug.aug_feat(feat_ori, Apt, adj_ori, drop_feature_rate_Expand, feat_score)
        
        
        adj_Base = process.normalize_adj(adj_Base + sp.eye(adj_Base.shape[0])) 
        adj_Student = process.compute_normalized_laplacian(adj_Student + torch.eye(adj_Student.shape[0])).to(device)
        adj_Expand = process.normalize_adj(adj_Expand + sp.eye(adj_Expand.shape[0]))

        if sparse: 
            adj_Base = process.sparse_mx_to_torch_sparse_tensor(adj_Base).to(device)
            adj_Expand = process.sparse_mx_to_torch_sparse_tensor(adj_Expand).to(device)
        else:
            aug_adj_Base = (adj_Base + sp.eye(adj_Base.shape[0])).todense()
            aug_adj_Expand = (adj_Expand + sp.eye(adj_Expand.shape[0])).todense()
            adj_Base = torch.FloatTensor(aug_adj_Base[np.newaxis]).to(device)
            adj_Expand = torch.FloatTensor(aug_adj_Expand[np.newaxis]).to(device)
        
        aug_Base_feat = aug_Base_feat.to(device)
        Student_feat = Student_feat.to(device)
        aug_Expand_feat = aug_Expand_feat.to(device)
        
        Student_proj = self.Student_encoder(adj_Student, Student_feat, sparse)
        
        Student_pred = self.Student_predictor(Student_proj)
        
        with torch.no_grad():
            teacherB_proj = self.teacherB_encoder(adj_Base, aug_Base_feat, sparse)
 
            teacherE_proj = self.teacherE_encoder(adj_Expand, aug_Expand_feat, sparse)
            
        loss1 = similar_loss(Student_pred, teacherB_proj.detach())
        
        loss2 = similar_loss(Student_pred, teacherE_proj.detach())
        
        P = torch.rand(Student_pred.shape[0], Student_pred.shape[1]).cuda()
        Ones = torch.ones_like(P)
        total_test = P*teacherB_proj.detach() + (Ones-P)*teacherE_proj.detach()    
        loss3 = similar_loss(Student_pred, total_test)
        
        loss = self.gamma * (loss1 + loss2) + (1-2*self.gamma)*loss3
        
        return loss.mean()
