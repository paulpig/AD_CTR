from lzma import CHECK_ID_MAX
from threading import local
from turtle import pos
import torch
from torch import nn
import numpy as np
# from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import EmbeddingDictLayer, MLP_Layer, Dice
import random
from time import time
import math
from ..torch_utils import get_device, get_optimizer, get_loss_fn, get_regularizer
import logging
import scipy.sparse as sp
import copy
import pdb
import pickle
import h5py
import torch.nn.functional as F
from .VAE import VAE
from fuxictr.pytorch.layers import MLP_Layer, EmbeddingLayer, SqueezeExcitationLayer, BilinearInteractionLayer, LR_Layer

# attritbue as label to model

torch.backends.cudnn.enabled = True 
torch.backends.cudnn.benchmark = True


class HyperGraphCustomBipartiteDisenGATVAEV3CTRObjSameIdxHyperGraph(nn.Module):
    def __init__(self,
                #  adj_user_mat,
                 gpu=-1,
                 graph_layer=1,
                 learning_rate=1e-3, 
                 graph_embedding_dim=10, 
                 epoch_pre=100,
                 epoch=100,
                 bpr_batch_size=10000,
                 weight_decay=1e-5,
                 iterations=1,
                 add_norm=False,
                 channels=2,
                 cl_w=1.0e-5,
                 add_vae=False):
        super(HyperGraphCustomBipartiteDisenGATVAEV3CTRObjSameIdxHyperGraph, self).__init__()
        
        # parameters
        self.emb_size = graph_embedding_dim
        self.layers = graph_layer
        # self.dataset = datasets
        self.cl_w = cl_w
        self.device = get_device(gpu)
       
        # path_origin = "../data/Taobao/taobao_ori/u_c_pos_mat_new_idx.npz" #3 day train data
        path_origin = "../data/Taobao/taobao_ori/u_c_pos_mat_new_idx_1_day.npz" #3 day train data


        adj_mat = sp.load_npz(path_origin)
        self.item_voc_len, self.user_voc_len = adj_mat.get_shape() #(item, user)
        self.adj_mat = adj_mat #(item, user)

        # path = "../data/Taobao/taobao_ori/u_c_pos_mat_pre_norm_new_idx.npz" #3 day
        # path = "../data/Taobao/taobao_ori/u_c_pos_mat_pre_norm_new_idx_add_selfnode.npz" #3 day + add self node;
        path = "../data/Taobao/taobao_ori/u_c_pos_mat_pre_norm_new_idx_1_day.npz" #1 day

        # self.inter_path = "../data/Taobao/taobao_ori/train_data_u_clist_new_idx.npz" #3 day
        self.inter_path = "../data/Taobao/taobao_ori/train_data_u_clist_new_idx_1_day.npz" #1 day
        
        # self.path_dict_path = "../data/Taobao/taobao_ori/u2id_c2id.pk" # 存在ID与index的字典, <user_id, index>, <item_id, index>        
        
        # self.path_pretrain_user_emb_path = "../data/Taobao/taobao_ori/u_c_pos_emb_new_idx.h5" # 3 days
        self.path_pretrain_user_emb_path = "../data/Taobao/taobao_ori/u_c_pos_emb_new_idx_1_day.h5" # 1 day
        # 这个路径不要变;

        # user-attribute hypergraph;
        # self.u_at_hypergraph_path = "../data/Taobao/taobao_ori/u_at_hypergraph_1_day.npz"
        self.u_at_hypergraph_path = "../data/Taobao/taobao_ori/u_at_hypergraph_1_day_rm_lfreq.npz"
        self.save_pre_norm_u_at_hypergraph_path = "../data/Taobao/taobao_ori/pre_norm_u_at_hypergraph_1_day_rm_lfreq.npz"
        self.u_at_adj_mat = sp.load_npz(self.u_at_hypergraph_path)
        self.u_at_voc_len, _ = self.u_at_adj_mat.get_shape()
        self.u_at_adj_norm_mat = self.get_ui_bipartite_adj_mat(self.u_at_adj_mat, 
                                                                self.save_pre_norm_u_at_hypergraph_path, 
                                                                user_voc=self.user_voc_len,
                                                                item_voc=self.u_at_voc_len,
                                                                add_self=True) #(user+item, user+item)

        self.target_user_list, self.pos_items_list = self.generate_train_data()
        self.t_c_list, self.pos_u_list= self.generate_train_data_c2u()

        labels = torch.tensor(list(set(self.target_user_list)))
        labels = labels.unsqueeze(0)
        self.multi_hot_label = torch.zeros(labels.size(0), self.user_voc_len).scatter_(1, labels, 1.).to(self.device)
        # pdb.set_trace()

        self.occur_item_voc_list = []
        for pos_item_list in self.pos_items_list:
            self.occur_item_voc_list.extend(pos_item_list)
        
        self.occur_item_voc_set = list(set(self.occur_item_voc_list))
        # y_onehot = nn.functional.one_hot( torch.LongTensor(self.target_user_list), num_classes=self.user_voc_len)
        # y_onehot = y_onehot.sum(dim=0).float()

        # pdb.set_trace()
        self.ui_mat = self.get_ui_bipartite_adj_mat(self.adj_mat, path, user_voc=self.user_voc_len, item_voc=self.item_voc_len, add_self=True) #(user+item, user+item)
        
        
        self.channels = channels
        self.c_dim = self.emb_size // self.channels

        self.weight_list = nn.ParameterList(
            nn.Parameter(torch.empty(size=(self.emb_size, self.c_dim), dtype=torch.float), requires_grad=True) for i in
            range(self.channels))
        self.bias_list = nn.ParameterList(
            nn.Parameter(torch.empty(size=(1, self.c_dim), dtype=torch.float), requires_grad=True) for i in
            range(self.channels))
        
        # self.weight_list = nn.ModuleList([nn.Linear(self.emb_size, self.c_dim, bias=True) for _ in range(self.channels)])
        # self.linear_1 = nn.Linear(self.in_dim, self.channels*self.c_dim, bias=True)
        self.iterations = iterations
        # self.A_in_list =  nn.ParameterList(
        #     nn.Parameter(torch.sparse.FloatTensor(self.user_voc_len + self.item_voc_len, self.user_voc_len + self.item_voc_len)) for i in 
        #     range(self.channels))
        self.A_in_list = [nn.Parameter(torch.sparse.FloatTensor(self.user_voc_len + self.item_voc_len, self.user_voc_len + self.item_voc_len)) for i in 
            range(self.channels)]
        self.A_in_list = nn.ParameterList(self.A_in_list)
        
        # 初始化;
        if self.ui_mat is not None:
            for i in range(self.channels):
                # self.A_in_list[i].data = self._convert_sp_mat_to_sp_tensor(self.ui_mat).coalesce().to(device=self.device)
                self.A_in_list[i].data = self._convert_sp_mat_to_sp_tensor(self.ui_mat).coalesce()
                self.A_in_list[i].requires_grad = False
                # self.A_in.data = self._convert_sp_mat_to_sp_tensor(self.ui_mat.coalesce().to(self.device))

        #get row indices and col indices
        self.row_indices = self.A_in_list[0].indices()[0]
        self.col_indices = self.A_in_list[0].indices()[1]
        
        self.embedding_user = nn.Embedding(self.user_voc_len, self.emb_size)
        self.embedding_item = nn.Embedding(self.item_voc_len, self.emb_size)

        # pdb.set_trace()
        self.weight_lightgcn = nn.Parameter(torch.empty(size=(self.emb_size,  self.emb_size), dtype=torch.float), requires_grad=True)
        self.bias_lightgcn = nn.Parameter(torch.empty(size=(1, self.emb_size), dtype=torch.float), requires_grad=True)
        self.merge_id_feat_mlp = nn.Parameter(torch.empty(size=(self.emb_size * 2,  self.emb_size), dtype=torch.float), requires_grad=True)

        self.weight_lightgcn_list = nn.ParameterList(
            nn.Parameter(torch.empty(size=(self.emb_size, self.emb_size), dtype=torch.float), requires_grad=True) for i in
            range(self.layers))
        self.bias_lightgcn_list = nn.ParameterList(
            nn.Parameter(torch.empty(size=(1, self.emb_size), dtype=torch.float), requires_grad=True) for i in
            range(self.layers))
        # =========================================
        with open('../data/Taobao/taobao_ori/u2_uat_list_new_idx.pk', "rb") as f: #与CTR模型编码一致;
            self.userid_atlist = pickle.load(f) # (n_num, uat_num)

        self.user_attris = np.array(list(self.userid_atlist.values()))
        self.user_keys = np.array(list(self.userid_atlist.keys()))
        # self.user_attri_len = self.user_attris.shape[1]
        self.each_u_at_num = self.user_attris.max(axis=0) #(7)
        # self.user_at_dim = self.emb_size // 2
        self.user_at_dim = self.emb_size // 1

        embedding_user_at_list = []
        for i in range(len(self.each_u_at_num)):
            embedding_user_at_list.append(nn.Embedding(self.each_u_at_num[i] + 1, self.user_at_dim))
        self.embedding_user_at_list = nn.ModuleList(embedding_user_at_list)
        
        # self.embedding_u_at = nn.Embedding(self.u_at_voc_len, self.emb_size)
        # pdb.set_trace()

        #=========vae模型============================
        self.user_vae = VAE(self.user_at_dim, device=self.device)

        self.u_at_mlps = nn.ModuleList([nn.Linear(self.user_at_dim, self.user_at_dim) for _ in range(len(self.each_u_at_num))])
        # self.deep_vae_mlp = nn.Linear(self.user_at_dim *len(self.each_u_at_num), self.emb_size)
        
        self.dense_user_self_biinter = nn.Linear(self.user_at_dim, self.user_at_dim)
        self.dense_user_onehop_siinter = nn.Linear(self.user_at_dim, self.user_at_dim)
        # self.dense_user_self_siinter = nn.Linear(self.user_at_dim, self.user_at_dim)

        self.merge_user_at = nn.Linear(self.user_at_dim + self.emb_size, self.emb_size)

        self.leakyrelu = nn.LeakyReLU()

        self.senet_layer = SqueezeExcitationLayer(len(self.each_u_at_num), 3)

        self.convert_mlp = nn.Linear(self.user_at_dim*len(self.each_u_at_num) + self.emb_size, self.emb_size)
        #====================================


        # self.compile(kwargs["optimizer"], loss=kwargs["loss"], lr=learning_rate)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        # self.optimizer_vae = torch.optim.Adam(self.parameters(), lr=learning_rate)

        # para_names = []
        # for n, p in self.named_parameters():
        #     para_names.append(n)
            # if 'vae' in n:
            #     pdb.set_trace()

        # print(list(self.named_parameters()))
        # pdb.set_trace()

        self.epoch_pre = epoch_pre
        self.epoch = epoch
        self.folds = 20
        # self.folds = 1000
        self.weight_decay = weight_decay
        # self.kwargs = kwargs
        self.bpr_batch_size = bpr_batch_size
        self.init_parameters() # replace other ways
        # self.init_parameters_v2()

        self.to(device=self.device)
        self.user_attri_emb = self.get_user_ats_for_train(device=self.device)
        self.add_mlp = False
        self.add_vae = add_vae

    def get_user_ats_for_train(self, device):
        #重新排序
        self.user_attri_ids = [] #(n_user, 7)
        for u_idx in range(len(self.embedding_user.weight)):
            if u_idx in self.userid_atlist:
                user_attris_val = self.userid_atlist[u_idx] #(7)
            else:
                user_attris_val = [0 for _ in range(len(self.each_u_at_num))] # padding is zero: right;
            
            self.user_attri_ids.append(user_attris_val)
        
        self.user_attri_ids = np.array(self.user_attri_ids) #(n_user, 7)

        # convert tensor
        self.user_attri_ids = torch.from_numpy(self.user_attri_ids).to(device=device)

        # pdb.set_trace()
        self.user_attri_emb = []
        for idx in range(len(self.embedding_user_at_list)):
            user_at_emb = self.embedding_user_at_list[idx](self.user_attri_ids[:, idx]) #(n_user, dim)
            self.user_attri_emb.append(user_at_emb)
        
        # self.user_attri_emb = torch.stack(self.user_attri_emb, dim=1).to(device=self.device) #(n_user, 7, dim)
        self.user_attri_emb = torch.stack(self.user_attri_emb, dim=1) #(n_user, 7, dim)
        # self.user_attri_emb.to(device=self.device)

        return self.user_attri_emb

    def generate_train_data_c2u(self):
        with open(self.inter_path, "rb") as r_f:
            user_items_d = pickle.load(r_f)
            print("successfully loading train data...")
        
        c2ulist = {}
        c_target = []
        pos_u_list = []
        for user_item_list in user_items_d.items():
            # target_user_list.append(user_item_list[0])
            # pos_items_list.append(user_item_list[1])
            u_idx = user_item_list[0]
            for c_item in user_item_list[1]:
                if c_item not in c2ulist:
                    c2ulist[c_item] = [u_idx]
                else:
                    c2ulist[c_item].append(u_idx)
        
        for c_idx, u_idx_list in c2ulist.items():
            c_target.append(c_idx)
            pos_u_list.append(u_idx_list)


        return c_target, pos_u_list

    
    def generate_train_data(self):
        with open(self.inter_path, "rb") as r_f:
            user_items_d = pickle.load(r_f)
            print("successfully loading train data...")
        
        target_user_list = []
        pos_items_list = []
        for user_item_list in user_items_d.items():
            target_user_list.append(user_item_list[0])
            pos_items_list.append(user_item_list[1])
            # target_user_list.extend(user_item_list[1])
            # pos_items_list.append(user_item_list[0])

        return target_user_list, pos_items_list


    def generate_train_data_for_train_c2u_add_uat(self, voc_len):

        # target_item_id_list, pos_users_list, neg_users_list = [], [], []
        t_c_list_result, pos_u_list_result, neg_u_list_result = [], [], []
        pos_uat_list_result, neg_uat_list_result = [], []

        assert len(self.t_c_list) == len(self.pos_u_list)

        for t_c, pos_u_list in zip(self.t_c_list, self.pos_u_list):
            # pos_u_idx = np.random.choice(pos_u_list, 1, replace=False)[0]
            for pos_u_idx in pos_u_list: #遍历已存在的所有边;
                if pos_u_idx in self.userid_atlist:
                    pos_u_ats = self.userid_atlist[pos_u_idx] #(7)
                else:
                    pos_u_ats = [0 for _ in range(len(self.each_u_at_num))] #(7)
                
                # select negative users
                while True:
                    # neg_idx = np.random.randint(0, self.user_voc_len)
                    neg_u_idx = np.random.randint(0, voc_len)
                    # neg_idx = np.random.choice(self.occur_item_voc_set)
                    if neg_u_idx in pos_u_list:
                        continue
                    else:
                        break
                

                if neg_u_idx in self.userid_atlist:
                    neg_u_ats = self.userid_atlist[neg_u_idx] #(7)
                else:
                    neg_u_ats = [0 for _ in range(len(self.each_u_at_num))] #(7)
                
                t_c_list_result.append(t_c)
                pos_u_list_result.append(pos_u_idx)
                neg_u_list_result.append(neg_u_idx)
                pos_uat_list_result.append(pos_u_ats)
                neg_uat_list_result.append(neg_u_ats)
        
        return t_c_list_result, pos_u_list_result, neg_u_list_result, pos_uat_list_result, neg_uat_list_result

    def generate_train_data_for_train(self, voc_len):

        target_item_id_list, pos_users_list, neg_users_list = [], [], []

        assert len(self.target_user_list) == len(self.pos_items_list)

        for t_user, pos_items_list in zip(self.target_user_list, self.pos_items_list):
            pos_user_id = np.random.choice(pos_items_list, 1, replace=False)[0]

            # select negative users
            while True:
                # neg_idx = np.random.randint(0, self.user_voc_len)
                neg_idx = np.random.randint(0, voc_len)
                # neg_idx = np.random.choice(self.occur_item_voc_set)
                if neg_idx in pos_items_list:
                    continue
                else:
                    break
            
            neg_user_id = neg_idx
            target_item_id_list.append(t_user)
            pos_users_list.append(pos_user_id)
            neg_users_list.append(neg_user_id)
        
        return target_item_id_list, pos_users_list, neg_users_list

    def init_parameters(self):
        stdv = 1.0 / math.sqrt(self.emb_size)
        # stdv = 1.0 / self.emb_size
        for weight in self.parameters():
            if weight.requires_grad == True:
                weight.data.uniform_(-stdv, stdv)
    
    def init_parameters_v2(self, std_val=1e-4):
        # stdv = 1.0 / math.sqrt(self.emb_size)
        for weight in self.parameters():
            # weight.data.uniform_(-stdv, stdv)
            if weight.requires_grad == True:
                # pdb.set_trace()
                weight.data.normal_(std=std_val)
            # nn.init.normal_(embedding_matrix.weight, std=std_val) #默认初始化;


    def countIndepentHyperEdgeNum(self):
        """
        1_day: 46%左右是出现过的;
        """
        # self.adj_mat: (I, U)
        self.adj_mat = self.adj_mat.T #(U, I)
        A = self.adj_mat.T #(user, item)
        user_mat = A.sum(axis=1)

        B = self.adj_mat.tolil() #(item, user)
        indenpent_hyperedge = 0
        none_count = 0 
        for index, user_list in enumerate(B):
            # pdb.set_trace()
            user_idx = np.nonzero(user_list)[1] #user index

            if len(user_idx) == 0:
                none_count += 1
                continue
            
            # pdb.set_trace()
            if int(user_mat[user_idx].sum()) == len(user_idx):
                indenpent_hyperedge += 1
        pdb.set_trace()
        return

    def disentangleComputer(self, adjacency_list, user_emb, item_emb, mode='train'):
        """
        adjacency_list: (n_user + n_item, n_user + n_item)
        """
        # item_embeddings = embedding
        # final = [torch.cat([user_emb, item_emb], dim=0)]

        if mode == "train":
            self.iterations_valid = self.iterations
        elif mode == "predict":
            self.iterations_valid = 1
        
        # self.after_mlp = []
        all_emb = torch.cat([user_emb, item_emb], dim=0)
        all_channel_emb_list = []
        for c_i in range(self.channels):  
            # z = torch.matmul(all_emb, self.weight_list[i]) + self.bias_list[i] #每层的参数设置相同;
            z = torch.matmul(all_emb, self.weight_list[c_i]) + self.bias_list[c_i] #每层的参数设置不同;
            # z = F.normalize(z, dim=1)
            all_channel_emb_list.append(z)
        

        final = [torch.cat(all_channel_emb_list, dim=-1)]

        for i in range(self.layers):
            # item_embeddings = torch.sparse.mm(self.trans_to_cuda(adjacency), item_embeddings)
            # pdb.set_trace() 
            # all_channel_emb_list = []
            # for c_i in range(self.channels):  
            #     # z = torch.matmul(all_emb, self.weight_list[i]) + self.bias_list[i] #每层的参数设置相同;
            #     z = torch.matmul(all_emb, self.weight_list[i*self.channels+c_i]) + self.bias_list[i*self.channels+c_i] #每层的参数设置不同;
            #     z = F.normalize(z, dim=1)
            #     all_channel_emb_list.append(z)
            
            # 如何尽可能要让表征之间是多样性的;
            for l in range(self.iterations_valid):
                item_embeddings_list = []
                start_idx, end_idx = 0, 0
                valid_ele_count = 0
                val_list_all = []
                count = 0
                for sub_adjacency in adjacency_list:
                    # pdb.set_trace()
                    end_idx = start_idx + sub_adjacency.shape[0]
                    batch_emb_list = [s_emb[start_idx: end_idx, :] for s_emb in all_channel_emb_list]
                    
                    if sub_adjacency.indices().shape[1] == 0:
                        batch_all_output_emb_tmp = torch.cat(batch_emb_list, dim=-1)
                        batch_all_output_emb = torch.zeros_like(batch_all_output_emb_tmp)
                    else:
                        # batch_all_output_emb = self.disentangledLearningOnLayer(sub_adjacency, batch_emb, all_emb, n=count)
                        batch_all_output_emb, val_list = self.disentangledLearningOnLayerGAT(sub_adjacency, batch_emb_list, all_channel_emb_list, start_index=valid_ele_count)
                        # batch_all_output_emb = self.disentangledLearningOnLayerTest(sub_adjacency, batch_emb_list, all_channel_emb_list, n=count)
                        # batch_all_output_emb = torch.cat(batch_emb_list, dim=-1)
                        val_list_all.append(val_list) # list of (3, M)
                        # pdb.set_trace()  # batch_all_output_emb.sum(1), torch.nonzero(batch_all_output_emb.sum(1))
                    start_idx = end_idx

                    valid_ele_count += len(sub_adjacency.indices()[0])
                    # add l2 normalize
                    # item_embeddings_sub = torch.nn.functional.normalize(item_embeddings_sub, p=2, dim=1)
                    item_embeddings_list.append(batch_all_output_emb)
                    # val_list_all.append(val_list) # list of (3, M)
                    count += 1
                    del batch_all_output_emb
                    torch.cuda.empty_cache()
                
                # pdb.set_trace()
                # update self.A_in
                update_A_val = torch.cat(val_list_all, dim=1) #(3, P), size 不对;

                # add previous vals
                pre_val_list = []
                for i in range(self.channels):
                    pre_val = self.A_in_list[i].coalesce().values()
                    pre_val_list.append(pre_val)
                
                # pdb.set_trace()
                # 删除的效果更好
                # update_A_val = torch.stack(pre_val_list, dim=0) + update_A_val

                # add softmax
                update_A_val = F.softmax(update_A_val, dim=0) # 在channel维度执行softmax;

                assert len(update_A_val[i]) == len(self.A_in_list[i].coalesce().indices()[0]) and valid_ele_count == len(update_A_val[i])
                
                # pdb.set_trace()
                for i in range(self.channels):
                    # pdb.set_trace()
                    # len(self.A_in_list[i].indices()[0])
                    # softmax on u_i axis
                    A_in = torch.sparse.FloatTensor(self.A_in_list[i].coalesce().indices(), update_A_val[i], self.A_in_list[i].shape).coalesce()

                    # A_in = torch.sparse.softmax(A_in, dim=1)
                    # lightGCN归一化
                    normalize = True
                    if normalize:
                        row_idx, col_idx = self.A_in_list[i].coalesce().indices()
                        # pdb.set_trace()
                        # row_idx, col_idx = row_idx.to_dense(), col_idx.to_dense() 
                        val = update_A_val[i]

                        row_sum = torch.sparse.sum(A_in, dim=-1)  # sum by row(tgt)
                        # col_sum = torch.sparse.sum(A_in, dim=0)  # sum by row(tgt)
                        # col_sum = A_in.sum(0, keepdim=True)  # sum by row(tgt)
                        row_val_p = torch.pow(row_sum, -0.5)

                        # torch.any(torch.isnan(col_val))
                        # torch.any(torch.isnan(row_val))
                        # torch.any(torch.isnan(val))
                        row_val_p = row_val_p.to_dense() #(n_user + n_item)
                        # replace inf
                        row_val_p = torch.where(torch.isinf(row_val_p), torch.zeros_like(row_val_p), row_val_p)

                        row_val, col_val = row_val_p[row_idx], row_val_p[col_idx]

                        val_norm = val * row_val * col_val
                        A_in = torch.sparse.FloatTensor(self.A_in_list[i].coalesce().indices(), val_norm, self.A_in_list[i].shape).coalesce()
                        if torch.any(torch.isnan(val_norm)) or torch.isinf(val_norm).any():
                            pdb.set_trace()
                            # torch.nonzero(torch.isnan(val_norm))
                        # torch.any(torch.isnan(val_norm))
                        # pdb.set_trace()
                        # torch.div(A_in, col_val_p)
                        # threshold on 1 to avoid div by 0
                        # torch.nn.functional.threshold(row_val_p, 0, 1, inplace=True)
                        # torch.nn.functional.threshold(col_val_p, 0, 1, inplace=True)
                        # pdb.set_trace()
                        # A_in.div_(col_val_p)
                        # A_in.div_(row_val_p)
                    # else:
                    #     A_in = torch.sparse.FloatTensor(self.A_in_list[i].coalesce().indices(), update_A_val[i], self.A_in_list[i].shape).coalesce()

                    #len(np.nonzero(self.ui_mat.sum(1))[0])
                    # col_val = torch.sparse.sum(A_in, dim=0) #(u+i)
                    # row_val = torch.sparse.sum(A_in, dim=1) #(u+i)
                    # col_val_p, row_val_p = torch.pow(col_val, 0.5), torch.pow(row_val, 0.5)
                    # pdb.set_trace()
                    self.A_in_list[i].data = A_in.to(self.device)
                all_channel_emb_list = torch.split(all_emb, self.c_dim, dim=-1)
            
            all_emb = torch.cat(item_embeddings_list, dim=0) #(n_item + n_user, dim)
            final.append(all_emb)
        
        all_embeddings = np.sum(final, 0) / (self.layers+1) #均值, sum的效果呢?
        # all_embeddings = np.sum(final, 0) #均值, sum的效果很差;

        return all_embeddings

    def disentangledLearningOnLayer(self, adj, item_emb_list, all_emb_list, n=0):
        """
            adj: (batch_n_item, n_user + n_item)
            item_emb: (b_n_item, dim)
            all_emb: (n_item + n_user, dim)
        """
        # all_emb = torch.cat([user_emb, item_emb], dim=0) #(all_size, dim)
        self.split_n = 100
        # user_channel_emb_list, item_channel_emb_list = [], []
        all_channel_emb_list = all_emb_list
        batch_channel_emb_list = item_emb_list

        # print("1:{}".format(torch.cuda.memory_allocated(1)))
        # A_last = [adj for _ in range(self.channels)] #(n_item, n_user), 0/1;
        for l in range(self.iterations):
            A_last = []
            for k in range(self.channels):

                #==========================
                # A_k = torch.sum(user_channel_emb_list[i].unsqueeze(0) * item_channel_emb_list[i].unsqueeze(1), dim=-1) #(n_item, n_user)
                # A_k = item_channel_emb_list[i].matmul(user_channel_emb_list[i].T) #(n_item, n_user)
                # batch_num = len(batch_channel_emb_list[k]) // self.split_n
                # A_fold = []
                # print("8:{}".format(torch.cuda.memory_allocated(1)))
                # for n in range(self.split_n):
                #     start = n*batch_num
                #     if n == self.split_n - 1:
                #         end = len(batch_channel_emb_list[k])
                #     else:
                #         end = (n + 1) * batch_num

                #     A_k_n = batch_channel_emb_list[k][start:end, :].matmul(all_channel_emb_list[k].T) #(b_n_item, n_item + n_user)
                #     row_indices, col_indices = adj.indices()[0], adj.indices()[1]
                #     row_mask = (row_indices < end) & (row_indices >= start)
                #     select_row_indices, select_col_indices = row_indices[row_mask], col_indices[row_mask]

                #     #只保留有效的值
                #     val_value = A_k_n[select_row_indices - start, select_col_indices]
                #     A_fold.append(val_value)
                #==========================

                A_k_n = batch_channel_emb_list[k].matmul(all_channel_emb_list[k].T)
                # A_k_n = torch.sparse.mm(batch_channel_emb_list[k], all_channel_emb_list[k].T)
                row_indices, col_indices = adj.indices()[0], adj.indices()[1]
                # row_mask = (row_indices < end) & (row_indices >= start)
                # select_row_indices, select_col_indices = row_indices[row_mask], col_indices[row_mask]
                A_k = A_k_n[row_indices, col_indices]
                
                # print("9:{}".format(torch.cuda.memory_allocated(1)))
                # A_k = torch.cat(A_fold, dim=0) #(value_list)
                # assert A_k.shape[0] == adj.indices().shape[1]
                # create coo sparse matrix
                A_last.append(torch.sparse_coo_tensor(adj.indices(), A_k, adj.shape))
                # A_last[k] = A_last[k] + A_k.multiply(adj) #
            
            # print("4:{}".format(torch.cuda.memory_allocated(1)))
            A_soft = torch.stack(A_last, dim=-1)
            # print("5:{}".format(torch.cuda.memory_allocated(1)))
            A_soft = torch.sparse.softmax(A_soft, dim=2) #(batch_n, n_user + n_item, K), no
            # print("6:{}".format(torch.cuda.memory_allocated(1)))

            batch_channel_emb_list = []
            for k in range(self.channels):
                A_k = torch.select(A_soft, dim=2, index=k)
                # batch_emb_after = A_k.matmul(all_channel_emb_list[k]) #(batch, emb)
                batch_emb_after = torch.sparse.mm(A_k, all_channel_emb_list[k]) #(batch, emb)
                batch_emb_after = F.normalize(batch_emb_after, dim=1)
                # batch_channel_emb_list[k] = batch_emb_after #这句话占内存?????? 为什么呢?
                batch_channel_emb_list.append(batch_emb_after)
            
            # print("99:{}".format(torch.cuda.memory_allocated(1)))
        
        # print("2:{}".format(torch.cuda.memory_allocated(1)))
        all_ouput_emb = torch.cat(batch_channel_emb_list, dim=-1)
            
        # item_emb = torch.cat(item_after_list, dim=-1)
        # return torch.split(all_ouput_emb, [self.user_voc_len, self.item_voc_len])
        # del A_last
        # del all_channel_emb_list
        del batch_channel_emb_list
        # torch.cuda.empty_cache()
        # del A_k
        # return all_emb
        # return torch.cat(item_emb_list, dim=-1)
        return all_ouput_emb
        # return all_ouput_emb

    def disentangledLearningOnLayerGAT(self, adj, item_emb_list, all_emb_list, start_index=0):
        """
            adj: (batch_n_item, n_user + n_item), 获取当前sub_adj的索引, 不利用值;
            item_emb: (b_n_item, dim)
            all_emb: (n_item + n_user, dim)
            n: start_index
        """

        # self.A_in_list[i]
        # all_emb = torch.cat([user_emb, item_emb], dim=0) #(all_size, dim)
        # self.split_n = 100
        # user_channel_emb_list, item_channel_emb_list = [], []
        all_channel_emb_list = all_emb_list
        batch_channel_emb_list = item_emb_list

        # print("1:{}".format(torch.cuda.memory_allocated(1)))
        # A_last = [adj for _ in range(self.channels)] #(n_item, n_user), 0/1;
        # for l in range(self.iterations):
        # A_soft_all = torch.stack(self.A_in_list, dim=2) #(N, N, 2)
        # A_soft = torch.stack(adj, dim=2) #(batch_n, N, 2)

        row_indices_ori, col_indices = adj.indices()[0], adj.indices()[1]
        # row_indices = row_indices_ori + n
        end_index = start_index + len(row_indices_ori)

        indices = torch.stack([row_indices_ori, col_indices])

        # pdb.set_trace()
        A_soft_list = []
        for A_soft_tmp in self.A_in_list:
            # vals = A_soft_tmp.coalesce().values()[row_indices] #(M)
            vals = A_soft_tmp.coalesce().values()[start_index: end_index] #(M)
            A_soft = torch.sparse.FloatTensor(indices, vals, adj.shape)
            A_soft_list.append(A_soft)
        # A_soft.data = A_in.to(self.device)
        # A_soft = torch.stack(A_soft_list, dim=2) #(batch_n, N, 2)

        # pdb.set_trace()
        
        # pdb.set_trace()
        # print("111111---222222")
        # A_soft = torch.sparse.softmax(A_soft, dim=2) #(batch_n, n_user + n_item, K), no
        # print("6:{}".format(torch.cuda.memory_allocated(1)))
        # pdb.set_trace()
        # print("2222222")
        batch_channel_emb_list = []
        for k in range(self.channels):
            # A_k = torch.select(A_soft, dim=2, index=k)
            A_k = A_soft_list[k]
            # batch_emb_after = A_k.matmul(all_channel_emb_list[k]) #(batch, emb)
            # pdb.set_trace()
            batch_emb_after = torch.sparse.mm(A_k, all_channel_emb_list[k]) #(batch, emb)
            batch_emb_after = F.normalize(batch_emb_after, dim=1)
            # batch_channel_emb_list[k] = batch_emb_after #这句话占内存?????? 为什么呢?
            batch_channel_emb_list.append(batch_emb_after)
        
        # print("33333333")
        # update A_in_K
        v_channel_list = []
        for k in range(self.channels):
            row_indices, col_indices = adj.indices()[0], adj.indices()[1]
            all_emb_row = all_channel_emb_list[k][row_indices] #(M, dim)
            all_emb_col = all_channel_emb_list[k][col_indices] #(M, dim)
            v_list = torch.sum(all_emb_row * torch.tanh(all_emb_col), dim=1) #(M)
            # pdb.set_trace()
            v_channel_list.append(v_list)
            # 更新self.A_in_list的部分; todo, 在训练外部更新;
            # 拆分后合并来实现局部更新的功能;


        # print("99:{}".format(torch.cuda.memory_allocated(1)))
        
        # print("2:{}".format(torch.cuda.memory_allocated(1)))
        all_ouput_emb = torch.cat(batch_channel_emb_list, dim=-1)
        v_channel_tensor = torch.stack(v_channel_list, dim=0)
        
        # pdb.set_trace()
        # item_emb = torch.cat(item_after_list, dim=-1)
        # return torch.split(all_ouput_emb, [self.user_voc_len, self.item_voc_len])
        # del A_last
        # del all_channel_emb_list
        del batch_channel_emb_list
        # torch.cuda.empty_cache()
        # del A_k
        # return all_emb
        # return torch.cat(item_emb_list, dim=-1)
        return all_ouput_emb, v_channel_tensor #(3, M)
        # return all_ouput_emb
    
    # code for hypergraph. ref:https://github.com/xiaxin1998/DHCN
    def normalizeAdj(self, data, path):
        """
            归一化矩阵, 用于hypergraphConv.
        input:
            data: ( item_num(user_num), user_num(item_num) )
        output:
            (user(item), user(item))
        
        """
        # path = "../data/Taobao/taobao_ori/user_id_Item_id_pre_adj_mat.npz"

        try:
            pre_adj_mat = sp.load_npz(path)
            print("successfully loaded...")
            adjacency_final = pre_adj_mat
        except :
            # self.raw = np.asarray(data[0])
            # H_T = data_masks(self.raw, n_node)
            print("starting generating adj...")
            
            H_T = data.todok()
            row_sum = 1.0/np.array(H_T.sum(axis=1)).flatten()
            row_sum[np.isinf(row_sum)] = 0.
            row_sum = row_sum.reshape(1, -1) #(1, item_num)
            BH_T = H_T.T.multiply(row_sum) #(user_num, item_num)
            BH_T = BH_T.T #(item_num, user_num)

            H = H_T.T
            col_sum = 1.0/np.array(H.sum(axis=1)).flatten()
            col_sum[np.isinf(col_sum)] = 0.
            col_sum = col_sum.reshape(1, -1)
            DH = H.T.multiply(col_sum)
            DH = DH.T #(user_num, item_num)

            # pdb.set_trace()
            # DHBH_T = np.dot(DH,BH_T)
            DHBH_T = DH.dot(BH_T) #耗时的大头; (user, user)
            print("dot done!")
            # adjacency = DHBH_T.tocoo()
            adjacency = DHBH_T.tocsr()

            # select top K
            count= 0
            # n = 100 #为每个用户选择相似的30个用户; 如何选择合适的值呢?
            n = 30 #为每个用户选择相似的30个用户; 如何选择合适的值呢?
            top_n_idx = []
            for le, ri in zip(adjacency.indptr[:-1], adjacency.indptr[1:]):
                count = count + 1
                if count % 1000 == 0:
                    print("read: ", count)
                n_row_click = min(n, ri - le)
                top_n_idx.append(adjacency.indices[le + np.argpartition(adjacency.data[le:ri], -n_row_click)[-n_row_click:]])
            
            adj_mat = sp.dok_matrix((self.user_voc_len, self.user_voc_len), dtype=np.float32)


            # pdb.set_trace()
            # 'aaa = list(click_user_sequence_queue.keys())[:20]'
            # adjacency = adjacency.todok()

            count = 0
            for hist_i in top_n_idx:
                if count % 1000 == 0:
                    print("write: ", count)
                for i in hist_i:
                    adj_mat[count, i] = adjacency[count, i]
                    adj_mat[i, count] = adjacency[i, count]
                count = count + 1
            adjacency_final = adj_mat.tocsr()
            # pdb.set_trace()
            sp.save_npz(path, adjacency_final)
            print("save successful!!")
        
        return adjacency_final

    def get_ui_bipartite_adj_mat(self, ui_adj_mat, path, user_voc, item_voc, add_self=False):
        """
        ui_adj_mat: (n_item, n_user)
        """
        try:
            s = time()
            norm_adj = sp.load_npz(path)
        except:
            R = ui_adj_mat #(item, user)
            # create adj_mat
            # adj_mat = sp.dok_matrix((self.user_voc_len + self.item_voc_len, self.user_voc_len + self.item_voc_len), dtype=np.float32)
            # adj_mat = adj_mat.tolil()
            R = ui_adj_mat.tolil() #(i, u)
            # adj_mat[:self.user_voc_len, self.user_voc_len:] = R.T #(u, i), 报错, OOM;
            # adj_mat[self.user_voc_len:, :self.user_voc_len] = R

            # u_ui_mat = sp.hstack((R.T,  sp.dok_matrix((self.user_voc_len, self.user_voc_len), dtype=np.float32))) #(u, u+i)
            u_ui_mat = sp.hstack((sp.dok_matrix((user_voc, user_voc), dtype=np.float32), R.T)) #(u, u+i)
            i_ui_mat = sp.hstack((R, sp.dok_matrix((item_voc, item_voc), dtype=np.float32))) #(i, u+i)
            adj_mat = sp.vstack((u_ui_mat, i_ui_mat)) #(u+i, u+i)

            if add_self:
                # add self node
                adj_mat = adj_mat + sp.eye(adj_mat.shape[0])

            # pdb.set_trace()
            adj_mat = adj_mat.todok()
            # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])
            # np.count_nonzero(adj_mat.sum(1))
            # col_sum = np.array(adj_mat.sum(axis=0))
            
            rowsum = np.array(adj_mat.sum(axis=1))
            d_inv = np.power(rowsum, -0.5).flatten()
            # pdb.set_trace()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat = sp.diags(d_inv)
            
            norm_adj = d_mat.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat)
            norm_adj = norm_adj.tocsr()

            # pdb.set_trace()

            end = time()
            print(f"costing {end-s}s, saved norm_mat...")
            sp.save_npz(path, norm_adj)
        
        return norm_adj
    
    def computer(self, adjacency_list, embedding, add_mlp=False):
        """
        intput:
            # adjacency: (user_num/item_num, user_num/item_num)
            adjacency_list: list of adj
            embedding: (user_num/item_num, dim)
        output:
            (user_num/item_num, dim)
        """
        # adjacency = self.adj_user_mat
        
        if add_mlp == False:
            item_embeddings = embedding
        else:
            item_embeddings = torch.matmul(embedding, self.weight_lightgcn) + self.bias_lightgcn #每层的参数设置不同吗?

        final = [item_embeddings]
        for i in range(self.layers):
            # item_embeddings = torch.sparse.mm(self.trans_to_cuda(adjacency), item_embeddings)
            # pdb.set_trace()
            item_embeddings_list = []
            for sub_adjacency in adjacency_list:
                # pdb.set_trace()
                item_embeddings_sub = torch.sparse.mm(sub_adjacency, item_embeddings) #adjacency: (N, user_num);  item_emb:; (user_num, dim)
                # add l2 normalize
                # item_embeddings_sub = torch.nn.functional.normalize(item_embeddings_sub, p=2, dim=1)
                item_embeddings_list.append(item_embeddings_sub)
            item_embeddings = torch.cat(item_embeddings_list, dim=0) #(item_num, dim)

            # add mlp
            # if add_mlp:
            #     item_embeddings = self.leakyrelu(torch.matmul(item_embeddings, self.weight_lightgcn_list[i]) + self.bias_lightgcn_list[i]) #每层的参数设置不同吗?
                
            final.append(item_embeddings)
      #  final1 = trans_to_cuda(torch.tensor([item.cpu().detach().numpy() for item in final]))
      #  item_embeddings = torch.sum(final1, 0)
        item_embeddings = np.sum(final, 0) / (self.layers+1)
        # item_embeddings = np.sum(final, 0)
        return item_embeddings

    def computer_u_at(self, adjacency_list, embedding, add_mlp=False):
        """
        intput:
            # adjacency: (user_num/item_num, user_num/item_num)
            adjacency_list: list of adj
            embedding: (user_num/item_num, dim)
        output:
            (user_num/item_num, dim)
        """
        # adjacency = self.adj_user_mat
        
        if add_mlp == False:
            item_embeddings = embedding
        else:
            item_embeddings = torch.matmul(embedding, self.weight_lightgcn) + self.bias_lightgcn #每层的参数设置不同吗?

        final = [item_embeddings]
        for i in range(self.layers):
            item_embeddings_list = []
            for sub_adjacency in adjacency_list:
                item_embeddings_sub = torch.sparse.mm(sub_adjacency, item_embeddings) #adjacency: (N, user_num);  item_emb:; (user_num, dim)
                # add l2 normalize
                # item_embeddings_sub = torch.nn.functional.normalize(item_embeddings_sub, p=2, dim=1)
                item_embeddings_list.append(item_embeddings_sub)
            item_embeddings = torch.cat(item_embeddings_list, dim=0) #(item_num, dim)

            # add mlp
            # if add_mlp:
            #     item_embeddings = self.leakyrelu(torch.matmul(item_embeddings, self.weight_lightgcn_list[i]) + self.bias_lightgcn_list[i]) #每层的参数设置不同吗?
                
            final.append(item_embeddings)
        item_embeddings = np.sum(final, 0) / (self.layers+1)
        # item_embeddings = np.sum(final, 0)
        return item_embeddings

    def forward(self):
        return


    def bpr_loss_bipartite(self, t_customers, pos_u_list, neg_u_list, pos_uat_list, neg_uat_list, 
                            model_name="disengcn", add_cl=False, add_uat=False, add_vae_loss=False, 
                            add_side_info=False, add_vae_loss_v2=False,
                            vae_merge_side_info=False):
        """
        input:
            users: (bs), user_id
            pos: (bs), item_id
            neg: (bs), item_id
        output:
            loss: scalar
        """

        loss = 0.

        if model_name == "lightgcn":
            # merge_u_uat_emb = torch.cat([self.embedding_user.weight, self.user_attri_emb.reshape(-1, len(self.each_u_at_num)*self.user_at_dim)], dim=-1) #(bs, u_dim + 7*aut_dim)
            # input_user_emb = self.convert_mlp(merge_u_uat_emb)

            input_user_emb = self.embedding_user.weight
            # self.all_user_embeddings = self.computer(self.Graph, torch.cat([input_user_emb, self.embedding_item.weight], dim=0)) #(user_num + item+num, dim)
            self.all_user_embeddings = self.computer(self.Graph, torch.cat([input_user_emb, self.embedding_item.weight], dim=0), add_mlp=self.add_mlp) #(user_num + item+num, dim)

            if add_uat:
                # add user-at lightgcn
                input_second_all_emb = torch.cat([self.all_user_embeddings[:self.user_voc_len], self.embedding_u_at.weight], dim=0)
                output_second_all_emb = self.computer_u_at(self.Graph_u_at, input_second_all_emb, add_mlp=self.add_mlp)
                output_second_user_emb =  output_second_all_emb[:self.user_voc_len]
        
        elif model_name == "disengcn":
            #替换为disentangled learning方法, 观察下效果, 后续添加二部图和disentangled learning之间的对比学习损失函数;
            self.all_user_embeddings = self.disentangleComputer(self.Graph, self.embedding_user.weight, self.embedding_item.weight, mode="train") #(user_num + item+num, dim)
        
        
        elif model_name == "mlp":
            all_embeddings = torch.matmul(torch.cat([self.embedding_user.weight, self.embedding_item.weight]), self.weight_lightgcn) + self.bias_lightgcn #每层的参数设置不同吗?
            # z = self.weight_list[i](all_emb)
            self.all_user_embeddings = F.normalize(all_embeddings, dim=1)

        users_emb_lookup_disen, items_emb_lookup_disen = torch.split(self.all_user_embeddings, [self.user_voc_len, self.item_voc_len])


        if add_side_info:
            batch_user_attri_emb = self.user_attri_emb #(u_voc, 7, dim)
            # 用户属性特征交互;
            user_self_feature = self.feat_interaction(batch_user_attri_emb, self.dense_user_self_biinter, self.dense_user_onehop_siinter, dimension=1)
            merge_id_feat_tensor = torch.cat([users_emb_lookup_disen, user_self_feature], dim=-1)
            # users_emb_lookup_disen = self.leakyrelu(self.merge_id_feat_mlp(merge_id_feat_tensor))
            users_emb_lookup_disen = self.leakyrelu(torch.matmul(merge_id_feat_tensor, self.merge_id_feat_mlp))

        if add_vae_loss:
            # VAE操作来解决冷启动问题
            batch_user_attri_emb = self.user_attri_emb #(u_voc, 7, dim)
            # 用户属性特征交互;
            # user_self_feature = batch_user_attri_emb.sum(dim=1)
            # user_self_feature = self.feat_interaction(batch_user_attri_emb, self.dense_user_self_biinter, self.dense_user_onehop_siinter, dimension=1) #(n_user, dim)
            user_self_feature = self.feat_interaction_mlp(batch_user_attri_emb) #(n_user, dim)
            self.user_mu, self.user_var = self.user_vae.Q(user_self_feature)
            self.user_z = self.user_vae.sample_z(self.user_mu, self.user_var)
            self.user_preference_sample = self.user_vae.P(self.user_z) #(n_user, dim)

            # add vae loss
            # recon_w, kl_w = 1.e-2, 1.e-2
            recon_w, kl_w = 1.e-2, 5.e-1
            # batch_input_user_emb = self.embedding_user.weight[batch_users.long()] #输入表征;
            all_user_embeddings_detach =  self.all_user_embeddings #gnn输出表征;
            batch_input_user_emb = all_user_embeddings_detach[pos_u_list.long()] #GNN输出的表征, 梯度不回传, 只更新VAE模块;
            batch_user_at_emb = self.user_preference_sample[pos_u_list.long()]
            # batch_user_at_emb = self.user_preference_sample
            batch_user_z, batch_user_mu, batch_user_var = self.user_z[pos_u_list.long()], self.user_mu[pos_u_list.long()], self.user_var[pos_u_list.long()]
            recon_loss = torch.norm(batch_user_at_emb - batch_input_user_emb)
            # kl_loss = torch.mean(0.5 * torch.sum(torch.exp(self.user_z) + self.user_mu ** 2 - 1. - self.user_var, 1))
            kl_loss = torch.mean(0.5 * torch.sum(torch.exp(batch_user_z) + batch_user_mu ** 2 - 1. - batch_user_var, 1))
            
            recon_loss_total = recon_w * recon_loss
            kl_loss_total = kl_w * kl_loss

            # pdb.set_trace()
            loss += recon_w * recon_loss
            loss += kl_w * kl_loss

            if vae_merge_side_info:
                merge_id_feat_tensor = torch.cat([users_emb_lookup_disen, self.user_z], dim=-1)
                # users_emb_lookup_disen = self.leakyrelu(self.merge_id_feat_mlp(merge_id_feat_tensor))
                users_emb_lookup_disen = self.leakyrelu(torch.matmul(merge_id_feat_tensor, self.merge_id_feat_mlp))
                # pdb.set_trace()
        
        customer_emb = items_emb_lookup_disen[t_customers.long()]
        pos_u_emb = users_emb_lookup_disen[pos_u_list.long()]
        neg_u_emb = users_emb_lookup_disen[neg_u_list.long()]

        pos_emb_pooling = pos_u_emb
        neg_emb_pooling = neg_u_emb
        pos_scores= torch.sum(customer_emb*pos_emb_pooling, dim=1) #(bs)
        neg_scores= torch.sum(customer_emb*neg_emb_pooling, dim=1)

        loss += torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        # loss = torch.sum(nn.functional.softplus(neg_scores - pos_scores))

        reg_loss = (1/2)*(customer_emb.norm(2).pow(2) + 
                          pos_u_emb.norm(2).pow(2) + 
                          neg_u_emb.norm(2).pow(2))/float(len(t_customers))

        if add_uat:
            pos_u_at_emb = output_second_user_emb[pos_u_list.long()]
            neg_u_at_emb = output_second_user_emb[neg_u_list.long()]
            pos_uat_scores= torch.sum(customer_emb*pos_u_at_emb, dim=1) #(bs)
            neg_uat_scores= torch.sum(customer_emb*neg_u_at_emb, dim=1)
            loss_uat = torch.mean(nn.functional.softplus(neg_uat_scores - pos_uat_scores))
            # loss += 1.0*loss_uat

            # add cl loss;
            loss_cl_uat = self.loss_contrastive_triple(pos_u_emb, pos_u_at_emb, add_local=True, add_global=False, cl_type="user", all_embedding=None)
            loss += 1.0e-5 * loss_cl_uat
            # pdb.set_trace()

        if add_vae_loss_v2:
            # user_self_feature = self.user_attri_emb.reshape(self.user_voc_len, -1) #(u_voc, 7*dim)
            # add attribute-based lightgcn
            # user_self_feature = torch.sum(self.user_attri_emb, dim=1) #(u_voc, dim)
            self.user_mu, self.user_var = self.user_vae.Q(user_self_feature)
            self.user_z = self.user_vae.sample_z(self.user_mu, self.user_var)
            
            self.user_preference_sample = self.user_vae.P(self.user_z) #(n_user, dim)
            # add kl lass and recon loss;
            recon_w, kl_w, vae_cl_w = 1.e-2, 5.e-1, 1.
            # batch_input_user_emb = self.embedding_user.weight[batch_users.long()] #输入表征;
            all_user_embeddings_detach = user_self_feature #gnn输出表征;
            batch_input_user_emb = all_user_embeddings_detach[pos_u_list.long()] #GNN输出的表征, 梯度不回传, 只更新VAE模块;
            batch_user_at_emb = self.user_preference_sample[pos_u_list.long()]
            batch_user_z, batch_user_mu, batch_user_var = self.user_z[pos_u_list.long()], self.user_mu[pos_u_list.long()], self.user_var[pos_u_list.long()]
            recon_loss = torch.norm(batch_user_at_emb - batch_input_user_emb) # A --> A
            kl_loss = torch.mean(0.5 * torch.sum(torch.exp(batch_user_z) + batch_user_mu ** 2 - 1. - batch_user_var, 1))

            # add cl loss
            ori_user_emb = self.all_user_embeddings[:self.user_voc_len]
            pos_u_emb_ori = ori_user_emb[pos_u_list.long()]
            neg_u_emb_ori = ori_user_emb[neg_u_list.long()]

            # vae_cl_loss = self.loss_contrastive_triple(pos_u_emb_ori, batch_user_z, add_local=True, add_global=False)
            pos_scores_vae = torch.sum(batch_user_z*pos_u_emb_ori, dim=1) #(bs)
            neg_scores_vae = torch.sum(batch_user_z*neg_u_emb_ori, dim=1)
            vae_cl_loss = torch.mean(nn.functional.softplus(neg_scores_vae - pos_scores_vae))

            vae_total_loss = recon_w * recon_loss + kl_w * kl_loss + vae_cl_w * vae_cl_loss
            loss += vae_total_loss
            # pdb.set_trace()
            # pdb.set_trace()
        
        # cl loss is valid;
        cl_loss = 0.
        if self.cl_w !=0.0:
            self.all_user_embeddings_cl = torch.matmul(torch.cat([self.embedding_user.weight, self.embedding_item.weight]), self.weight_lightgcn) + self.bias_lightgcn #每层的参数设置不同吗?
            # self.all_user_embeddings_cl = F.normalize(all_embeddings, dim=1)

            users_emb_lookup_cl, items_emb_lookup_cl = torch.split(self.all_user_embeddings_cl, [self.user_voc_len, self.item_voc_len])
            customer_cl_emb = items_emb_lookup_cl[t_customers.long()]
            pos_u_cl_emb = users_emb_lookup_cl[pos_u_list.long()]
            # neg_u_cl_emb = users_emb_lookup_cl[neg_u_list.long()]
            
            cl_loss_u = self.loss_contrastive_triple(pos_u_emb, pos_u_cl_emb, add_local=True, add_global=True, cl_type="user", all_embedding=self.all_user_embeddings) # local 与 global之间的权重可调整;
            cl_loss_i = self.loss_contrastive_triple(customer_emb, customer_cl_emb, add_local=True, add_global=True, cl_type="item", all_embedding=self.all_user_embeddings) # local 与 global之间的权重可调整;
            cl_loss = cl_loss_u + cl_loss_i
            # pdb.set_trace()
            cl_loss = self.cl_w * cl_loss
            loss += cl_loss
            # pdb.set_trace()

        return loss, reg_loss, cl_loss

    # def loss_contrastive_triple(self, tensor_anchor_v1, tensor_anchor_v2, temp_value=0.1):
    def loss_contrastive_triple(self, tensor_anchor_v1, tensor_anchor_v2, add_local=False, add_global=False, cl_type='user', all_embedding=None):
        """
            triple-based cl loss, row shuffle or row and column shuffle;
            refer to Yu et al. Self-Supervised Multi-Channel Hypergraph Convolutional Network for Social Recommendation, WWW'21.
        """

        def row_shuffle(embedding):
            random_idx = torch.randperm(embedding.size(0))
            return embedding[random_idx, :]
        
        def row_column_shuffle(embedding):
            # col random
            embedding_t = embedding.t()
            random_idx = torch.randperm(embedding_t.size(0))
            embedding_t = embedding_t[random_idx, :]
            corrupted_embedding = embedding_t.t()
            # row random
            random_idx = torch.randperm(corrupted_embedding.size(0))
            corrupted_embedding = corrupted_embedding[random_idx, :]
            return corrupted_embedding
        
        def score(x1,x2):
            return x1.multiply(x2).sum(1)
        
        user_embeddings = tensor_anchor_v1
        # user_embeddings = tf.math.l2_normalize(em,1) #For Douban, normalization is needed.
        # edge_embeddings = tf.sparse_tensor_dense_matmul(adj,user_embeddings)
        edge_embeddings = tensor_anchor_v2
        #Local MIM
        pos = score(user_embeddings, edge_embeddings)
        neg1 = score(row_shuffle(user_embeddings), edge_embeddings)
        # neg1 = score(row_shuffle(edge_embeddings), user_embeddings)
        neg2 = score(row_column_shuffle(edge_embeddings), user_embeddings)
        # local_loss = tf.reduce_sum(-tf.log(tf.sigmoid(pos-neg1))-tf.log(tf.sigmoid(neg1-neg2)))

        if add_local:
            loss = (-torch.log(F.sigmoid(pos-neg1))-torch.log(F.sigmoid(neg1-neg2))).sum()
        # local_loss = (-torch.log(F.sigmoid(pos-neg1))-torch.log(F.sigmoid(neg1-neg2))).mean()
        #Global MIM
        # graph = torch.mean(edge_embeddings,0, keepdim=True) #(1, dim)

        if add_global:
            # global层面的对比学习;
            if cl_type == "user":
                # graph = torch.mean(user_embeddings,0, keepdim=True) #(1, dim)
                graph = torch.mean(all_embedding[:self.user_voc_len], 0, keepdim=True) #(1, dim)
            if cl_type == "item":
                graph = torch.mean(all_embedding[self.user_voc_len:], 0, keepdim=True) #(1, dim)

            pos = score(user_embeddings, graph)
            neg1 = score(row_column_shuffle(user_embeddings), graph)
            global_loss = torch.sum(-torch.log(F.sigmoid(pos-neg1)))
            loss += global_loss
        
        # global_loss = torch.mean(-torch.log(F.sigmoid(pos-neg1)))
        return loss
    

    def ssl_layer_loss_infoNce(self, current_embedding, previous_embedding, previous_all_embeddings, ssl_tmp=0.1):

        """
            CL分母是整个用户或者物品的词表;
        """
        # current_user_embeddings, current_item_embeddings = torch.split(current_embedding, [self.n_users, self.n_items])
        # previous_user_embeddings_all, previous_item_embeddings_all = torch.split(previous_embedding, [self.n_users, self.n_items])

        # current_user_embeddings = current_user_embeddings[user]
        # previous_user_embeddings = previous_user_embeddings_all[user]

        current_user_embeddings = current_embedding
        previous_user_embeddings = previous_embedding

        norm_user_emb1 = F.normalize(current_user_embeddings)
        norm_user_emb2 = F.normalize(previous_user_embeddings)
        norm_all_user_emb = F.normalize(previous_all_embeddings)
        pos_score_user = torch.mul(norm_user_emb1, norm_user_emb2).sum(dim=1)
        ttl_score_user = torch.matmul(norm_user_emb1, norm_all_user_emb.transpose(0, 1))
        pos_score_user = torch.exp(pos_score_user / ssl_tmp)
        # pdb.set_trace()
        ttl_score_user = torch.exp(ttl_score_user / ssl_tmp).sum(dim=1)
        # ttl_score_user = torch.exp(ttl_score_user).sum(dim=1)

        ssl_loss_user = -torch.log(pos_score_user / ttl_score_user).sum()

        # current_item_embeddings = current_item_embeddings[item]
        # previous_item_embeddings = previous_item_embeddings_all[item]
        # norm_item_emb1 = F.normalize(current_item_embeddings)
        # norm_item_emb2 = F.normalize(previous_item_embeddings)
        # norm_all_item_emb = F.normalize(previous_item_embeddings_all)
        # pos_score_item = torch.mul(norm_item_emb1, norm_item_emb2).sum(dim=1)
        # ttl_score_item = torch.matmul(norm_item_emb1, norm_all_item_emb.transpose(0, 1))
        # pos_score_item = torch.exp(pos_score_item / ssl_tmp)
        # ttl_score_item = torch.exp(ttl_score_item / ssl_tmp).sum(dim=1)

        # ssl_loss_item = -torch.log(pos_score_item / ttl_score_item).sum()

        # ssl_loss = ssl_reg * (ssl_loss_user + item_alpha * ssl_loss_item)
        return ssl_loss_user


    def loss_contrastive(self, tensor_anchor_v1, tensor_anchor_v2, temp_value=0.1):
        """
        infoNCE
        tensor_anchor_v1: (bs, dim)
        tensor_anchor_v2: (bs, dim)
        """   
        head_feat_1, head_feat_2 = F.normalize(tensor_anchor_v1, dim=1), F.normalize(tensor_anchor_v2, dim=1)
        # all_score = torch.exp(torch.sum(tensor_anchor*tensor_all, dim=1)/temp_value).view(-1, 1+self.num_neg)
        pos_score = (head_feat_1 * head_feat_2).sum(-1) #(bs)

        # true version
        # pos_item = torch.cat([head_feat_1, head_feat_2], dim=0) #(2*bs, dim)
        # # pdb.set_trace()
        # all_tensors = pos_item.unsqueeze(0).repeat(tensor_anchor_v1.size(0)*2, 1, 1) #(2*bs, 2*bs, dim)
        # all_scores = (pos_item.unsqueeze(1) * all_tensors).sum(-1) #(2*bs, 2*bs)
        # all_scores_mask = all_scores + (torch.eye(tensor_anchor_v1.size(0)*2) * (-1e8)).to(self.device)

        # simple version
        all_scores_mask = torch.matmul(head_feat_1, head_feat_2.t())

        all_score = torch.sum(torch.exp(all_scores_mask/temp_value), dim=1) #(2*bs)
        # pos_score = torch.exp(torch.cat([pos_score, pos_score], dim=0)) #(2*bs)
        pos_score = torch.exp(pos_score) #(2*bs)

        # cl_loss = (-torch.log(pos_score / all_score)).mean()
        cl_loss = (-torch.log(pos_score / all_score)).sum()

        return cl_loss
    
    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))


    def _split_A_hat_no_torch(self,A):
        A_fold = []
        size = A.shape[0]
        fold_len = size // self.folds
        for i_fold in range(self.folds):
            start = i_fold*fold_len
            if i_fold == self.folds - 1:
                end = size
            else:
                end = (i_fold + 1) * fold_len
            # A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(self.device))
            A_fold.append(A[start:end])
        return A_fold

    
    def _split_A_hat(self,A):
        A_fold = []
        size = A.shape[0]
        fold_len = size // self.folds
        for i_fold in range(self.folds):
            start = i_fold*fold_len
            if i_fold == self.folds - 1:
                end = size
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(self.device))
        return A_fold

    def generate_train_data_for_train_add_attributes(self, voc_len):
        """
            生成训练数据
        """

        target_item_id_list, pos_users_list, neg_users_list, target_user_ats_list = [], [], [], []

        assert len(self.target_user_list) == len(self.pos_items_list)

        for t_user, pos_items_list in zip(self.target_user_list, self.pos_items_list):
            pos_user_id = np.random.choice(pos_items_list, 1, replace=False)[0]

            # select user attributes
            if t_user in self.userid_atlist:
                user_attri = self.userid_atlist[t_user] #(7)
            else:
                user_attri = [0 for _ in range(len(self.each_u_at_num))] #(7)
            
            # select negative users
            while True:
                # neg_idx = np.random.randint(0, self.user_voc_len)
                neg_idx = np.random.randint(0, voc_len)
                if neg_idx in pos_items_list:
                    continue
                else:
                    break
            
            neg_user_id = neg_idx
            target_item_id_list.append(t_user)
            pos_users_list.append(pos_user_id)
            neg_users_list.append(neg_user_id)
            target_user_ats_list.append(user_attri)
        
        return target_item_id_list, pos_users_list, neg_users_list, target_user_ats_list

    def feat_interaction(self, feature_embedding, fun_bi, fun_si, dimension):
        """
            feature_embedding: (n_user, 7, dim)
        """
        summed_features_emb_square = (torch.sum(feature_embedding, dim=dimension)).pow(2)
        squared_sum_features_emb = torch.sum(feature_embedding.pow(2), dim=dimension)
        deep_fm = 0.5 * (summed_features_emb_square - squared_sum_features_emb)
        # pdb.set_trace()
        deep_fm = self.leakyrelu(fun_bi(deep_fm))
        bias_fm = self.leakyrelu(fun_si(feature_embedding.sum(dim=dimension)))
        nfm = deep_fm + bias_fm
        return nfm

    def feat_interaction_mlp(self, feature_embedding):
        # concat_u_at_emb = []
        # for i in range(len(self.each_u_at_num)):
        #     u_at_after_mlp = self.u_at_mlps[i](feature_embedding[:, i, :])
        #     concat_u_at_emb.append(u_at_after_mlp)
        
        user_embs = self.leakyrelu(self.deep_vae_mlp(feature_embedding.reshape(self.user_voc_len, -1))) #(user_num, dim)
        # senet
        # concat_u_at_after_emb = torch.stack(concat_u_at_emb, dim=1) #(n_user, 7, dim)

        # attention_merge_u_at_emb = self.senet_layer(concat_u_at_after_emb) #(n_user, dim)
        # attention_merge_u_at_emb = attention_merge_u_at_emb.sum(1)

        # return attention_merge_u_at_emb
        return user_embs

    
    def cal_loss_v2(self, add_cl=False, model_name="disengcn"):
        """
        define node-level and hyperedge-level losses
        """

        # if split == True:
        # self.Graph = self._split_A_hat(self.adj_user_mat) # list of sparse tensors;
        # pdb.set_trace()
        # self.Graph = self._split_A_hat(self.ui_mat) # list of sparse tensors; self.ui_mat: (user+item, user+item)

        print("done split matrix torch")
        # train_data_sub_list = self._split_A_hat_no_torch(self.adj_user_mat)
        # train_data_sub_list = self._split_A_hat_no_torch(self.adj_mat) #(item, user)

        # train_data = [[] for _  in range(3)]
        train_data = [[] for _  in range(5)]

        # t_user_list, pos_items_list, neg_items_list = self.generate_train_data_for_train(voc_len = self.item_voc_len)
        # t_user_list, pos_items_list, neg_items_list, t_user_ats_list = self.generate_train_data_for_train_add_attributes(voc_len = self.item_voc_len)
        t_c_list, pos_u_list, neg_u_list, pos_uat_list, neg_uat_list = self.generate_train_data_for_train_c2u_add_uat(voc_len = self.user_voc_len)

        train_data[0] = t_c_list
        train_data[1] = pos_u_list
        train_data[2] = neg_u_list
        train_data[3] = pos_uat_list
        train_data[4] = neg_uat_list
        
        # pdb.set_trace()
        print("train sample data: done!")
        t_customers = torch.Tensor(train_data[0]).long().to(self.device) #item
        pos_users = torch.Tensor(train_data[1]).long().to(self.device) # padding,
        neg_users = torch.Tensor(train_data[2]).long().to(self.device) # padding
        pos_user_ats = torch.Tensor(train_data[3]).long().to(self.device) # padding
        neg_user_ats = torch.Tensor(train_data[4]).long().to(self.device) # padding
        
        # shuffle traindata
        # target_users, pos_users, pos_len, neg_users, neg_len = self.shuffle(target_users, pos_users, pos_len, neg_users, neg_len)
        t_custormers, pos_users, neg_users, pos_user_ats, neg_user_ats = self.shuffle(t_customers, pos_users, neg_users, pos_user_ats, neg_user_ats)
        total_batch = len(t_custormers) // self.bpr_batch_size + 1
        aver_loss, aver_u_c_pair_loss = 0., 0.
        aver_recon_loss, aver_kl_loss = 0., 0.

        start_time = time()
        for (batch_i,
         (batch_custormer,
          batch_pos_users,
          batch_neg_users,
          batch_pos_user_ats,
          batch_neg_user_ats)) in enumerate(self.minibatch(t_custormers,
                                                   pos_users,
                                                   neg_users,
                                                   pos_user_ats,
                                                   neg_user_ats,
                                                   batch_size=self.bpr_batch_size)):

            if add_cl == True:
                # loss, reg_loss, cl_loss = self.bpr_loss_bipartite_cl(batch_users, batch_pos_users, batch_neg_users, add_cl=True)
                pass
            elif model_name == "disengcn":
                loss, reg_loss, cl_loss = self.bpr_loss_bipartite(batch_custormer,
                                                                    batch_pos_users,
                                                                    batch_neg_users,
                                                                    batch_pos_user_ats,
                                                                    batch_neg_user_ats, 
                                                                    model_name=model_name,
                                                                    add_uat=True)
            elif model_name == "lightgcn":
                loss, reg_loss, cl_loss = self.bpr_loss_bipartite(batch_custormer,
                                                                    batch_pos_users,
                                                                    batch_neg_users,
                                                                    batch_pos_user_ats,
                                                                    batch_neg_user_ats, 
                                                                    model_name=model_name,
                                                                    add_uat=False,
                                                                    add_vae_loss=self.add_vae,
                                                                    add_side_info=True,
                                                                    add_vae_loss_v2=True,
                                                                    vae_merge_side_info=False)
            elif model_name == "mlp":
                loss, reg_loss, cl_loss = self.bpr_loss_bipartite(batch_custormer,
                                                                    batch_pos_users,
                                                                    batch_neg_users,
                                                                    batch_pos_user_ats,
                                                                    batch_neg_user_ats, 
                                                                    model_name=model_name,
                                                                    add_uat=True)

            # pdb.set_trace()
            u_c_pair_loss = loss.detach()
            # add vae loss
            u_c_pair_loss_cpu = u_c_pair_loss.cpu().item()


            reg_loss = reg_loss*self.weight_decay
            loss = loss + reg_loss
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()
            loss_cpu = loss.cpu().item()
            aver_loss += loss_cpu
            aver_u_c_pair_loss += u_c_pair_loss_cpu
            
                
        aver_loss = aver_loss / total_batch
        aver_u_c_pair_loss = aver_u_c_pair_loss / total_batch
        aver_recon_loss = aver_recon_loss / total_batch
        aver_kl_loss = aver_kl_loss / total_batch

        return f"loss{aver_loss:.3f}--u_c_pair_loss{aver_u_c_pair_loss: .3f}--KL_loss{aver_kl_loss: .3f}--recon_loss{aver_recon_loss: .3f}--{time()-start_time}"


    def train_model(self, add_cl=False, model_name="disengcn"):
        # for epoch training
        self.Graph = self._split_A_hat(self.ui_mat)
        self.Graph_u_at = self._split_A_hat(self.u_at_adj_norm_mat)
        # pdb.set_trace()
        # for epoch in range(self.epoch):
        for epoch in range(self.epoch_pre):
            # self.generate_inter_status()
            # loss_info = self.cal_loss()
            loss_info = self.cal_loss_v2(add_cl=add_cl, model_name=model_name)
            print(f"epoch: {epoch}: ", loss_info)
        # return

    def save_embeddings(self, save_model_name="disengcn", use_vae=True):
        self.forward_embeddings(save_model_name=save_model_name, use_vae=use_vae, is_save=True)
        
    def forward_embeddings(self, save_model_name="disengcn", use_vae=False, is_add_uat=False, add_side_info=False, vae_merge_side_info=False):
        """
        save user embeddings and item embeddings
        """
        input_user_emb = self.embedding_user.weight

        if use_vae:
            # user_self_feature = self.user_attri_emb.sum(1)
            # user_self_feature = self.feat_interaction(self.user_attri_emb, self.dense_user_self_biinter, self.dense_user_onehop_siinter, dimension=1) #(n_user, dim)
            user_self_feature = self.feat_interaction_mlp(self.user_attri_emb) #(n_user, dim)
            user_mu, user_var = self.user_vae.Q(user_self_feature)
            user_z = self.user_vae.sample_z(user_mu, user_var) #(n_user, dim)
            # self.vae_user_emb = self.user_vae.P(user_z) #(n_user, dim)
            self.vae_user_emb = user_z #(n_user, dim), use output of encoder as features;
        
        if save_model_name == "embedding":
            # 1. original user embeddings
            self.user_embeddings = self.embedding_user.weight
            self.item_embeddings = self.embedding_item.weight
        elif save_model_name == "mlp":
            all_embeddings = torch.matmul(torch.cat([input_user_emb, self.embedding_item.weight]), self.weight_lightgcn) + self.bias_lightgcn #每层的参数设置不同吗?
            all_embeddings = F.normalize(all_embeddings, dim=1)  
            self.user_embeddings = all_embeddings[:self.user_voc_len]
            self.item_embeddings = all_embeddings[self.user_voc_len:]
        elif save_model_name == "lightgcn":
            input_user_emb = self.embedding_user.weight
            all_embeddings = self.computer(self.Graph, torch.cat([input_user_emb, self.embedding_item.weight]), add_mlp=self.add_mlp)

            all_embeddings_cpu = all_embeddings.detach().cpu().numpy()
            user_embeddings, item_embeddings = all_embeddings[:self.user_voc_len], all_embeddings[self.user_voc_len:]
            
            if is_add_uat:
                # add user-at lightgcn
                input_second_all_emb = torch.cat([all_embeddings[:self.user_voc_len], self.embedding_u_at.weight], dim=0)
                output_second_all_emb = self.computer_u_at(self.Graph_u_at, input_second_all_emb, add_mlp=self.add_mlp)
                user_embeddings =  output_second_all_emb[:self.user_voc_len]

            if add_side_info:
                batch_user_attri_emb = self.user_attri_emb #(u_voc, 7, dim)
                # 用户属性特征交互, # add cold user embedding
                user_self_feature = self.feat_interaction(batch_user_attri_emb, self.dense_user_self_biinter, self.dense_user_onehop_siinter, dimension=1)
                user_mu, user_var = self.user_vae.Q(user_self_feature)
                user_z = self.user_vae.sample_z(user_mu, user_var) # ID features of cold users;
                # pdb.set_trace()
                user_embeddings = torch.where(torch.transpose(self.multi_hot_label, 0, 1)>0, user_embeddings, user_z)
                
                merge_id_feat_tensor = torch.cat([user_embeddings, user_self_feature], dim=-1)
                user_embeddings = self.leakyrelu(torch.matmul(merge_id_feat_tensor, self.merge_id_feat_mlp))
            
            if vae_merge_side_info:
                merge_id_feat_tensor = torch.cat([user_embeddings, self.vae_user_emb], dim=-1)
                # users_emb_lookup_disen = self.leakyrelu(self.merge_id_feat_mlp(merge_id_feat_tensor))
                user_embeddings = self.leakyrelu(torch.matmul(merge_id_feat_tensor, self.merge_id_feat_mlp))

            
            self.user_embeddings = user_embeddings
            self.item_embeddings = item_embeddings

        elif save_model_name == "disengcn":
            all_embeddings = self.disentangleComputer(self.Graph, self.embedding_user.weight, self.embedding_item.weight, mode="predict") #(user_num + item+num, dim)
            self.user_embeddings = all_embeddings[:self.user_voc_len]
            self.item_embeddings = all_embeddings[self.user_voc_len:]


    
    def shuffle(self, *arrays, **kwargs):
        require_indices = kwargs.get('indices', False)

        if len(set(len(x) for x in arrays)) != 1:
            raise ValueError('All inputs to shuffle must have '
                            'the same length.')

        shuffle_indices = np.arange(len(arrays[0]))
        np.random.shuffle(shuffle_indices)

        if len(arrays) == 1:
            result = arrays[0][shuffle_indices]
        else:
            result = tuple(x[shuffle_indices] for x in arrays)

        if require_indices:
            return result, shuffle_indices
        else:
            return result

    def minibatch(self, *tensors, **kwargs):

        batch_size = kwargs.get('batch_size', self.bpr_batch_size)

        if len(tensors) == 1:
            tensor = tensors[0]
            for i in range(0, len(tensor), batch_size):
                yield tensor[i:i + batch_size]
        else:
            for i in range(0, len(tensors[0]), batch_size):
                yield tuple(x[i:i + batch_size] for x in tensors)

    def count_parameters(self, count_embedding=True):
        total_params = 0
        for name, param in self.named_parameters(): 
            if not count_embedding and "embedding" in name:
                continue
            if param.requires_grad:
                total_params += param.numel()
        logging.info("Total number of parameters: {}.".format(total_params))
