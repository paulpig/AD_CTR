from tkinter import W

from requests import session
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
import pdb


class HyperGraphSim(nn.Module):
    def __init__(self,
                 adj_mat,
                #  adj_user_mat,
                 gpu=-1,
                 graph_layer=1,
                 learning_rate=1e-3, 
                 embedding_dim=10, 
                 epoch=100,
                 bpr_batch_size=10000):
        super(HyperGraphSim, self).__init__()
        
        # parameters
        self.emb_size = embedding_dim
        self.layers = graph_layer
        # self.dataset = dataset
        self.item_voc_len, self.user_voc_len = adj_mat.get_shape()
        self.adj_mat = adj_mat #(item, user)
        path = "../data/Taobao/taobao_ori/user_id_Item_id_pre_adj_mat_for_gcn.npz"
        path_v2 = "../data/Taobao/taobao_ori/user_id_Item_id_pre_adj_mat_item2item.npz"
        # self.adj_user_mat = self.normalizeAdj(self.adj_mat, path) #(user, user)
        self.adj_user_mat = self.normalizeAdjGCN(self.adj_mat, path) #(user, user)
        # self.adj_item_mat = self.normalizeAdj(self.adj_mat.T, path_v2) #(item, item)

        self.embedding_user = nn.Embedding(self.user_voc_len, self.emb_size)
        self.embedding_item = nn.Embedding(self.item_voc_len, self.emb_size)
        # self.compile(kwargs["optimizer"], loss=kwargs["loss"], lr=learning_rate)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.epoch = epoch
        self.folds = 100
        # self.kwargs = kwargs
        self.bpr_batch_size = bpr_batch_size
        self.init_parameters()
        self.device = get_device(gpu)
        # self.model_to_device()
        self.to(device=self.device)

    def init_parameters(self):
        stdv = 1.0 / math.sqrt(self.emb_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
    

    def trans_to_cuda(self, variable):
        if torch.cuda.is_available():
            return variable.cuda()
        else:
            return variable

    def trans_to_cpu(self, variable):
        if torch.cuda.is_available():
            return variable.cpu()
        else:
            return variable
    
    def normalizeAdjGCN(self, data, path):
        try:
            pre_adj_mat = sp.load_npz(path)
            print("successfully loaded...")
            adjacency = pre_adj_mat
        except:
            print("generating adjacency matrix")
            UserItemNet = data
            s = time()
            adj_mat = sp.dok_matrix((self.user_voc_len + self.item_voc_len, self.user_voc_len + self.item_voc_len), dtype=np.float32)
            adj_mat = adj_mat.tolil()
            R = UserItemNet.tolil()
            pdb.set_trace()
            adj_mat[:self.user_voc_len, self.user_voc_len:] = R.T #(U, I) graph
            adj_mat[self.user_voc_len:, :self.user_voc_len] = R
            adj_mat = adj_mat.todok()
            # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])
            
            rowsum = np.array(adj_mat.sum(axis=1))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat = sp.diags(d_inv)
            
            norm_adj = d_mat.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat)
            adjacency = norm_adj.tocsr()
            end = time()
            print(f"costing {end-s}s, saved norm_mat...")
            pdb.set_trace()
            sp.save_npz(path, adjacency)
            print("save successful!!")
        
        return adjacency


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
            adjacency = pre_adj_mat
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
            DHBH_T = DH.dot(BH_T) #耗时的大头;
            # adjacency = DHBH_T.tocoo()
            adjacency = DHBH_T.tocsr()
            # n_node = 
            # np.sum(np.isnan(BH_T.data))
            # targets = np.asarray(data[1])
            # self.length = len(self.raw)
            # self.shuffle = shuffle
            # adjacency = sp.load_npz(path)
            sp.save_npz(path, adjacency)
            print("save successful!!")
        
        return adjacency
    
    # def pre_train_model(self):
    #     """
    #         pretrain hypergraph model for initializing the embedding layer of CTR model.
    #     """
    #     # new hypergraph model

    #     hypergraph = HyperGraph()

    #     # for epoch in range(x): train hypergraph model

    #     return None

    
    def computer(self, adjacency, embedding):
        """
        intput:
            # adjacency: (user_num/item_num, user_num/item_num)
            embedding: (user_num/item_num, dim)
        output:
            (user_num/item_num, dim)
        """
        # adjacency = self.adj_user_mat
        item_embeddings = embedding
        item_embedding_layer0 = item_embeddings
        final = [item_embedding_layer0]
        for i in range(self.layers):
            item_embeddings = torch.sparse.mm(self.trans_to_cuda(adjacency), item_embeddings)
            final.append(item_embeddings)
      #  final1 = trans_to_cuda(torch.tensor([item.cpu().detach().numpy() for item in final]))
      #  item_embeddings = torch.sum(final1, 0)
        item_embeddings = np.sum(final, 0) / (self.layers+1)

        return item_embeddings

    def forward(self):
        return
    # def forward(self, adjacency_user, adjacency_item):
    #     output_user_embeddings = self.computer(adjacency_user, self.embedding_user.weight)
    #     output_item_embeddings = self.computer(adjacency_item, self.embedding_item.weight)
    #     return output_user_embeddings, output_item_embeddings
    
    def train_sample_gen(self, adjaceny_ori, max_pos_len = 10):
        """
        构建训练数据, return: (target_item, pos_item_list, neg_item_list)
        input:
            adjaceny_ori: (n_item, n_user)
            max_pos_len: the number of postive users
        return:
            list of (target_user_id, pos_users, pos_len, neg_users, neg_len)
        """
        adj_list_of_list = adjaceny_ori.tolil() #kill
        max_len = max_pos_len
        item_voc = len(adj_list_of_list)
        # result = []
        total_start = time()
        target_user_id_list = []
        pos_users_list = []
        pos_len_list = []
        neg_users_list = []
        neg_len_list = []

        pdb.set_trace()
        for index, user_id_list in enumerate(adj_list_of_list):
            item_id = index + 1

            # sample the target user
            if len(user_id_list) < 2:
                continue
            else:
                random.shuffle(user_id_list)
                target_user_id = user_id_list[0]
            
            # remove the target item
            user_id_list.pop(0)

            # 1. random sample users from the current session for positive samples
            if len(user_id_list) > max_len:
                pos_users = np.random.choice(user_id_list, max_len, replace=False)
            else:
                pos_users = user_id_list
            pos_len = len(pos_users)
            # padding
            if len(pos_users) < max_len:
                pos_users += [0] * (max_len - len(pos_users))
            # 2. random sample users from the other session for negative samples
            while True:
                session_id = np.random.randint(0, item_voc)
                if session_id == index:
                    continue
                else:
                    break
            neg_sess = adj_list_of_list[session_id]
            if len(neg_sess) > max_len:
                neg_users = np.random.choice(neg_sess, max_len, replace=False)
            else:
                neg_users = neg_sess
            neg_len = len(neg_users)
            # padding
            if len(neg_users) < max_len:
                neg_users += [0] * (max_len - len(neg_users))
            
            # result.append((target_user_id, pos_users, pos_len, neg_users, neg_len))
            target_user_id_list.append(target_user_id)
            pos_users_list.append(pos_users)
            pos_len_list.append(pos_len)
            neg_users_list.append(neg_users)
            neg_len_list.append(neg_len)
        
        print("cost time of generating dataset: ", time() - total_start)
        return (target_user_id_list, pos_users_list, pos_len_list, neg_users_list, neg_len_list)

    def bpr_loss(self, users, pos, pos_len, neg, neg_len):
        """
        input:
            users: (bs)
            pos: (bs, max_len)
            pos_len: (bs)
            neg: (bs, max_len)
            neg_len(bs)
        output:
            loss: scalar
        """
        all_user_embeddings = self.computer(self.adj_user_mat, self.embedding_user.weight)
        # users_emb = self.embedding_user(users.long()) #(bs, dim)
        # pos_emb   = self.embedding_user(pos.long()) #(bs, max_len, dim)
        # neg_emb   = self.embedding_user(neg.long()) #(bs, max_len, dim)

        users_emb = all_user_embeddings[users.long()]
        pos_emb = all_user_embeddings[pos.long()]
        neg_emb = all_user_embeddings[neg.long()]

        pos_mask = torch.arange(pos.shape[1])[None, :] < pos_len[:, None] #(bs, max_len)
        neg_mask = torch.arange(neg.shape[1])[None, :] < neg_len[:, None] #(bs, max_len)

        # mean pooling
        pos_emb_pooling = (pos_mask[:, :, None] * pos_emb).sum(1) #(bs, dim)
        neg_emb_pooling = (neg_mask[:, :, None] * neg_emb).sum(1)

        pos_scores= torch.sum(users_emb*pos_emb_pooling, dim=1) #(bs)
        neg_scores= torch.sum(users_emb*neg_emb_pooling, dim=1)

        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (1/2)*(users_emb.norm(2).pow(2) + 
                          pos_emb.norm(2).pow(2) + 
                          neg_emb.norm(2).pow(2))/float(len(users))

        return loss, reg_loss
    

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))


    def _split_A_hat_no_torch(self,A):
        A_fold = []
        fold_len = self.user_voc_len // self.folds
        for i_fold in range(self.folds):
            start = i_fold*fold_len
            if i_fold == self.folds - 1:
                end = self.user_voc_len
            else:
                end = (i_fold + 1) * fold_len
            # A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(self.device))
            A_fold.append(A[start:end])
        return A_fold

    
    def _split_A_hat(self,A):
        A_fold = []
        fold_len = self.user_voc_len // self.folds
        for i_fold in range(self.folds):
            start = i_fold*fold_len
            if i_fold == self.folds - 1:
                end = self.user_voc_len
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(self.device))
        return A_fold

    def cal_loss(self, split=True):
        """
        define node-level and hyperedge-level losses
        """

        # if split == True:
        self.Graph = self._split_A_hat(self.adj_user_mat) # list of 
        print("done split matrix torch")
        
        train_data_sub_list = self._split_A_hat_no_torch(self.adj_user_mat)
        
        print("done split matrix no torch")

        pdb.set_trace()
        train_data = []
        for adj_user in train_data_sub_list:
            train_data.append(self.train_sample_gen(adj_user))
        
        
        #padding pos_users and negative users accoording user length
        target_users = torch.Tensor(train_data[0]).long().to(self.device)
        pos_len = torch.Tensor(train_data[2]).long().to(self.device)
        neg_len = torch.Tensor(train_data[4]).long().to(self.device)

        pos_users = torch.Tensor(train_data[1]).long().to(self.device) # padding
        neg_users = torch.Tensor(train_data[3]).long().to(self.device) # padding
        
        # shuffle traindata
        target_users, pos_users, pos_len, neg_users, neg_len = self.shuffle(target_users, pos_users, pos_len, neg_users, neg_len)
        total_batch = len(target_users) // self.bpr_batch_size + 1
        aver_loss = 0.

        start_time = time()
        for (batch_i,
         (batch_users,
          batch_pos_users,
          batch_pos_len,
          batch_neg_users,
          batch_neg_len)) in enumerate(self.minibatch(target_users,
                                                   pos_users,
                                                   pos_len,
                                                   neg_users,
                                                   neg_len,
                                                   batch_size=self.bpr_batch_size)):
            loss, reg_loss = self.bpr_loss(batch_users, batch_pos_users, batch_pos_len, batch_neg_users, batch_neg_len)
            reg_loss = reg_loss*self.weight_decay
            loss = loss + reg_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_cpu = loss.cpu().item()
            aver_loss += loss_cpu

        aver_loss = aver_loss / total_batch

        return f"loss{aver_loss:.3f}-{time()-start_time}"

    def train_model(self,):
        # for epoch training
        for epoch in range(self.epoch):
            loss_info = self.cal_loss()
            print(f"epoch: {epoch}: ", loss_info)
        # return

    def save_embeddings(self):
        """
        save user embeddings and item embeddings
        """

        return

    
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
    # def forward(self, inputs):
    #     X, y = self.inputs_to_device(inputs)
    #     feature_emb_dict = self.embedding_layer(X)
    #     # pdb.set_trace()

    #     mask_label = self.params["add_sequence"]
    #     # pdb.set_trace()
    #     if mask_label == 1:
    #         for idx, (target_field, sequence_field) in enumerate(zip(self.din_target_field, 
    #                                                                 self.din_sequence_field)):
                
    #             # target_field, sequence_field = tuple(target_field), tuple(sequence_field)
    #             target_emb = self.concat_embedding(target_field, feature_emb_dict) #(bs, dim)
    #             sequence_emb = self.concat_embedding(sequence_field, feature_emb_dict) #(bs, sl, dim)
    #             seq_field = np.array([sequence_field]).flatten()[0] # pick a sequence field
    #             padding_idx = self.feature_map.feature_specs[seq_field]['padding_idx']
    #             # mask: (bs, sl), sl=128;
    #             mask = (X[:, self.feature_map.feature_specs[seq_field]["index"]].long() != padding_idx).float() # 128, 'index' means the positions of columns;
    #             # pdb.set_trace()
    #             # 1. attention layer
    #             pooling_emb = self.attention_layers[idx](target_emb, sequence_emb, mask) #(bs, dim)
    #             # 2. max pooling
    #             # pooling_emb, _ = torch.max(mask.unsqueeze(dim=-1) * sequence_emb, dim=1)
    #             # 3. mean pooling
    #             # pooling_emb = torch.max(mask.unsqueeze(dim=-1) * sequence_emb, dim=1)

    #             # pdb.set_trace()
    #             for field, field_emb in zip(np.hstack([sequence_field]),
    #                                         pooling_emb.split(self.embedding_dim, dim=-1)):
    #                 # pdb.set_trace()
    #                 feature_emb_dict[field] = field_emb
    #                 # feature_emb_dict[field] = sequence_emb[:, 0, :]
    #                 # remove for test;
    #                 # pass
    #     feature_emb = self.embedding_layer.dict2tensor(feature_emb_dict)
    #     y_pred = self.dnn(feature_emb.flatten(start_dim=1))
    #     return_dict = {"y_true": y, "y_pred": y_pred}
    #     return return_dict
    # X[11, self.feature_map.feature_specs[seq_field]["index"]]

    # def concat_embedding(self, field, feature_emb_dict):
    #     if type(field) == tuple:
    #         emb_list = [feature_emb_dict[f] for f in field]
    #         return torch.cat(emb_list, dim=-1)
    #     else:
    #         return feature_emb_dict[field]

# class DIN_Attention(nn.Module):
#     def __init__(self, 
#                  embedding_dim=64,
#                  attention_units=[32], 
#                  hidden_activations="ReLU",
#                  output_activation=None,
#                  dropout_rate=0,
#                  batch_norm=False,
#                  use_softmax=False):
#         super(DIN_Attention, self).__init__()
#         self.embedding_dim = embedding_dim
#         self.use_softmax = use_softmax
#         if isinstance(hidden_activations, str) and hidden_activations.lower() == "dice":
#             hidden_activations = [Dice(units) for units in attention_units]
#         self.attention_layer = MLP_Layer(input_dim=4 * embedding_dim,
#                                          output_dim=1,
#                                          hidden_units=attention_units,
#                                          hidden_activations=hidden_activations,
#                                          output_activation=output_activation,
#                                          dropout_rates=dropout_rate,
#                                          batch_norm=batch_norm, 
#                                          use_bias=True)

#     def forward(self, target_item, history_sequence, mask=None):
#         # target_item: b x emd
#         # history_sequence: b x len x emb
#         seq_len = history_sequence.size(1)
#         target_item = target_item.unsqueeze(1).expand(-1, seq_len, -1)
#         attention_input = torch.cat([target_item, history_sequence, target_item - history_sequence, 
#                                      target_item * history_sequence], dim=-1) # b x len x 4*emb
#         attention_weight = self.attention_layer(attention_input.view(-1, 4 * self.embedding_dim))
#         attention_weight = attention_weight.view(-1, seq_len) # b x len
#         if mask is not None:
#             attention_weight = attention_weight * mask.float()
#         if self.use_softmax:
#             if mask is not None:
#                 attention_weight += -1.e9 * (1 - mask.float())
#             attention_weight = attention_weight.softmax(dim=-1)
#         output = (attention_weight.unsqueeze(-1) * history_sequence).sum(dim=1)
#         return output

