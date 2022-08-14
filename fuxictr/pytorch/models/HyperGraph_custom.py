# from copyreg import pickle
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
import copy
import pdb
import pickle
import h5py


class HyperGraphCustom(nn.Module):
    def __init__(self,
                #  adj_user_mat,
                 gpu=-1,
                 graph_layer=1,
                 learning_rate=1e-3, 
                 embedding_dim=10, 
                 epoch=100,
                 bpr_batch_size=10000,
                 weight_decay=1e-5):
        super(HyperGraphCustom, self).__init__()
        
        # parameters
        self.emb_size = embedding_dim
        self.layers = graph_layer
        # self.dataset = dataset

        # path_origin = "../data/Taobao/taobao_ori/user_id_Item_id_net_v2_ori_num_replace_customer.npz" #构建customer-user hypergraph
        path_origin = "../data/Taobao/taobao_ori/user_id_Item_id_net_v2_ori_num_replace_customer_true.npz" #构建customer-user hypergraph
        adj_mat = sp.load_npz(path_origin)
        self.item_voc_len, self.user_voc_len = adj_mat.get_shape()
        self.adj_mat = adj_mat #(item, user)

        # path = "../data/Taobao/taobao_ori/user_id_Item_id_pre_adj_mat_filter_top_100_ori_num_replace_customer.npz" #超图卷积处理后的user-user graph, 只保留每个用户前N个相似的用户; customer-user graph;
        path = "../data/Taobao/taobao_ori/user_id_Item_id_pre_adj_mat_filter_top_30_ori_num_replace_customer_true.npz" #超图卷积处理后的user-user graph, 只保留每个用户前N个相似的用户; customer-user graph;


        self.path_train_data = "../data/Taobao/taobao_ori/user_item_train_data_top30_ori_num_replace_customer_true.npz" #训练数据, <user, in_session_positive_users>, <user, out_session_negative_users>; customer-user graph;
        


        # self.path_dict_path = "../data/Taobao/taobao_ori/userItmeDict_v2_ori_num_replace_customer.pk" # 存在ID与index的字典, <user_id, index>, <item_id, index>
        self.path_dict_path = "../data/Taobao/taobao_ori/userItmeDict_v2_ori_num_replace_customer_true.pk" # 存在ID与index的字典, <user_id, index>, <item_id, index>
       

        # self.path_pretrain_user_emb_path = "../data/Taobao/taobao_ori/user_emb_hypergraph_dim10_final_output_epoch_20_ori_num_input_v2_replace_customer.h5" # 训练20个epoch, 保存输出的user representation;
        self.path_pretrain_user_emb_path = "../data/Taobao/taobao_ori/user_emb_hypergraph_dim10_final_output_epoch_20_ori_num_top30_customer_user_true.h5" # 训练20个epoch, 保存输出的user representation;

        # path_v2 = "../data/Taobao/taobao_ori/user_id_Item_id_pre_adj_mat_item2item.npz" #与物品相关的, 暂时不考虑;
        self.adj_user_mat = self.normalizeAdj(self.adj_mat, path) #(user, user)
        # self.adj_item_mat = self.normalizeAdj(self.adj_mat.T, path_v2) #(item, item)
        # pdb.set_trace()
        self.embedding_user = nn.Embedding(self.user_voc_len, self.emb_size)
        self.embedding_item = nn.Embedding(self.item_voc_len, self.emb_size)
        # self.compile(kwargs["optimizer"], loss=kwargs["loss"], lr=learning_rate)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.epoch = epoch
        self.folds = 20
        # self.folds = 1000
        self.weight_decay = weight_decay
        # self.kwargs = kwargs
        self.bpr_batch_size = bpr_batch_size
        # self.init_parameters() # replace other ways
        self.init_parameters_v2()
        self.device = get_device(gpu)
        # self.model_to_device()
        self.to(device=self.device)

    def init_parameters(self):
        stdv = 1.0 / math.sqrt(self.emb_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
    
    def init_parameters_v2(self, std_val=1e-4):
        # stdv = 1.0 / math.sqrt(self.emb_size)
        for weight in self.parameters():
            # weight.data.uniform_(-stdv, stdv)
            weight.data.normal_(std=std_val)
            # nn.init.normal_(embedding_matrix.weight, std=std_val) #默认初始化;

    def trans_to_cuda(self, variable):
        pdb.set_trace()
        if torch.cuda.is_available():
            return variable.to(device=self.device)
        else:
            return variable

    def trans_to_cpu(self, variable):
        if torch.cuda.is_available():
            return variable.cpu()
        else:
            return variable
    
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
    
    def computer(self, adjacency_list, embedding):
        """
        intput:
            # adjacency: (user_num/item_num, user_num/item_num)
            adjacency_list: list of adj
            embedding: (user_num/item_num, dim)
        output:
            (user_num/item_num, dim)
        """
        # adjacency = self.adj_user_mat
        item_embeddings = embedding
        item_embedding_layer0 = item_embeddings
        final = [item_embedding_layer0]
        for i in range(self.layers):
            # item_embeddings = torch.sparse.mm(self.trans_to_cuda(adjacency), item_embeddings)
            # pdb.set_trace() 
            item_embeddings_list = []
            # for sub_adjacency in self.Graph:
            for sub_adjacency in adjacency_list:
                item_embeddings_sub = torch.sparse.mm(sub_adjacency, item_embeddings) #adjacency: (N, user_num);  item_emb:; (user_num, dim)
                # add l2 normalize
                # item_embeddings_sub = torch.nn.functional.normalize(item_embeddings_sub, p=2, dim=1)
                item_embeddings_list.append(item_embeddings_sub)
            item_embeddings = torch.cat(item_embeddings_list, dim=0) #(item_num, dim)
            final.append(item_embeddings)
      #  final1 = trans_to_cuda(torch.tensor([item.cpu().detach().numpy() for item in final]))
      #  item_embeddings = torch.sum(final1, 0)
        item_embeddings = np.sum(final, 0) / (self.layers+1)
        # item_embeddings = np.sum(final, 0)

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
        # pdb.set_trace()
        # adj_list_of_list = adjaceny_ori.tolil() #kill
        adj_list_of_list = adjaceny_ori.toarray()
        max_len = max_pos_len
        item_voc = len(adj_list_of_list)
        # item_voc = self.item_voc_len
        # result = []
        total_start = time()
        target_user_id_list = []
        pos_users_list = []
        pos_len_list = []
        neg_users_list = []
        neg_len_list = []

        print("start preprocessing train data!")
        # 过滤序列长度小于0的数据
        neg_list = []
        neg_list_idx = []
        for index, data in enumerate(adj_list_of_list):
            # print(index)
            # if sum(data) > 0: # 慢
            if np.any(data != 0.): # 很快
            # if len(np.nonzero(data)[0]) > 0: # 快
                neg_list.append(data)
                neg_list_idx.append(index)
        # pdb.set_trace()
        print("start generating train data!")
        # pdb.set_trace()
        item_id = -1
        for index, user_ids in enumerate(adj_list_of_list):
            if index % 10000 == 0:
                print("{}/{}".format(index, len(adj_list_of_list)))
            
            if np.any(user_ids != 0.): # 很快
                item_id = item_id + 1
            
            user_id_list = np.nonzero(user_ids)[0]
            user_id_list = user_id_list.tolist()
            # pdb.set_trace()
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
                # pos_users = np.random.choice(user_id_list, max_len, replace=False)
                pos_users = user_id_list[:max_len]
            else:
                pos_users = user_id_list
            pos_len = len(pos_users)
            # padding
            if len(pos_users) < max_len:
                pos_users += [0] * (max_len - len(pos_users))
            # 2. random sample users from the other session for negative samples
            # neg_list_idx.copy()
            # neg_list_cp = copy.deepcopy(neg_list)
            # if index in neg_list_idx:
            #     neg_list_cp.remove(user_ids)
            
            while True:
                neg_idx = np.random.randint(0, len(neg_list))
                if neg_idx == item_id:
                    continue
                else:
                    break
                
            # while True:
            #     session_id = np.random.randint(0, item_voc)
            #     if session_id == index:
            #         continue
            #     else:
            #         if len(np.nonzero(adj_list_of_list[session_id])[0]) > 0:
            #             break
            #         else:
            #             continue
            
            neg_sess = np.nonzero(neg_list[neg_idx])[0].tolist()

            # pdb.set_trace()
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

            # del neg_list_cp
            # pdb.set_trace()
        # pdb.set_trace()
        assert (item_id+1) == len(neg_list)
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
        # test only subgraph
        # all_user_embeddings = self.computer(self.adj_user_mat, self.embedding_user.weight)
        all_embs = []
        # for sub_graph in self.Graph:
        all_user_embeddings = self.computer(self.Graph, self.embedding_user.weight)
        # all_user_embeddings = self.computer(self.Graph, self.embedding_user.weight)
        # users_emb = self.embedding_user(users.long()) #(bs, dim)
        # pos_emb   = self.embedding_user(pos.long()) #(bs, max_len, dim)
        # neg_emb   = self.embedding_user(neg.long()) #(bs, max_len, dim)

        users_emb = all_user_embeddings[users.long()]
        pos_emb = all_user_embeddings[pos.long()]
        neg_emb = all_user_embeddings[neg.long()]

        pos_mask = torch.arange(pos.shape[1])[None, :].to(self.device) < pos_len[:, None] #(bs, max_len)
        neg_mask = torch.arange(neg.shape[1])[None, :].to(self.device) < neg_len[:, None] #(bs, max_len)

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
        self.Graph = self._split_A_hat(self.adj_user_mat) # list of sparse tensors;
        print("done split matrix torch")
        
        train_data_sub_list = self._split_A_hat_no_torch(self.adj_user_mat)
        print("done split matrix no torch")

        # pdb.set_trace()

        # save train data
        try:
            # pre_adj_mat = sp.load_npz(self.path_train_data)
            with open(self.path_train_data, "rb") as r_f:
                train_data = pickle.load(r_f)
            print("successfully loading train data...")
            # adjacency = pre_adj_mat
        except :
            # self.raw = np.asarray(data[0])
            # H_T = data_masks(self.raw, n_node)
            print("Start generate train data.")
            train_data = [[] for _  in range(5)]
            count = 0 # 测试只考虑第一部分的训练数据;
            for adj_user in train_data_sub_list:
                # if count == 0:
                print(f"{count}/{len(train_data_sub_list)}")
                target_user_id_list, pos_users_list, pos_len_list, neg_users_list, neg_len_list = self.train_sample_gen(adj_user)
                train_data[0].extend(target_user_id_list)
                train_data[1].extend(pos_users_list)
                train_data[2].extend(pos_len_list)
                train_data[3].extend(neg_users_list)
                train_data[4].extend(neg_len_list)
                # else:
                #     break
                count += 1
            with open(self.path_train_data, "wb") as w_f:
                pickle.dump(train_data, w_f)

        # train_data = [[] for _  in range(5)]
        # count = 0 # 测试只考虑第一部分的训练数据;
        # for adj_user in train_data_sub_list:
        #     if count == 0:
        #         target_user_id_list, pos_users_list, pos_len_list, neg_users_list, neg_len_list = self.train_sample_gen(adj_user)
        #         train_data[0].extend(target_user_id_list)
        #         train_data[1].extend(pos_users_list)
        #         train_data[2].extend(pos_len_list)
        #         train_data[3].extend(neg_users_list)
        #         train_data[4].extend(neg_len_list)
        #     else:
        #         break
        #     count += 1

        # train_data.extend(self.train_sample_gen(train_data_sub_list[0]))
        print("train sample data: done!")
        # pdb.set_trace()
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
            # pdb.set_trace()
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
        with open(self.path_dict_path, 'rb') as r_f:
            userAt2id, itemAt2id = pickle.load(r_f)
        
        # 1. original user embeddings
        # user_embeddings = self.embedding_user.weight.cpu().detach().numpy()

        # 2. original item embeddings
        user_embeddings = self.computer(self.Graph, self.embedding_user.weight).detach().cpu().numpy()

        user_id_list = []
        user_rep_list = []
        for key, index in userAt2id.items():
            user_rep = user_embeddings[index]
            user_id = key.strip().split("_")[1]
            # save embedding
            user_id_list.append(user_id)
            user_rep_list.append(user_rep)
        
        # pdb.set_trace() # f['value'][10]
        with h5py.File(self.path_pretrain_user_emb_path, 'w') as hf:
            hf.create_dataset("key", data=user_id_list)
            hf.create_dataset("value", data=user_rep_list)
        # pdb.set_trace()
        print("successfully saving user embeddings.")

    
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

