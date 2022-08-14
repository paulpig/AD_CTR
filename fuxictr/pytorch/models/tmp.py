    def train_sample_gen(self, adjaceny_ori, max_pos_len = 10):
        """
        构建训练数据, return: (target_item, pos_item_list, neg_item_list)
        input:
            adjaceny_ori: (n_user, n_user)
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


    def train_sample_gen_bipartite(self, adjaceny_ori, max_pos_len = 10, start_index=0):
        """
        构建训练数据, 为什么这么耗时呢?
        input:
            # adjaceny_ori: (n_item, n_user): 0/1
            adjaceny_ori: (batch_n_user, n_item): 0/1
            max_pos_len: the number of postive users
        return:
            return: list of (target_item, pos_user_list, neg_user_list)
        """
        # pdb.set_trace()
        # adj_list_of_list = adjaceny_ori.tolil() #kill
        adj_list_of_list = adjaceny_ori.toarray()
        voc_len = adjaceny_ori.shape[1]
        # item_voc = self.item_voc_len
        # result = []
        total_start = time()
        target_item_id_list = []
        pos_users_list = []
        neg_users_list = []

        print("start preprocessing train data!")
        print("start generating train data!")
        for index, user_ids in enumerate(adj_list_of_list):
            if index % 10000 == 0:
                print("{}/{}".format(index, len(adj_list_of_list)))
            # if np.any(user_ids != 0.): # 很快
            #     item_id = item_id + 1
            user_id_list = np.nonzero(user_ids)[0]
            user_id_list = user_id_list.tolist()
            # pdb.set_trace()
            # sample the target user
            if len(user_id_list) < 1:
                continue
            # else:
            #     random.shuffle(user_id_list)
            #     target_user_id = user_id_list[0]
            
            # remove the target item
            # user_id_list.pop(0)
            target_item_id = index + start_index # 不对;

            # select positive users
            pos_user_id = np.random.choice(user_id_list, 1, replace=False)[0]

            # select negative users
            while True:
                # neg_idx = np.random.randint(0, self.user_voc_len)
                neg_idx = np.random.randint(0, voc_len)
                if neg_idx in user_id_list:
                    continue
                else:
                    break
            
            neg_user_id = neg_idx
            target_item_id_list.append(target_item_id)
            pos_users_list.append(pos_user_id)
            neg_users_list.append(neg_user_id)
        print("cost time of generating dataset: ", time() - total_start)
        return (target_item_id_list, pos_users_list, neg_users_list)


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


            def generate_intermediate_status(self, adj, start_idx):
        adj_list_of_list = adj.toarray()
        target_id_list = []
        pos_ids_list = []
        print("start generating train data!")
        for index, user_ids in enumerate(adj_list_of_list):
            if index % 10000 == 0:
                print("{}/{}".format(index, len(adj_list_of_list)))
            # if np.any(user_ids != 0.): # 很快
            #     item_id = item_id + 1
            user_id_list = np.nonzero(user_ids)[0]
            user_id_list = user_id_list.tolist()
            # pdb.set_trace()
            # sample the target user
            if len(user_id_list) < 1:
                continue

            
            # else:
            #     random.shuffle(user_id_list)
            #     target_user_id = user_id_list[0]
            
            # remove the target item
            # user_id_list.pop(0)
            target_item_id = index + start_idx # 不对;
            
            target_id_list.append(target_item_id)
            pos_ids_list.append(user_id_list)
        
        return target_id_list, pos_ids_list

    def generate_inter_status(self):
        train_data_sub_list = self._split_A_hat_no_torch(self.adj_mat.T) #(user, item)

        print("done split matrix no torch")

        try: #目前存放的数据不正确, 先调试正确;
            # pre_adj_mat = sp.load_npz(self.path_train_data)
            with open(self.inter_path, "rb") as r_f:
                target_id_list, pos_ids_list = pickle.load(r_f)
            print("successfully loading train data...")
            # adjacency = pre_adj_mat
        except :
            # self.raw = np.asarray(data[0])
            # H_T = data_masks(self.raw, n_node)
            print("Start generate train data.")
            train_data = [[] for _  in range(5)]
            target_id_list, pos_ids_list = [], []
            count = 0 # 测试只考虑第一部分的训练数据;
            start_index = 0
            for adj_user in train_data_sub_list:
                # if count == 0:
                print(f"{count}/{len(train_data_sub_list)}")
                # target_user_id_list, pos_users_list, pos_len_list, neg_users_list, neg_len_list = self.train_sample_gen(adj_user)
                 
                batch_target_id_list, batch_pos_ids_list = self.generate_intermediate_status(adj_user, start_index) # target: user,  last two items: items;
                
                # generate intermediate status:  list of user id, list of pos ids;
                # self.generate_intermediate_status(adj_user, start_index, self.inter_path)
                

                start_index += adj_user.shape[0]

                # train_data[0].extend(target_user_id_list)
                # train_data[1].extend(pos_users_list)
                # train_data[2].extend(neg_users_list)

                target_id_list.extend(batch_target_id_list)
                pos_ids_list.extend(batch_pos_ids_list)

                count += 1
            
            with open(self.inter_path, "wb") as w_f:
                pickle.dump((target_id_list, pos_ids_list), w_f)
            pdb.set_trace()

        return


            def cal_loss(self, split=True):
        """
        define node-level and hyperedge-level losses
        """

        # if split == True:
        # self.Graph = self._split_A_hat(self.adj_user_mat) # list of sparse tensors;
        # pdb.set_trace()
        self.Graph = self._split_A_hat(self.ui_mat) # list of sparse tensors;

        print("done split matrix torch")
        
        # train_data_sub_list = self._split_A_hat_no_torch(self.adj_user_mat)
        # train_data_sub_list = self._split_A_hat_no_torch(self.adj_mat) #(item, user)

        train_data_sub_list = self._split_A_hat_no_torch(self.adj_mat.T) #(user, item)

        print("done split matrix no torch")

        # pdb.set_trace()

        # save train data
        try: #目前存放的数据不正确, 先调试正确;
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
            start_index = 0
            for adj_user in train_data_sub_list:
                # if count == 0:
                print(f"{count}/{len(train_data_sub_list)}")
                # target_user_id_list, pos_users_list, pos_len_list, neg_users_list, neg_len_list = self.train_sample_gen(adj_user)
                 
                target_user_id_list, pos_users_list, neg_users_list = self.train_sample_gen_bipartite(adj_user, start_index) # target: user,  last two items: items;
                
                # generate intermediate status:  list of user id, list of pos ids;
                # self.generate_intermediate_status(adj_user, start_index, self.inter_path)
                

                start_index += adj_user.shape[0]

                train_data[0].extend(target_user_id_list)
                train_data[1].extend(pos_users_list)
                train_data[2].extend(neg_users_list)

                count += 1
            
            # pdb.set_trace()
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
        target_users = torch.Tensor(train_data[0]).long().to(self.device) #item
        # pos_len = torch.Tensor(train_data[2]).long().to(self.device)
        # neg_len = torch.Tensor(train_data[4]).long().to(self.device)

        pos_users = torch.Tensor(train_data[1]).long().to(self.device) # padding,
        neg_users = torch.Tensor(train_data[2]).long().to(self.device) # padding
        
        # shuffle traindata
        # target_users, pos_users, pos_len, neg_users, neg_len = self.shuffle(target_users, pos_users, pos_len, neg_users, neg_len)
        target_users, pos_users, neg_users = self.shuffle(target_users, pos_users, neg_users)
        total_batch = len(target_users) // self.bpr_batch_size + 1
        aver_loss = 0.

        start_time = time()
        for (batch_i,
         (batch_users,
          batch_pos_users,
          batch_neg_users)) in enumerate(self.minibatch(target_users,
                                                   pos_users,
                                                   neg_users,
                                                   batch_size=self.bpr_batch_size)):
            # pdb.set_trace()
            # loss, reg_loss = self.bpr_loss(batch_users, batch_pos_users, batch_pos_len, batch_neg_users, batch_neg_len)
            loss, reg_loss = self.bpr_loss_bipartite(batch_users, batch_pos_users, batch_neg_users)
            reg_loss = reg_loss*self.weight_decay
            loss = loss + reg_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_cpu = loss.cpu().item()
            aver_loss += loss_cpu

        aver_loss = aver_loss / total_batch

        return f"loss{aver_loss:.3f}-{time()-start_time}"


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


    def init_parameters(self):
        stdv = 1.0 / math.sqrt(self.emb_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)


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

    
        def bpr_loss_bipartite_cl(self, t_customers, pos_u_list, neg_u_list, pos_uat_list, neg_uat_list, add_cl=True):
        """
        input:
            users: (bs), user_id
            pos: (bs), item_id
            neg: (bs), item_id
        output:
            loss: scalar
        """
        # lightgcn效果不佳, 还能作为对比学习的分支吗?
        # all_user_embeddings_lightgcn = self.computer(self.Graph, torch.cat([self.embedding_user.weight, self.embedding_item.weight], dim=0)) #(user_num + item+num, dim)

        #替换为disentangled learning方法, 观察下效果, 后续添加二部图和disentangled learning之间的对比学习损失函数;
        all_user_embeddings_disen = self.disentangleComputer(self.Graph, self.embedding_user.weight, self.embedding_item.weight) #(user_num + item+num, dim)
        
        
        all_user_embeddings_lightgcn = self.after_mlp

        users_emb_lookup_lightgcn, items_emb_lookup_lightgcn = torch.split(all_user_embeddings_lightgcn, [self.user_voc_len, self.item_voc_len])
        users_emb_lookup_disen, items_emb_lookup_disen = torch.split(all_user_embeddings_disen, [self.user_voc_len, self.item_voc_len])
        # all_user_embeddings = self.computer(self.Graph, self.embedding_user.weight)
        # users_emb = self.embedding_user(users.long()) #(bs, dim)
        # pos_emb   = self.embedding_user(pos.long()) #(bs, max_len, dim)
        # neg_emb   = self.embedding_user(neg.long()) #(bs, max_len, dim)
        # pdb.set_trace()

        user_emb = users_emb_lookup_lightgcn[users.long()]
        # pos_emb = items_emb_lookup_lightgcn[pos.long()]
        # neg_emb = items_emb_lookup_lightgcn[neg.long()]


        user_emb_d = users_emb_lookup_disen[users.long()]
        pos_emb_d = items_emb_lookup_disen[pos.long()]
        neg_emb_d = items_emb_lookup_disen[neg.long()]
        
        # pos_mask = torch.arange(pos.shape[1])[None, :].to(self.device) < pos_len[:, None] #(bs, max_len)
        # neg_mask = torch.arange(neg.shape[1])[None, :].to(self.device) < neg_len[:, None] #(bs, max_len)

        # mean pooling
        # pos_emb_pooling = (pos_mask[:, :, None] * pos_emb).sum(1) #(bs, dim)
        # neg_emb_pooling = (neg_mask[:, :, None] * neg_emb).sum(1)

        # pos_emb_pooling = pos_emb
        # neg_emb_pooling = neg_emb
        user_emb_t = user_emb_d
        pos_emb_pooling = pos_emb_d
        neg_emb_pooling = neg_emb_d

        pos_scores= torch.sum(user_emb_t*pos_emb_pooling, dim=1) #(bs)
        neg_scores= torch.sum(user_emb_t*neg_emb_pooling, dim=1)

        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        # loss = torch.sum(nn.functional.softplus(neg_scores - pos_scores))

        reg_loss = (1/2)*(user_emb_t.norm(2).pow(2) + 
                          pos_emb_pooling.norm(2).pow(2) + 
                          neg_emb_pooling.norm(2).pow(2))/float(len(users))
        
        # pdb.set_trace()
        # add cl loss
        if add_cl:
            # add cl loss based items
            # cl_loss = self.loss_contrastive(user_emb, user_emb_d, temp_value=0.1)

            # v1 triple loss
            # cl_loss = self.loss_contrastive_triple(user_emb, user_emb_d, temp_value=0.1)
            # cl_loss = self.loss_contrastive_triple(user_emb_d, user_emb) # local 与 global之间的权重可调整;
            cl_loss = self.loss_contrastive_triple(user_emb_d, user_emb, add_local=True, add_global=False) # local 与 global之间的权重可调整;
            # cl_loss = self.loss_contrastive_triple(users_emb_lookup_disen, users_emb_lookup_lightgcn, temp_value=0.1) # local 与 global之间的权重可调整;

            # MLP之后的表征和过了GNN后表征之间的对比学习;
            # cl_loss = self.loss_contrastive_triple(user_emb_d, user_emb) # local 与 global之间的权重可调整;
            # user_cl_loss = self.ssl_layer_loss_infoNce(user_emb, user_emb_d, )
            # v2 infoNCE loss
            # cl_loss = self.ssl_layer_loss_infoNce(user_emb_d, user_emb, users_emb_lookup_lightgcn) # OOM

            # ssl_loss = ssl_reg * (ssl_loss_user + item_alpha * ssl_loss_item)
            # ssl_reg=1e-2, item_alpha=1.5
            # pdb.set_trace()
            # loss += 1e-5 * cl_loss
            # cl_loss = 5e-6 * cl_loss
            # cl_loss = 1e-5 * cl_loss
            cl_loss = 5e-5 * cl_loss
            loss += cl_loss

            # add weight regularation;
            # for key in self.weights:
            #     reg_loss += 0.001*tf.nn.l2_loss(self.weights[key])
            return loss, reg_loss, cl_loss
            

        return loss, reg_loss, 0.



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
