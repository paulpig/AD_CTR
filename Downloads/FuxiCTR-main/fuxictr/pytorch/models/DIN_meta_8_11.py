# =========================================================================
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

""" This model implements the paper "Zhou et al., Deep Interest Network for 
    Click-Through Rate Prediction, KDD'2018".
    [PDF] https://arxiv.org/pdf/1706.06978.pdf
    [Code] https://github.com/zhougr1993/DeepInterestNetwork
"""

from tkinter import W
import torch
from torch import nn
import numpy as np
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import EmbeddingDictLayer, MLP_Layer, Dice
import pdb
from ..torch_utils import get_device, get_optimizer, get_loss_fn, get_regularizer
import torch.nn.functional as F

class DIN_Meta_JOINT(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="DIN",
                 gpu=-1, 
                 task="binary_classification",
                 graph_embedding_dim=64,
                 dnn_hidden_units=[512, 128, 64],
                 dnn_activations="ReLU",
                 attention_hidden_units=[64],
                 attention_hidden_activations="Dice",
                 attention_output_activation=None,
                 attention_dropout=0,
                 learning_rate=1e-3, 
                 embedding_dim=10, 
                 net_dropout=0, 
                 batch_norm=False, 
                 din_target_field=[("item_id", "cate_id")],
                 din_sequence_field=[("click_history", "cate_history")],
                 din_use_softmax=False,
                 embedding_regularizer=None, 
                 net_regularizer=None, 
                 pre_model=None,
                 **kwargs):
        super(DIN_Meta_JOINT, self).__init__(feature_map, 
                                  model_id=model_id, 
                                  gpu=gpu, 
                                  embedding_regularizer=embedding_regularizer, 
                                  net_regularizer=net_regularizer,
                                  **kwargs)
        # pdb.set_trace()
        if not isinstance(din_target_field, list):
            din_target_field = [din_target_field]
        self.din_target_field = din_target_field
        if not isinstance(din_sequence_field, list):
            din_sequence_field = [din_sequence_field]
        self.din_sequence_field = din_sequence_field
        assert len(self.din_target_field) == len(self.din_sequence_field), \
            "Error: len(self.din_target_field) != len(self.din_sequence_field)"
        
        # convert inner list to tuple
        if type(self.din_target_field[0]) == list:
            self.din_target_field = [tuple(item) for item in self.din_target_field]
        if type(self.din_sequence_field[0]) == list:
            self.din_sequence_field = [tuple(item) for item in self.din_sequence_field]
        
        if isinstance(dnn_activations, str) and dnn_activations.lower() == "dice":
            dnn_activations = [Dice(units) for units in dnn_hidden_units]
        self.params = kwargs
        self.feature_map = feature_map
        self.embedding_dim = embedding_dim
        self.embedding_layer = EmbeddingDictLayer(feature_map, embedding_dim, use_pretrain=self.params["use_pretrain"])
        # pdb.set_trace()
        # self.attention_layers = nn.ModuleList(
        #     [DIN_Attention(embedding_dim * len(target_field) if type(target_field) == tuple \
        #                                                      else embedding_dim,
        #                    attention_units=attention_hidden_units,
        #                    hidden_activations=attention_hidden_activations,
        #                    output_activation=attention_output_activation,
        #                    dropout_rate=attention_dropout,
        #                    batch_norm=batch_norm,
        #                    use_softmax=din_use_softmax)
        #      for target_field in self.din_target_field])
        # self.dnn = MLP_Layer(input_dim=feature_map.num_fields * embedding_dim,
        # self.dnn = MLP_Layer(input_dim=(feature_map.num_fields + 1)* embedding_dim,
        self.use_vae = self.params["add_vae"]
        self.add_side_info = self.params["add_side_info"]

        # if self.use_vae:
        #     self.dnn_ori = MLP_Layer(input_dim=(feature_map.num_fields + 3)* embedding_dim,
        #                         output_dim=1,
        #                         hidden_units=dnn_hidden_units,
        #                         hidden_activations=dnn_activations,
        #                         output_activation=self.get_output_activation(task), 
        #                         dropout_rates=net_dropout,
        #                         batch_norm=batch_norm, 
        #                         use_bias=True)
        # else:
        self.dnn_ori = MLP_Layer(input_dim=(feature_map.num_fields + 2)* embedding_dim,
                            output_dim=1,
                            hidden_units=dnn_hidden_units,
                            hidden_activations=dnn_activations,
                            output_activation=self.get_output_activation(task), 
                            dropout_rates=net_dropout,
                            batch_norm=batch_norm, 
                            use_bias=True)
        
        # self.dnn = MLP_Layer(input_dim=(feature_map.num_fields -1) * embedding_dim,
        # self.dnn = MLP_Layer(input_dim=(feature_map.num_fields + 2) * embedding_dim,
        #                      output_dim=embedding_dim,
        #                      hidden_units=dnn_hidden_units,
        #                      hidden_activations=dnn_activations,
        #                     #  output_activation='RELU', 
        #                      output_activation=None, 
        #                      dropout_rates=net_dropout,
        #                      batch_norm=batch_norm, 
        #                      use_bias=True)
        
        self.graph_embedding_dim = graph_embedding_dim
        # self.pretrain_mlp_freeze = torch.nn.Linear(embedding_dim, embedding_dim)
        # self.pretrain_mlp_freeze = torch.nn.Linear(embedding_dim, embedding_dim)
        # para_names = []
        # for n, p in self.named_parameters():
        #     para_names.append(n)
        #     if 'pretrain_mlp_freeze' in n:
        #         # pdb.set_trace()
        #         p.requires_grad = False
        # pdb.set_trace()

        # print(list(self.named_parameters()))
        # pdb.set_trace()
        # pdb.set_trace()
        # self.dnn_output = MLP_Layer(input_dim=embedding_dim,
        #                      output_dim=1,
        #                      hidden_units=[],
        #                      hidden_activations=None,
        #                      output_activation=self.get_output_activation(task), 
        #                      dropout_rates=net_dropout,
        #                      batch_norm=batch_norm, 
        #                      use_bias=True)

        if "use_meta" in kwargs:
            self.use_meta = kwargs["use_meta"]
        else:
            self.use_meta = False
        
        if "use_attention" in kwargs:
            self.use_attention = kwargs["use_attention"]
        else:
            self.use_attention = False
        
        if "use_pretrain" in kwargs:
            self.use_pretrain = kwargs["use_pretrain"]
        else:
            self.use_pretrain = False
        
        if "concat_other_feat" in kwargs:
            self.concat_other_feat = kwargs["concat_other_feat"]
        else:
            self.concat_other_feat = False


        # self.meta_net = MetaNet(embedding_dim, embedding_dim)
        # # self.meta_net_v2 = MetaNet(embedding_dim, embedding_dim)
        # self.meta_net_flat = MetaNetFlat(embedding_dim, embedding_dim)


        # self.pretrain_emb_mlp = MetaNetFlat(embedding_dim, embedding_dim)
        self.pretrain_emb_mlp = MetaNetFlat(self.graph_embedding_dim, embedding_dim)
        self.pretrain_c_emb_mlp = MetaNetFlat(self.graph_embedding_dim, embedding_dim)
        # self.meta_net_flat_v1 = MetaNetFlatV1(embedding_dim*2, embedding_dim)
        self.meta_net_flat_v1 = MetaNetFlatV1(embedding_dim, embedding_dim)
        self.meta_net_flat_v2 = MetaNetFlatV1(embedding_dim, embedding_dim)
        self.output_linear = torch.nn.Linear(embedding_dim, 1)

        self.pretrain_vae_emb_mlp = MetaNetFlat(self.graph_embedding_dim, embedding_dim)

        
        # self.test = torch.nn.Linear(embedding_dim, embedding_dim)
        # self.layer_norm = torch.nn.LayerNorm(embedding_dim)
        # self.gate_layer = torch.nn.Linear(embedding_dim*2, 1, True)

        self.pre_model = pre_model

        # pdb.set_trace()
        # para_names = []
        # for n, p in self.named_parameters():
        #     para_names.append(n)

        # para_names_v2 = []
        # for n, p in self.pre_model.named_parameters():
        #     para_names_v2.append(n)
        # pdb.set_trace()
        self.pre_encoder_model = kwargs["encoder_model"]

        self.compile(kwargs["optimizer"], loss=kwargs["loss"], lr=learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def reload_pretrain_model(self):
        return self.pre_model

    
    def compile(self, optimizer, loss, lr):
        self.optimizer = get_optimizer(optimizer, self.parameters(), lr)
        self.loss_fn = get_loss_fn(loss)
    
    def reset_parameters(self):
        def reset_param(m):
            if type(m) == nn.ModuleDict:
                for k, v in m.items():
                    if type(v) == nn.Embedding:
                        # 防止共享的表征重新初始化; 最基础的DNN + meta模型, 不初始化user embedding;
                        # if "share_embedding" in self._feature_map.feature_specs[k]:
                        #     continue

                        # 跳过预训练的embedding layer;
                        # if "pretrained_emb" in self._feature_map.feature_specs[k]: # skip pretrained
                        #     # click_user_sequence
                        #     continue
                        # tmp
                        if k == "userid_cp":
                            continue
                        if self._embedding_initializer is not None:
                            try:
                                if v.padding_idx is not None:
                                    # the last index is padding_idx
                                    initializer = self._embedding_initializer.replace("(", "(v.weight[0:-1, :],")
                                else:
                                    initializer = self._embedding_initializer.replace("(", "(v.weight,")
                                eval(initializer)
                            except:
                                raise NotImplementedError("embedding_initializer={} is not supported."\
                                                          .format(self._embedding_initializer))
            if type(m) == nn.Linear:
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0)
        self.apply(reset_param)

    # def set_reindex_user_ids(self, reindex_user_ids):
    #     self.reindex_user_ids = reindex_user_ids
    #     return 
    
    # def set_reindex_customer_ids(self, reindex_customer_ids):
    #     self.reindex_customer_ids = reindex_customer_ids
    #     return 

    def forward(self, inputs):
        X, y = self.inputs_to_device(inputs)
        feature_emb_dict = self.embedding_layer(X)
        # pdb.set_trace() # self.embedding_layer.embedding_hooks['userid']
        # assert len(self.din_target_field) == 2

        
        self.pre_model.forward_embeddings(save_model_name=self.pre_encoder_model, use_vae=self.use_vae, add_side_info=self.add_side_info, vae_merge_side_info=False) #以u_c graph为输出;
        # self.pre_model.forward_embeddings(save_model_name=self.pre_encoder_model, is_add_uat=True) # 以u_uat graph为输出;
        # self.embedding_layer.other_user_emb_layer = self.pre_model.user_embeddings[self.reindex_user_ids]
        self.embedding_layer.other_user_emb_layer = self.pre_model.user_embeddings
        # self.embedding_layer.other_customer_emb_layer = self.pre_model.item_embeddings[self.reindex_customer_ids]
        self.embedding_layer.other_customer_emb_layer = self.pre_model.item_embeddings

        # if self.use_vae:
        #     self.embedding_layer.other_user_vae_emb_layer = self.pre_model.vae_user_emb
        #     feature_spec = self.embedding_layer._feature_map.feature_specs['userid']
        #     inp = X[:, feature_spec["index"]].long()
        #     user_vae_embedding_hypergraph = self.embedding_layer.other_user_vae_emb_layer[inp] #消融实验;
        #     convert_vae_emb_hyper = self.pretrain_vae_emb_mlp(user_vae_embedding_hypergraph)
        #     feature_emb_dict['vae_hyper_emb'] = convert_vae_emb_hyper
            # pdb.set_trace()

        # if self.use_meta and self.use_pretrain:
        # add pretrain user emb
        feature_spec = self.embedding_layer._feature_map.feature_specs['userid']
        inp = X[:, feature_spec["index"]].long()
        # user_embedding_ori = self.embedding_layer.embedding_layer['userid'](inp)
        # user_embedding_hypergraph = self.embedding_layer.embedding_layer['userid_cp'](inp) #消融实验;
        user_embedding_hypergraph = self.embedding_layer.other_user_emb_layer[inp] #消融实验;
        # user_embedding_hypergraph = self.embedding_layer.embedding_layer['userid'](inp)

        feature_spec = self.embedding_layer._feature_map.feature_specs['customer']
        inp = X[:, feature_spec["index"]].long()
        customer_embedding_hypergraph = self.embedding_layer.other_customer_emb_layer[inp] #消融实验;


        mask_label = self.params["add_sequence"]
        assert ((mask_label == 1) and (self.use_attention or self.use_meta)) or (mask_label == 0)
             
        # pdb.set_trace()
        # remove 'click_user_sequence', 已转化为其他特征;
        if 'click_user_sequence' in feature_emb_dict:
            del feature_emb_dict['click_user_sequence']
        
        # del feature_emb_dict['userid']

        convert_emb_hyper = self.pretrain_emb_mlp(user_embedding_hypergraph)
        feature_emb_dict['hyper_emb'] = convert_emb_hyper


        convert_customer_emb_hyper = self.pretrain_c_emb_mlp(customer_embedding_hypergraph)
        feature_emb_dict['c_hyper_emb'] = convert_customer_emb_hyper
        # user_embedding_hypergraph = self.pretrain_mlp_freeze(user_embedding_hypergraph)
        # # add hypergraph emb
        # feature_emb_dict["hyper_emb"] = user_embedding_hypergraph
        # pdb.set_trace()
        # convert_emb_hyper = self.pretrain_emb_mlp(user_embedding_hypergraph)
        # # convert_emb_hyper = self.test(user_embedding_hypergraph)
        # # convert_emb_hyper = user_embedding_hypergraph
        # feature_emb_dict["hyper_emb"] = convert_emb_hyper

        # pdb.set_trace()
        feature_emb = self.embedding_layer.dict2tensor(feature_emb_dict)
        # list(feature_emb_dict.keys())[14] #
        # feature_emb_dict['click_user_sequence']
        # feature_emb_dict['adgroup_id']
        # feature_emb_dict['userid']
        # feature_emb_dict['age_level']
        # feature_emb_dict['campaign_id']
        # pdb.set_trace()
        # test: self.embedding_layer.embedding_layer['userid'].weight[200:400]
        # test: self.embedding_layer.embedding_layer['adgroup_id'].weight[200:400]
        # 构建关于user embedding与pooling的meta net;

        
        # tmp = torch.nn.functional.normalize(feature_emb_dict['userid'], p=2, dim=1)
        # tmp1 = torch.nn.functional.normalize(feature_emb_dict['adgroup_id'], p=2, dim=1)
        # 1. 尝试;
        # output_emb = self.dnn(feature_emb.flatten(start_dim=1)) #(bs, emb_size)
        # meta_mapping = self.meta_net_flat(feature_emb_dict['userid']) #(bs, emb_size, emb_size)
        # pooling_emb = torch.bmm(output_emb.unsqueeze(1), meta_mapping).squeeze(1) #(bs, dim)
        # y_pred = self.dnn_output(pooling_emb) #(bs)

        # pdb.set_trace()
        # 为什么user embedding的初始化效果很差.
        # pdb.set_trace()
        y_pred = self.dnn_ori(feature_emb.flatten(start_dim=1))

        # =========== meta learning =========================
        # hidden_emb = self.dnn(feature_emb.flatten(start_dim=1)) #(bs, dim)

        # # POSO method
        # # weight = nn.functional.sigmoid(self.meta_net_flat(user_embedding_hypergraph)) #(bs, dim)
        # # hidden_emb = hidden_emb * weight * 2.0
        # # y_pred = nn.functional.sigmoid(self.output_linear(hidden_emb))
        
        # # meta learning
        # # pooling_emb_mapping = self.meta_net_flat_v1(user_embedding_hypergraph) #(bs, dim, dim)
        # # pooling_emb_mapping, bias_emb_mapping = self.meta_net_flat_v1(torch.cat([convert_emb_hyper, convert_customer_emb_hyper], dim=-1)) #(bs, dim, dim)
        # pooling_emb_mapping, bias_emb_mapping = self.meta_net_flat_v1(convert_emb_hyper) #(bs, dim, dim)
        # pooling_customer_emb_mapping, bias_customer_emb_mapping = self.meta_net_flat_v2(convert_customer_emb_hyper) #(bs, dim, dim)
        # # pdb.set_trace()
        # hidden_emb_v1 = torch.bmm(hidden_emb.unsqueeze(1), pooling_emb_mapping).squeeze(1) + bias_emb_mapping #(bs, dim), 加上bias效果非常好...
        # hidden_emb_v2 = torch.bmm(hidden_emb.unsqueeze(1), pooling_customer_emb_mapping).squeeze(1) + bias_customer_emb_mapping #(bs, dim), 加上bias效果非常好...
        # # y_pred = torch.nn.ReLU()(self.output_linear(hidden_emb + hidden_emb_v1)) # 添加bias效果很好;
        # y_pred = nn.functional.sigmoid(self.output_linear(hidden_emb + hidden_emb_v1 + hidden_emb_v2)) # 添加bias效果很好;
        # # pdb.set_trace()

        return_dict = {"y_true": y, "y_pred": y_pred}
        return return_dict
    
    def concat_embedding(self, field, feature_emb_dict):
        if type(field) == tuple:
            emb_list = [feature_emb_dict[f] for f in field]
            return torch.cat(emb_list, dim=-1)
        else:
            return feature_emb_dict[field]

    def loss_contrastive(self, tensor_anchor_v1, tensor_anchor_v2, temp_value=0.1):
        """
        tensor_anchor_v1: (bs, dim)
        tensor_anchor_v2: (bs, dim)
        """   
        head_feat_1, head_feat_2 = F.normalize(tensor_anchor_v1, dim=1), F.normalize(tensor_anchor_v2, dim=1)
        # all_score = torch.exp(torch.sum(tensor_anchor*tensor_all, dim=1)/temp_value).view(-1, 1+self.num_neg)
        pos_score = (head_feat_1 * head_feat_2).sum(-1) #(bs)
        pos_item = torch.cat([head_feat_1, head_feat_2], dim=0) #(2*bs, dim)
        # pdb.set_trace()
        all_tensors = pos_item.unsqueeze(0).repeat(tensor_anchor_v1.size(0)*2, 1, 1) #(2*bs, 2*bs, dim)
        all_scores = (pos_item.unsqueeze(1) * all_tensors).sum(-1) #(2*bs, 2*bs)
        all_scores_mask = all_scores + (torch.eye(tensor_anchor_v1.size(0)*2) * (-1e8)).to(device=self.device)

        all_score = torch.sum(torch.exp(all_scores_mask/temp_value), dim=1) #(2*bs)
        pos_score = torch.exp(torch.cat([pos_score, pos_score], dim=0)) #(2*bs)

        cl_loss = (-torch.log(pos_score / all_score)).mean()

        return cl_loss

class MetaNet(torch.nn.Module):
    def __init__(self, emb_dim, meta_dim):
        super().__init__()
        self.event_K = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.ReLU(),
                                           torch.nn.Linear(emb_dim, 1, False))
        self.event_softmax = torch.nn.Softmax(dim=1)
        self.decoder = torch.nn.Sequential(torch.nn.Linear(emb_dim, meta_dim), torch.nn.ReLU(),
                                           torch.nn.Linear(meta_dim, emb_dim * emb_dim))
        self.emb_dim = emb_dim

    def forward(self, emb_fea, mask):
        """
        input:
            emb_fea: (bs, sl, dim)
            mask: (bs, sl)
        return:
            (bs, emb_dim, emb_dim)
        """
        # mask = (seq_index == 0).float()
        mask = 1.0 - mask # reverse
        event_K = self.event_K(emb_fea) #(bs, sl, 1)
        t = event_K - torch.unsqueeze(mask, 2) * 1e8 #(bs, sl, 1)
        att = self.event_softmax(t)
        his_fea = torch.sum(att * emb_fea, 1) #(bs, dim)
        output = self.decoder(his_fea) #(bs, dim*dim)
        return output.reshape(-1, self.emb_dim, self.emb_dim)


class MetaNetFlat(torch.nn.Module):
    def __init__(self, emb_dim, meta_dim):
        super().__init__()
        # self.event_K = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.ReLU(),
        #                                    torch.nn.Linear(emb_dim, 1, False))
        # self.event_softmax = torch.nn.Softmax(dim=1)
        self.decoder = torch.nn.Sequential(torch.nn.Linear(emb_dim, meta_dim), torch.nn.ReLU(),
                                           torch.nn.Linear(meta_dim, meta_dim))
        # self.emb_dim = emb_dim
        self.emb_dim = meta_dim

    def forward(self, emb_fea, mask=None):
        """
        input:
            emb_fea: (bs, dim)
            # mask: (bs, sl)
        return:
            (bs, emb_dim, emb_dim)
        """
        # mask = (seq_index == 0).float()
        # mask = 1.0 - mask # reverse
        # event_K = self.event_K(emb_fea) #(bs, sl, 1)
        # t = event_K - torch.unsqueeze(mask, 2) * 1e8 #(bs, sl, 1)
        # att = self.event_softmax(t)
        # his_fea = torch.sum(att * emb_fea, 1) #(bs, dim)
        output = self.decoder(emb_fea) #(bs, dim*dim)
        # return output.reshape(-1, self.emb_dim, self.emb_dim)
        return output.reshape(-1, self.emb_dim)


class MetaNetFlatV1(torch.nn.Module):
    def __init__(self, emb_dim, meta_dim):
        super().__init__()
        # self.event_K = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.ReLU(),
        #                                    torch.nn.Linear(emb_dim, 1, False))
        # self.event_softmax = torch.nn.Softmax(dim=1)
        # self.decoder = torch.nn.Sequential(torch.nn.Linear(emb_dim, meta_dim), torch.nn.ReLU(),
        #                                    torch.nn.Linear(meta_dim, emb_dim*emb_dim))

        self.decoder = torch.nn.Sequential(torch.nn.Linear(emb_dim, meta_dim), torch.nn.ReLU(),
                                           torch.nn.Linear(meta_dim, meta_dim*(meta_dim + 1)))
        self.emb_dim = meta_dim

    def forward(self, emb_fea, mask=None):
        """
        input:
            emb_fea: (bs, dim)
            # mask: (bs, sl)
        return:
            (bs, emb_dim, emb_dim)
        """
        # mask = (seq_index == 0).float()
        # mask = 1.0 - mask # reverse
        # event_K = self.event_K(emb_fea) #(bs, sl, 1)
        # t = event_K - torch.unsqueeze(mask, 2) * 1e8 #(bs, sl, 1)
        # att = self.event_softmax(t)
        # his_fea = torch.sum(att * emb_fea, 1) #(bs, dim)
        output = self.decoder(emb_fea) #(bs, dim*dim)
        # return output.reshape(-1, self.emb_dim, self.emb_dim)
        mal_mat, bias_mat = torch.split(output, [self.emb_dim * self.emb_dim, self.emb_dim], dim=-1)
        return mal_mat.reshape(-1, self.emb_dim, self.emb_dim), bias_mat

class DIN_Attention(nn.Module):
    def __init__(self, 
                 embedding_dim=64,
                 attention_units=[32], 
                 hidden_activations="ReLU",
                 output_activation=None,
                 dropout_rate=0,
                 batch_norm=False,
                 use_softmax=False):
        super(DIN_Attention, self).__init__()
        self.embedding_dim = embedding_dim
        self.use_softmax = use_softmax
        if isinstance(hidden_activations, str) and hidden_activations.lower() == "dice":
            hidden_activations = [Dice(units) for units in attention_units]
        self.attention_layer = MLP_Layer(input_dim=4 * embedding_dim,
                                         output_dim=1,
                                         hidden_units=attention_units,
                                         hidden_activations=hidden_activations,
                                         output_activation=output_activation,
                                         dropout_rates=dropout_rate,
                                         batch_norm=batch_norm, 
                                         use_bias=True)

    def forward(self, target_item, history_sequence, mask=None):
        # target_item: b x emd
        # history_sequence: b x len x emb
        seq_len = history_sequence.size(1)
        target_item = target_item.unsqueeze(1).expand(-1, seq_len, -1)
        attention_input = torch.cat([target_item, history_sequence, target_item - history_sequence, 
                                     target_item * history_sequence], dim=-1) # b x len x 4*emb
        attention_weight = self.attention_layer(attention_input.view(-1, 4 * self.embedding_dim))
        attention_weight = attention_weight.view(-1, seq_len) # b x len
        if mask is not None:
            attention_weight = attention_weight * mask.float()
        if self.use_softmax:
            if mask is not None:
                attention_weight += -1.e9 * (1 - mask.float())
            attention_weight = attention_weight.softmax(dim=-1)
        output = (attention_weight.unsqueeze(-1) * history_sequence).sum(dim=1)
        return output

