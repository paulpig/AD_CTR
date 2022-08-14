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
import torch.nn.functional as F

class DIN_Meta(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="DIN",
                 gpu=-1, 
                 task="binary_classification",
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
                 **kwargs):
        super(DIN_Meta, self).__init__(feature_map, 
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
        self.embedding_layer = EmbeddingDictLayer(feature_map, embedding_dim)
        # pdb.set_trace()
        self.attention_layers = nn.ModuleList(
            [DIN_Attention(embedding_dim * len(target_field) if type(target_field) == tuple \
                                                             else embedding_dim,
                           attention_units=attention_hidden_units,
                           hidden_activations=attention_hidden_activations,
                           output_activation=attention_output_activation,
                           dropout_rate=attention_dropout,
                           batch_norm=batch_norm,
                           use_softmax=din_use_softmax)
             for target_field in self.din_target_field])
        # self.dnn = MLP_Layer(input_dim=feature_map.num_fields * embedding_dim,
        # self.dnn = MLP_Layer(input_dim=(feature_map.num_fields + 1)* embedding_dim,
        self.dnn_ori = MLP_Layer(input_dim=(feature_map.num_fields)* embedding_dim,
                             output_dim=1,
                             hidden_units=dnn_hidden_units,
                             hidden_activations=dnn_activations,
                             output_activation=self.get_output_activation(task), 
                             dropout_rates=net_dropout,
                             batch_norm=batch_norm, 
                             use_bias=True)
        
        # self.dnn = MLP_Layer(input_dim=(feature_map.num_fields -1) * embedding_dim,
        self.dnn = MLP_Layer(input_dim=(feature_map.num_fields - 0) * embedding_dim,
                             output_dim=embedding_dim,
                             hidden_units=dnn_hidden_units,
                             hidden_activations=dnn_activations,
                            #  output_activation='RELU', 
                             output_activation=None, 
                             dropout_rates=net_dropout,
                             batch_norm=batch_norm, 
                             use_bias=True)
        
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


        self.meta_net = MetaNet(embedding_dim, embedding_dim)
        # self.meta_net_v2 = MetaNet(embedding_dim, embedding_dim)
        self.meta_net_flat = MetaNetFlat(embedding_dim, embedding_dim)
        self.pretrain_emb_mlp = MetaNetFlat(embedding_dim, embedding_dim)

        self.meta_net_flat_v1 = MetaNetFlatV1(embedding_dim, embedding_dim)
        # self.output_linear = torch.nn.Linear(embedding_dim, 1)
        self.output_linear = torch.nn.Linear(embedding_dim, 1)

        self.layer_norm = torch.nn.LayerNorm(embedding_dim)


        self.gate_layer = torch.nn.Linear(embedding_dim*2, 1, True)

        self.compile(kwargs["optimizer"], loss=kwargs["loss"], lr=learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        X, y = self.inputs_to_device(inputs)
        feature_emb_dict = self.embedding_layer(X)
        # pdb.set_trace()
        # assert len(self.din_target_field) == 2

        # # 对userid添加l2 norm, 保持维度的相同;
        # for key, value in feature_emb_dict.items():
        #     # feature_emb_dict[key] = torch.nn.functional.normalize(feature_emb_dict[key], p=2, dim=1)
        #     feature_emb_dict[key] = self.layer_norm(value)
        # feature_emb_dict['userid'] = feature_emb_dict['userid']/10.0
        # feature_emb_dict['click_user_sequence'] = feature_emb_dict['click_user_sequence']/10.0
        # pdb.set_trace()

        if self.use_meta and self.use_pretrain:
            # add pretrain user emb
            feature_spec = self.embedding_layer._feature_map.feature_specs['userid']
            inp = X[:, feature_spec["index"]].long()
            # user_embedding_hypergraph = self.embedding_layer.embedding_layer['userid'](inp)
            user_embedding_hypergraph = self.embedding_layer.embedding_layer['userid_cp'](inp) #消融实验;
            # self.embedding_layer.embedding_layer['userid'](inp)

        mask_label = self.params["add_sequence"]

        assert ((mask_label == 1) and (self.use_attention or self.use_meta)) or (mask_label == 0)
        # pdb.set_trace()
        # if mask_label == 1:
        #     for idx, (target_field, sequence_field) in enumerate(zip(self.din_target_field, 
        #                                                             self.din_sequence_field)):
        #         # target_field, sequence_field = tuple(target_field), tuple(sequence_field)
        #         target_emb = self.concat_embedding(target_field, feature_emb_dict) #(bs, dim)
        #         sequence_emb = self.concat_embedding(sequence_field, feature_emb_dict) #(bs, sl, dim)
        #         seq_field = np.array([sequence_field]).flatten()[0] # pick a sequence field
        #         padding_idx = self.feature_map.feature_specs[seq_field]['padding_idx']
        #         # mask: (bs, sl), sl=128;
        #         mask = (X[:, self.feature_map.feature_specs[seq_field]["index"]].long() != padding_idx).float() # (bs, sl), 'index' means the positions of columns;
        #         # pdb.set_trace()

        #         if self.use_attention:
        #             # 1. attention layer
        #             pooling_emb = self.attention_layers[idx](target_emb, sequence_emb, mask) #(bs, dim)
        #             # pdb.set_trace()
        #         # 2. max pooling
        #         # pooling_emb, _ = torch.max(mask.unsqueeze(dim=-1) * sequence_emb, dim=1)
        #         # 3. mean pooling
        #         # pooling_emb = torch.max(mask.unsqueeze(dim=-1) * sequence_emb, dim=1)
                
        #         if self.use_meta:
        #             # 4. meta learning
        #             if idx == 0:
        #                 #1. 基于序列的meta unit
        #                 meta_mapping =  self.meta_net(sequence_emb, mask) #(bs, dim, dim)
        #                 pooling_emb = torch.bmm(target_emb.unsqueeze(1), meta_mapping).squeeze(1) #(bs, dim)

        #                 # 2. 基于预训练表征的meta unit
        #                 if self.use_pretrain and not self.concat_other_feat:
        #                     # add pretrain user embedding meta
        #                     # v1
        #                     # pooling_emb_pretrain = self.meta_net_flat(user_embedding_hypergraph) #(bs, dim)
        #                     meta_mapping_pretrain = self.meta_net_flat_v1(user_embedding_hypergraph) #(bs, dim, dim)
        #                     pooling_emb_pretrain = torch.bmm(target_emb.unsqueeze(1), meta_mapping_pretrain).squeeze(1) #(bs, dim)

        #                     # v1: add gate 
        #                     # gate = nn.functional.sigmoid(self.gate_layer(torch.cat([pooling_emb, pooling_emb_pretrain], dim=-1)))
        #                     # pooling_emb = gate * pooling_emb + (1. - gate) * pooling_emb_pretrain
        #                     # v2: add op
        #                     # pooling_emb = pooling_emb + pooling_emb_pretrain

        #                     # origin feature
        #                     # meta_mapping_bias = user_embedding_hypergraph

        #                     # v2 concat
        #                     # pooling_emb = torch.cat([pooling_emb, meta_mapping_bias], dim=-1)
        #                     # pooling_emb = meta_mapping_bias
                            
        #                     # v2: only pretrain
        #                     pooling_emb = pooling_emb_pretrain

        #                 if self.use_pretrain and self.concat_other_feat:
        #                     pooling_emb = user_embedding_hypergraph
                
        #         index = 0
        #         for field_emb in pooling_emb.split(self.embedding_dim, dim=-1):
        #             # pdb.set_trace()
        #             feature_emb_dict["other_{}".format(index)] = field_emb
        #             index += 1
        
        # # 对userid添加l2 norm, 保持维度的相同;
        # for key, value in feature_emb_dict.items():
        #     # feature_emb_dict[key] = torch.nn.functional.normalize(feature_emb_dict[key], p=2, dim=1)
        #     feature_emb_dict[key] = self.layer_norm(value)
        
        
        # pdb.set_trace()
        # remove 'click_user_sequence', 已转化为其他特征;
        del feature_emb_dict['click_user_sequence']
        # del feature_emb_dict['userid']
        
        # add hypergraph emb
        # feature_emb_dict["hyper_emb"] = user_embedding_hypergraph
        convert_emb_hyper = self.pretrain_emb_mlp(user_embedding_hypergraph)
        feature_emb_dict["hyper_emb"] = convert_emb_hyper

        feature_emb = self.embedding_layer.dict2tensor(feature_emb_dict)
        # list(feature_emb_dict.keys())[14]
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
        # y_pred = self.dnn_ori(feature_emb.flatten(start_dim=1))
        hidden_emb = self.dnn(feature_emb.flatten(start_dim=1))

        # POSO method
        # weight = nn.functional.sigmoid(self.meta_net_flat(user_embedding_hypergraph)) #(bs, dim)
        # hidden_emb = hidden_emb * weight * 2.0
        # y_pred = nn.functional.sigmoid(self.output_linear(hidden_emb))
        
        # meta learning
        # mapping
        # pooling_emb_mapping = self.meta_net_flat_v1(user_embedding_hypergraph) #(bs, dim, dim)
        pooling_emb_mapping, bias_emb_mapping = self.meta_net_flat_v1(convert_emb_hyper) #(bs, dim, dim)
        hidden_emb_v1 = torch.bmm(hidden_emb.unsqueeze(1), pooling_emb_mapping).squeeze(1) + bias_emb_mapping #(bs, dim), 加上bias效果非常好...
        # hidden_emb_v1 = torch.nn.ReLU()(torch.bmm(hidden_emb.unsqueeze(1), pooling_emb_mapping).squeeze(1)) #(bs, dim), 添加RELU效果下降明显;
        # y_pred = nn.functional.sigmoid(self.output_linear(torch.nn.ReLU()(hidden_emb + hidden_emb_v1)))# 添加bias效果很好;
        # y_pred = nn.functional.sigmoid(self.output_linear(torch.nn.ReLU()(hidden_emb)))# 添加bias效果很好;
        # y_pred = nn.functional.sigmoid(self.output_linear(hidden_emb))# baseline, 只考虑hidden_emb: 0.632339
        y_pred = nn.functional.sigmoid(self.output_linear(hidden_emb + hidden_emb_v1)) # 添加bias效果很好;
        # y_pred = nn.functional.sigmoid(self.output_linear(torch.cat([hidden_emb, hidden_emb_v1], dim=-1))) # concat效果不佳;
        # y_pred = nn.functional.sigmoid(self.output_linear(hidden_emb_v1)) # 添加bias效果很好;
        # pdb.set_trace()

        return_dict = {"y_true": y, "y_pred": y_pred}
        # return_dict = {"y_true": y, "y_pred": y_pred, "aux_loss": self.cl_loss}
        return return_dict
    # X[11, self.feature_map.feature_specs[seq_field]["index"]]

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
                                           torch.nn.Linear(meta_dim, emb_dim))
        self.emb_dim = emb_dim

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
                                           torch.nn.Linear(meta_dim, emb_dim*(emb_dim + 1)))
        self.emb_dim = emb_dim

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

