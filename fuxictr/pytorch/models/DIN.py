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


class DIN(BaseModel):
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
        super(DIN, self).__init__(feature_map, 
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
        self.embedding_layer = EmbeddingDictLayer(feature_map, embedding_dim, use_pretrain=False)
        # self.embedding_layer.embedding_layer['userid'].weight[:20]
        #  self.embedding_layer.embedding_layer['adgroup_id'].weight[:20]
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
        self.dnn = MLP_Layer(input_dim=(feature_map.num_fields - 0) * embedding_dim,
                             output_dim=1,
                             hidden_units=dnn_hidden_units,
                             hidden_activations=dnn_activations,
                             output_activation=self.get_output_activation(task), 
                             dropout_rates=net_dropout,
                             batch_norm=batch_norm, 
                             use_bias=True)
        self.compile(kwargs["optimizer"], loss=kwargs["loss"], lr=learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        X, y = self.inputs_to_device(inputs)
        feature_emb_dict = self.embedding_layer(X)
        # pdb.set_trace()

        mask_label = self.params["add_sequence"]
        # pdb.set_trace()
        if mask_label == 1:
            for idx, (target_field, sequence_field) in enumerate(zip(self.din_target_field, 
                                                                    self.din_sequence_field)):
                
                # target_field, sequence_field = tuple(target_field), tuple(sequence_field)
                target_emb = self.concat_embedding(target_field, feature_emb_dict) #(bs, dim)
                sequence_emb = self.concat_embedding(sequence_field, feature_emb_dict) #(bs, sl, dim)
                seq_field = np.array([sequence_field]).flatten()[0] # pick a sequence field
                padding_idx = self.feature_map.feature_specs[seq_field]['padding_idx']
                # mask: (bs, sl), sl=128;
                mask = (X[:, self.feature_map.feature_specs[seq_field]["index"]].long() != padding_idx).float() # 128, 'index' means the positions of columns;
                # pdb.set_trace()
                # 1. attention layer
                pooling_emb = self.attention_layers[idx](target_emb, sequence_emb, mask) #(bs, dim)
                # 2. max pooling
                # pooling_emb, _ = torch.max(mask.unsqueeze(dim=-1) * sequence_emb, dim=1)
                # 3. mean pooling
                # pooling_emb = torch.max(mask.unsqueeze(dim=-1) * sequence_emb, dim=1)

                # pdb.set_trace()
                for field, field_emb in zip(np.hstack([sequence_field]),
                                            pooling_emb.split(self.embedding_dim, dim=-1)):
                    # pdb.set_trace()
                    feature_emb_dict[field] = field_emb
                    # feature_emb_dict[field] = sequence_emb[:, 0, :]
                    # remove for test;
                    # pass
        # pdb.set_trace()
        if 'his_clk' in feature_emb_dict:
            del feature_emb_dict['his_clk']
        
        feature_emb = self.embedding_layer.dict2tensor(feature_emb_dict)
        y_pred = self.dnn(feature_emb.flatten(start_dim=1))
        return_dict = {"y_true": y, "y_pred": y_pred}
        return return_dict
    # X[11, self.feature_map.feature_specs[seq_field]["index"]]

    def concat_embedding(self, field, feature_emb_dict):
        if type(field) == tuple:
            emb_list = [feature_emb_dict[f] for f in field]
            return torch.cat(emb_list, dim=-1)
        else:
            return feature_emb_dict[field]


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

