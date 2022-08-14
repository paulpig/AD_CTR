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


import torch
from torch import nn
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import MLP_Layer, EmbeddingLayer, SqueezeExcitationLayer, BilinearInteractionLayer, LR_Layer
import torch.nn.functional as F
import pdb

class FiBiNET_CL(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="FiBiNET_CL", 
                 gpu=-1, 
                 task="binary_classification",
                 learning_rate=1e-3, 
                 embedding_initializer="torch.nn.init.normal_(std=1e-4)",
                 embedding_dim=10, 
                 hidden_units=[], 
                 hidden_activations="ReLU",
                 reduction_ratio=3,
                 bilinear_type="field_interaction",
                 net_dropout=0, 
                 batch_norm=False, 
                 embedding_regularizer=None, 
                 net_regularizer=None, 
                 **kwargs):
        super(FiBiNET_CL, self).__init__(feature_map, 
                                      model_id=model_id, 
                                      gpu=gpu, 
                                      embedding_regularizer=embedding_regularizer, 
                                      net_regularizer=net_regularizer,
                                      **kwargs)
        self.embedding_layer = EmbeddingLayer(feature_map, embedding_dim) # handle various variables.
        num_fields = feature_map.num_fields
        self.senet_layer = SqueezeExcitationLayer(num_fields, reduction_ratio)
        self.bilinear_interaction = BilinearInteractionLayer(num_fields, embedding_dim, bilinear_type)
        self.lr_layer = LR_Layer(feature_map, output_activation=None, use_bias=False)
        input_dim = num_fields * (num_fields - 1) * embedding_dim
        self.dnn = MLP_Layer(input_dim=input_dim,
                             output_dim=1, 
                             hidden_units=hidden_units, 
                             hidden_activations=hidden_activations,
                             output_activation=None,
                             dropout_rates=net_dropout, 
                             batch_norm=batch_norm, 
                             use_bias=True)

        self.output_activation = self.get_output_activation(task)
        self.compile(kwargs["optimizer"], loss=kwargs["loss"], lr=learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        """
        Inputs: [X, y]
        """
        X, y = self.inputs_to_device(inputs)
        feature_emb = self.embedding_layer(X) # list of b x embedding_dim
        senet_emb = self.senet_layer(feature_emb)
        bilinear_p = self.bilinear_interaction(feature_emb)
        bilinear_q = self.bilinear_interaction(senet_emb)
        comb_out = torch.flatten(torch.cat([bilinear_p, bilinear_q], dim=1), start_dim=1)
        # add cl loss: <bilinear_p, bilinear_q> is the positive pair
        # bilinear_p: (bs, num_fields * (num_fields - 1) * embedding_dim)
        bilinear_p_cl, bilinear_q_cl = bilinear_p.reshape(bilinear_p.size(0), -1), bilinear_q.reshape(bilinear_q.size(0), -1)
        cl_loss = self.loss_contrastive(bilinear_p_cl, bilinear_q_cl)

        dnn_out = self.dnn(comb_out)
        y_pred = self.lr_layer(X) + dnn_out

        if self.output_activation is not None:
            y_pred = self.output_activation(y_pred)
        return_dict = {"y_true": y, "y_pred": y_pred, "aux_loss": cl_loss}
        
        return return_dict

    
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
        all_scores_mask = all_scores + (torch.eye(tensor_anchor_v1.size(0)*2) * (-1e8)).cuda()

        all_score = torch.sum(torch.exp(all_scores_mask/temp_value), dim=1) #(2*bs)
        pos_score = torch.exp(torch.cat([pos_score, pos_score], dim=0)) #(2*bs)

        cl_loss = (-torch.log(pos_score / all_score)).mean()

        return cl_loss

        # all_score = all_score.view(-1, 1+self.num_neg)
        # pos_score = all_score[:, 0]
        # all_score = torch.sum(all_score, dim=1)
        # self.mat = (1-pos_score/all_score).mean()
        # contrastive_loss = (-torch.log(pos_score / all_score)).mean()
        # return contrastive_loss

    # def loss_contrastive(self, tensor_anchor_v1, tensor_anchor_v2, temp_value):
    #     """
    #     tensor_anchor_v1: (bs, dim)
    #     tensor_anchor_v2: (bs, dim)
    #     """   
    #     all_score = torch.exp(torch.sum(tensor_anchor*tensor_all, dim=1)/temp_value).view(-1, 1+self.num_neg)
    #     all_score = all_score.view(-1, 1+self.num_neg)
    #     pos_score = all_score[:, 0]
    #     all_score = torch.sum(all_score, dim=1)
    #     self.mat = (1-pos_score/all_score).mean()
    #     contrastive_loss = (-torch.log(pos_score / all_score)).mean()
    #     return contrastive_loss

    # def compute_kl_loss(p, q, pad_mask=None):
    
    #     p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    #     q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
        
    #     # pad_mask is for seq-level tasks
    #     if pad_mask is not None:
    #         p_loss.masked_fill_(pad_mask, 0.)
    #         q_loss.masked_fill_(pad_mask, 0.)

    #     # You can choose whether to use function "sum" and "mean" depending on your task
    #     p_loss = p_loss.sum()
    #     q_loss = q_loss.sum()

    #     loss = (p_loss + q_loss) / 2
    #     return loss