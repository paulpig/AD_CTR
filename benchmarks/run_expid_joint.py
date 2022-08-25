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


from calendar import c
import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import sys
sys.path.append('../')
from fuxictr import datasets
from datetime import datetime
from fuxictr.utils import load_config, set_logger, print_to_json, print_to_list
from fuxictr.features import FeatureMap, FeatureEncoder
from fuxictr.pytorch import models
from fuxictr.pytorch.torch_utils import seed_everything
import gc
import argparse
import logging
import torch
import os
from pathlib import Path
import pdb
import pickle


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str, default='pytorch', help='The model version.')
    parser.add_argument('--config', type=str, default='../config/', help='The config directory.')
    parser.add_argument('--expid', type=str, default='FM_test', help='The experiment id to run.')
    parser.add_argument('--gpu', type=int, default=-1, help='The gpu index, -1 for cpu')
    
    args = vars(parser.parse_args())
    experiment_id = args['expid']
    params = load_config(args['config'], experiment_id)
    params['gpu'] = args['gpu']
    params['version'] = args['version']
    set_logger(params)
    logging.info(print_to_json(params))
    seed_everything(seed=params['seed'])

    # preporcess the dataset
    dataset = params['dataset_id'].split('_')[0].lower()
    data_dir = os.path.join(params['data_root'], params['dataset_id'])
    # pdb.set_trace()
    if params.get("data_format") == 'h5': # load data from h5
        feature_map = FeatureMap(params['dataset_id'], data_dir, params['version'])
        json_file = os.path.join(os.path.join(params['data_root'], params['dataset_id']), "feature_map.json")
        if os.path.exists(json_file):
            feature_map.load(json_file)
        else:
            raise RuntimeError('feature_map not exist!')
    else: # load data from csv
        try:
            feature_encoder = getattr(datasets, dataset).FeatureEncoder(**params) # feature encoder;
        except:
            feature_encoder = FeatureEncoder(**params)
        
        #===========new===============
        # if params.get("data_format") == 'h5':
        #     if os.path.exists(feature_encoder.json_file):
        #         feature_encoder.feature_map.load(feature_encoder.json_file)
        #     else:
        #         raise RuntimeError('feature_map not exist!')
        # elif params.get('pickle_feature_encoder') and os.path.exists(feature_encoder.pickle_file):
        #     feature_encoder = feature_encoder.load_pickle(feature_encoder.pickle_file)
        # else:
        #     feature_encoder.fit(**params)
        #===========old===============
        if os.path.exists(feature_encoder.json_file):
            feature_encoder.feature_map.load(feature_encoder.json_file)
        else: # Build feature_map and transform h5 data
            datasets.build_dataset(feature_encoder, **params)
            # train_ddf = feature_encoder.preprocess(train_ddf)
            # train_array = feature_encoder.fit_transform(train_ddf, **kwargs)

        params["train_data"] = os.path.join(data_dir, 'train*.h5')
        params["valid_data"] = os.path.join(data_dir, 'valid*.h5')
        params["test_data"] = os.path.join(data_dir, 'test*.h5')
        feature_map = feature_encoder.feature_map
        # behavior_df[(behavior_df['userid'] == '49912')  & (behavior_df['clk']=='1')]

    # pretrain model;
    # model_id_pretrain = "HyperGraphCustomBipartiteDisenGATVAEV3"
    # model_id_pretrain = "HyperGraphCustomBipartiteDisenGATVAEV3CTRObj"
    # model_id_pretrain = "HyperGraphCustomBipartiteDisenGATVAEV3CTRObjSameIdx"
    # model_id_pretrain = "HyperGraphCustomBipartiteDisenGATVAEV3CTRObjSameIdxHyperGraph"
    model_id_pretrain = "LightGCNCLHuaWei"
    
    encoder_model_pre = params['encoder_model']
    # model_class = getattr(models, 'HyperGraph')
    model_class = getattr(models, model_id_pretrain)
    model_pre = model_class(gpu=args['gpu'], graph_layer=params['graph_layer'],
                                    learning_rate=params['learning_rate'], graph_embedding_dim=params['graph_embedding_dim'],
                                    bpr_batch_size=params['bpr_batch_size'], weight_decay=params['weight_decay'], epoch_pre=params['epoch_pre'],
                                    iterations=params['iterations'], add_norm=params['add_norm'], channels=params['channels'],cl_w=params['cl_w'],
                                    add_vae=params['add_vae'], learning_rate_vae=params["learning_rate_vae"])
    
    model_pre.count_parameters()
    # fit the model
    model_pre.train_model(add_cl=False, model_name=encoder_model_pre)
    pdb.set_trace()
    # save user embedding
    print("start saving user embeddings.")
    # get train and validation data
    train_gen, valid_gen = datasets.h5_generator(feature_map, stage='train', **params)
    # initialize model
    model_class = getattr(models, params['model'])
    model = model_class(feature_map, pre_model=model_pre, **params)

    # model.set_reindex_user_ids(reindex_user_ids)
    # model.set_reindex_customer_ids(reindex_customer_ids)
    # pdb.set_trace()
    # 初始化model embeddings
    # model_pre = model.reload_pretrain_model()
    
    # model_pre.forward_embeddings(save_model_name=encoder_model_pre)
    # # model.embedding_layer.embedding_layer['userid_cp'] = model_pre.user_embeddings[reindex_user_ids]
    # model.embedding_layer.other_emb_layer = model_pre.user_embeddings[reindex_user_ids]
    # pdb.set_trace() # torch.nn.Parameter( 
    # model.embedding_layer.embedding_layer["userid"].weight = torch.nn.Parameter(model_pre.user_embeddings[reindex_user_ids])

    # pdb.set_trace()
    # print number of parameters used in model
    model.count_parameters()
    # fit the model
    model.fit_generator(train_gen, validation_data=valid_gen, pre_model=model_pre, encoder_model_pre=encoder_model_pre, **params)

    # load the best model checkpoint
    logging.info("Load best model: {}".format(model.checkpoint))
    model.load_weights(model.checkpoint)

    # get evaluation results on validation
    logging.info('****** Validation evaluation ******')

    valid_result = model.evaluate_generator(valid_gen)
    del train_gen, valid_gen
    gc.collect()

    # get evaluation results on test
    logging.info('******** Test evaluation ********')


    test_gen = datasets.h5_generator(feature_map, stage='test', **params)
    if test_gen:
        test_result = model.evaluate_generator(test_gen)
    else:
        test_gen = {}
    
    # save the results to csv
    result_file = Path(args['config']).name.replace(".yaml", "") + '.csv'
    with open(result_file, 'a+') as fw:
        fw.write(' {},[command] python {},[exp_id] {},[dataset_id] {},[train] {},[val] {},[test] {}\n' \
            .format(datetime.now().strftime('%Y%m%d-%H%M%S'), 
                    ' '.join(sys.argv), experiment_id, params['dataset_id'],
                    "N.A.", print_to_list(valid_result), print_to_list(test_result)))