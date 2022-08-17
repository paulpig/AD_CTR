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
import os
import scipy.sparse as sp
from pathlib import Path
import pdb


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

    # pdb.set_trace()
    model_id = experiment_id
    encoder_model = params['encoder_model']
    # model_class = getattr(models, 'HyperGraph')
    model_class = getattr(models, model_id)
    model = model_class(gpu=args['gpu'], graph_layer=params['graph_layer'],
                                    learning_rate=params['learning_rate'], embedding_dim=params['embedding_dim'],
                                    bpr_batch_size=params['bpr_batch_size'], weight_decay=params['weight_decay'], epoch=params['epochs'],
                                    iterations=params['iterations'], add_norm=params['add_norm'], channels=params['channels'])
    # model = model_class(feature_map, **params)
    # print number of parameters used in model
    model.count_parameters()
    # fit the model
    # model.fit_generator(train_gen, validation_data=valid_gen, **params)
    # model.train_model(add_cl=False, model_name="disengcn")
    model.train_model(add_cl=False, model_name=encoder_model)
    # model.train_model(add_cl=False, model_name="lightgcn")
    # model.train_model(add_cl=True, model_name="disengcn")

    # save user embedding
    print("start saving user embeddings.")
    # model.save_embeddings(save_model_name="disengcn")
    model.save_embeddings(save_model_name=encoder_model)
    # model.save_embeddings(save_model_name = "lightgcn")
    # model.save_embeddings( save_model_name = "embedding")
    # load the best model checkpoint
    # logging.info("Load best model: {}".format(model.checkpoint))
    # model.load_weights(model.checkpoint)

    # get evaluation results on validation
    # logging.info('****** Validation evaluation ******')
    # valid_result = model.evaluate_generator(valid_gen)
    # del train_gen, valid_gen
    # gc.collect()

    # # get evaluation results on test
    # logging.info('******** Test evaluation ********')
    # test_gen = datasets.h5_generator(feature_map, stage='test', **params)
    # if test_gen:
    #     test_result = model.evaluate_generator(test_gen)
    # else:
    #     test_gen = {}
    
    # save the results to csv
    # result_file = Path(args['config']).name.replace(".yaml", "") + '.csv'
    # with open(result_file, 'a+') as fw:
    #     fw.write(' {},[command] python {},[exp_id] {},[dataset_id] {},[train] {},[val] {},[test] {}\n' \
    #         .format(datetime.now().strftime('%Y%m%d-%H%M%S'), 
    #                 ' '.join(sys.argv), experiment_id, params['dataset_id'],
    #                 "N.A.", print_to_list(valid_result), print_to_list(test_result)))