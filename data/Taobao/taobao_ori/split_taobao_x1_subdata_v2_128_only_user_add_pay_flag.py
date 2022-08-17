# -*- coding: utf-8 -*-
import pandas as pd
import os
import numpy as np
import time
import hashlib
from collections import defaultdict
import pdb
import random

datapath = "./"
#sequence_max_len = 128
sequence_max_len = 170
user_sequence_max_len = 1000
#sequence_max_len = 128
#user_sequence_max_len = 128

ad_features_df = pd.read_csv(os.path.join(datapath, "ad_feature.csv"), dtype=object)
print("ad_features_df shape", ad_features_df.shape)
user_features_df = pd.read_csv(os.path.join(datapath, "user_profile.csv"), dtype=object)
print("user_features_df shape", user_features_df.shape)

raw_sample = pd.read_csv(os.path.join(datapath, "raw_sample.csv"), dtype=object)
#raw_sample = pd.read_csv(os.path.join(datapath, "raw_sample_test.csv"), dtype=object)
raw_sample.rename(columns={'user':'userid'}, inplace=True)
print("raw_sample shape", raw_sample.shape)
raw_sample = raw_sample.sort_values(by='time_stamp')

raw_sample_with_ad = pd.merge(raw_sample, right=ad_features_df, on="adgroup_id", how="left").reset_index()
raw_sample_with_features = pd.merge(raw_sample_with_ad, right=user_features_df, on="userid", how="left").reset_index()
time_delta = time.mktime(time.strptime('2017-05-06 08:55:10', "%Y-%m-%d %H:%M:%S")) - 1494032110
assert time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(1494032110 + time_delta)) == "2017-05-06 08:55:10" # make sure the timezone is correct
raw_sample_with_features.loc[:, 'time_stamp'] = raw_sample_with_features.loc[:, 'time_stamp'].map(lambda x: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(x) + time_delta)))
raw_sample_with_features = raw_sample_with_features.reindex(columns=["clk","time_stamp","userid","adgroup_id","pid","cate_id","campaign_id","customer","brand","price","cms_segid","cms_group_id","final_gender_code","age_level","pvalue_level","shopping_level","occupation","new_user_class_level"])
print("raw_sample_with_features shape", raw_sample_with_features.shape)

# Split by time as suggested by the original dataset
df = raw_sample_with_features
all_df = df[df['time_stamp'] < '2017-05-13']
train_df = df[(df['time_stamp'] < '2017-05-13') & (df['time_stamp'] >= '2017-05-10')] #筛选近三天的数据;
#train_df = df[(df['time_stamp'] < '2017-05-13')] #筛选近三天的数据;
test_df = df[df['time_stamp'] >= '2017-05-13']
behavior_df = df[df['time_stamp'] < '2016-05-10'] # 构建超图的df;

#get paying set;
#user_id_df = behavior_df.groupby("clk")['userid'].nunique()
#df['his_clk'] = df.groupby("userid")['clk'].max()
user_clk_map_train_df = behavior_df.groupby("userid", as_index=False)['clk'].agg({"his_clk": "max"})
#user_id_df = df.groupby("clk")['userid'].nunique()
user_clk_map_test_df = all_df.groupby("userid", as_index=False)['clk'].agg({"his_clk": "max"})

#user_id_map_df = df[['userid', 'his_clk']]

train_df = pd.merge(train_df, user_clk_map_train_df, how='left', on=['userid'])
train_df['his_clk'].fillna(0, inplace=True)

test_df = pd.merge(test_df, user_clk_map_test_df, how='left', on=['userid'])
test_df['his_clk'].fillna(0, inplace=True)

#pdb.set_trace()
#user_his_voc = set(user_clk_map.keys())


# user_voc, item_voc
user_voc = pd.unique(train_df['userid'])
item_voc = pd.unique(train_df['adgroup_id'])

# Generate click and no_click sequence
item_set = pd.unique(all_df['userid'])
user_set = pd.unique(all_df['adgroup_id'])

print("Start handling data.")

#print("before: all_df length: {}".format(all_df.shape[0]))
#all_df = pd.merge(all_df, train_df, on=['userid'], how='inner')
#all_df = pd.merge(all_df, train_df, on=['adgroup_id'], how='inner')
#print("after: all_df length: {}".format(all_df.shape[0]))
user_voc_set = set(user_voc)
item_voc_set = set(item_voc)

click_sequence_queue = defaultdict(list)
click_user_sequence_queue = defaultdict(list)
click_time = ""
click_history_col = []
click_history_users_col = []
his_clk_col = []

for idx, row in all_df.iterrows():
    if idx % 10000 == 0:
        print("Processing {} lines".format(idx))
    if row["time_stamp"] < click_time:
        sys.exit("data not sorted by timestamp!")
    user_id = row["userid"]
    item_id = row["adgroup_id"]
    click = row['clk']

    #if item_id not in item_set:
    #    item_set.append(item_id)
    #if user_id not in user_set:
    #    user_set.append(user_id)

    if item_id in item_voc_set and user_id in user_voc_set:
        pass
    else:
        continue

    #if user_id in user_his_voc:
    #    his_clk_col.append(int(user_clk_map[user_id]))
    #else:
    #    his_clk_col.append(0)

    # item-based sequences
    click_history = click_sequence_queue[user_id]
    # if len(click_history) > sequence_max_len:
        # click_history = click_history[-sequence_max_len:]
    click_sequence_queue[user_id] = click_history
    #click_history_col.append("^".join(click_history)) #每个<user, item> pair, add click history_col
    if click == "1": # click sequences;
        click_sequence_queue[user_id].append(item_id)

    # user-based sequence
    click_user_history = click_user_sequence_queue[item_id]
    if len(click_user_history) > user_sequence_max_len:
        click_user_history = click_user_history[-user_sequence_max_len:]
    click_user_sequence_queue[item_id] = click_user_history
    click_history_users_col.append("^".join(click_user_history)) #每个<user, item> pair, add click history_col
    if click == "1":
        click_user_sequence_queue[item_id].append(user_id)


print("common user: {}, train: {}, seq: {}".format(len(list(set(user_set)-set(user_voc))), len(user_voc), len(user_set)))
print("common item: {}, train: {}, seq: {}".format(len(list(set(item_set)-set(item_voc))), len(item_voc), len(item_set)))

# train_df.loc[:, "click_sequence"] = click_history_col
#train_df.loc[:, "click_sequence"] = train_df["userid"].map(lambda x: "^".join(click_sequence_queue[x][-sequence_max_len:]))
#test_df.loc[:, "click_sequence"] = test_df["userid"].map(lambda x: "^".join(click_sequence_queue[x][-sequence_max_len:]))
#pdb.set_trace()
# 随机mask行为序列中的单个元素, 用于模拟测试阶段;
#x = train_df.apply(lambda x: x["adgroup_id"] + x["userid"], axis=1)
def mask_ele(x):
    # random mask history sequence
    his_seq = list(set(click_sequence_queue[x["userid"]][-sequence_max_len:]) - set([x["adgroup_id"]]))
    #if len(click_sequence_queue[x["userid"]]) > 1:
    if len(his_seq) > 1:
        # randmo mask
        random.shuffle(his_seq)
        his_seq.pop()
        return "^".join(his_seq)
    else:
        #return click_sequence_queue[x["userid"]][-sequence_max_len:]
        return "^".join(his_seq)


# mask sequence
#train_df.loc[:, "click_sequence"] = train_df.apply(lambda x: "^".join(list(set(click_sequence_queue[x["userid"]][-sequence_max_len:]) - set([x["adgroup_id"]]))), axis=1)
# train_df.loc[:, "click_sequence"] = train_df.apply(mask_ele, axis=1)
#test_df.loc[:, "click_sequence"] = test_df[["userid", "adgroup_id"]].apply(lambda x: "^".join(list(set(click_sequence_queue[x["userid"]][-sequence_max_len:]) - set([x["adgroup_id"]]))), axis=1)
# test_df.loc[:, "click_sequence"] = test_df.apply(mask_ele, axis=1)

#pdb.set_trace()
# user-based data
train_df.loc[:, "click_user_sequence"] = click_history_users_col[-len(train_df):]
test_df.loc[:, "click_user_sequence"] = test_df["adgroup_id"].map(lambda x: "^".join(click_user_sequence_queue[x][-user_sequence_max_len:]))
# user-based
# train_df.loc[:, "click_user_sequence"] = click_history_users_col
#train_df.loc[:, "click_user_sequence"] = train_df["adgroup_id"].map(lambda x: "^".join(click_user_sequence_queue[x][-sequence_max_len:]))
#test_df.loc[:, "click_user_sequence"] = test_df["adgroup_id"].map(lambda x: "^".join(click_user_sequence_queue[x][-sequence_max_len:]))
#train_df.loc[:, "click_user_sequence"] = train_df["adgroup_id"].map(lambda x: "^".join(click_user_sequence_queue[x][-user_sequence_max_len:]))
#test_df.loc[:, "click_user_sequence"] = test_df["adgroup_id"].map(lambda x: "^".join(click_user_sequence_queue[x][-user_sequence_max_len:]))

#train_df.loc[:, "click_user_sequence"] = train_df[["adgroup_id", "userid"]].apply(lambda x: "^".join(list(set(click_user_sequence_queue[x["adgroup_id"]][-user_sequence_max_len:]) - set([x["userid"]]))), axis=1)
#test_df.loc[:, "click_user_sequence"] = test_df[["adgroup_id", "userid"]].apply(lambda x: "^".join(list(set(click_user_sequence_queue[x["adgroup_id"]][-user_sequence_max_len:]) - set([x["userid"]]))), axis=1)

#train_df.to_csv("train_170_1000.csv", index=False)
#test_df.to_csv("test_170_1000.csv", index=False)
train_df.to_csv("train_norm_only_user_true_add_his_clk.csv", index=False)
test_df.to_csv("test_norm_only_user_true_add_his_clk.csv", index=False)
print('Train samples:', len(train_df), 'postives:', sum(train_df['clk'] > '0'))
print('Test samples:', len(test_df), 'postives:', sum(test_df['clk'] > '0'))

# Check md5sum for correctness
# assert("e4487021d20750121725e880556bfdc1" == hashlib.md5(open('train.csv', 'r').read().encode('utf-8')).hexdigest())
# assert("1de0b75cbb473b0c3ea2dd146dc4af28" == hashlib.md5(open('test.csv', 'r').read().encode('utf-8')).hexdigest())

print("Reproducing data succeeded!")
