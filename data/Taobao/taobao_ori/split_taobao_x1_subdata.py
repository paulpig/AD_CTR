# -*- coding: utf-8 -*-
import pandas as pd
import os
import numpy as np
import time
import hashlib
from collections import defaultdict


datapath = "./"
sequence_max_len = 128

ad_features_df = pd.read_csv(os.path.join(datapath, "ad_feature.csv"), dtype=object)
print("ad_features_df shape", ad_features_df.shape)
user_features_df = pd.read_csv(os.path.join(datapath, "user_profile.csv"), dtype=object)
print("user_features_df shape", user_features_df.shape)

raw_sample = pd.read_csv(os.path.join(datapath, "raw_sample.csv"), dtype=object)
raw_sample.rename(columns={'user':'userid'}, inplace=True)

# remove timestamp is NAN
raw_sample.dropna(subset = ["time_stamp"], inplace=True)

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
train_df = df[df['time_stamp'] < '2017-05-13']
# train_df = df[(df['time_stamp'] < '2017-05-13') & df['time_stamp'] >= '2017-05-10'] #筛选近三天的数据;
test_df = df[df['time_stamp'] >= '2017-05-13']

# Generate click and no_click sequence
min_seq_len, max_seq_len = 0, 0
min_user_seq_len, max_user_seq_len = 0, 0
click_sequence_queue = defaultdict(list)
click_user_sequence_queue = defaultdict(list)
click_time = ""
click_history_col = []
click_history_users_col = []
for idx, row in train_df.iterrows():
    if idx % 10000 == 0:
        print("Processing {} lines".format(idx))
    if row["time_stamp"] < click_time:
        sys.exit("data not sorted by timestamp!")
    user_id = row["userid"]
    item_id = row["adgroup_id"]
    click = row['clk']

    # item-based sequences
    click_history = click_sequence_queue[user_id]
    # if len(click_history) > sequence_max_len:
        # click_history = click_history[-sequence_max_len:]
    click_sequence_queue[user_id] = click_history
    click_history_col.append("^".join(click_history)) #每个<user, item> pair, add click history_col
    if click == "1": # click sequences;
        click_sequence_queue[user_id].append(item_id)

    if min_seq_len > len(click_history):
        min_seq_len = len(click_history)
    if max_seq_len < len(click_history):
        max_seq_len = len(click_history)

    # user-based sequence
    click_user_history = click_user_sequence_queue[item_id]
    # if len(click_user_history) > sequence_max_len:
        # click_user_history = click_user_history[-sequence_max_len:]
    click_user_sequence_queue[item_id] = click_user_history
    click_history_users_col.append("^".join(click_user_history)) #每个<user, item> pair, add click history_col
    if click == "1":
        click_user_sequence_queue[item_id].append(user_id)

    if min_user_seq_len > len(click_user_history):
        min_user_seq_len = len(click_user_history)
    if max_user_seq_len < len(click_user_history):
        max_user_seq_len = len(click_user_history)
print("min and max item-based sequence length: {}, {}".format(min_seq_len, max_seq_len))
print("min and max user-based sequence length: {}, {}".format(min_user_seq_len, max_user_seq_len))

train_df.loc[:, "click_sequence"] = click_history_col
#test_df.loc[:, "click_sequence"] = test_df["userid"].map(lambda x: "^".join(click_sequence_queue[x][-sequence_max_len:]))
test_df.loc[:, "click_sequence"] = test_df["userid"].map(lambda x: "^".join(click_sequence_queue[x]))
# user-based
train_df.loc[:, "click_user_sequence"] = click_history_users_col
#test_df.loc[:, "click_user_sequence"] = test_df["adgroup_id"].map(lambda x: "^".join(click_user_sequence_queue[x][-sequence_max_len:]))
test_df.loc[:, "click_user_sequence"] = test_df["adgroup_id"].map(lambda x: "^".join(click_user_sequence_queue[x]))


train_df.to_csv("train_ext.csv", index=False)
test_df.to_csv("test_ext.csv", index=False)
print('Train samples:', len(train_df), 'postives:', sum(train_df['clk'] > '0'))
print('Test samples:', len(test_df), 'postives:', sum(test_df['clk'] > '0'))

# Check md5sum for correctness
# assert("e4487021d20750121725e880556bfdc1" == hashlib.md5(open('train.csv', 'r').read().encode('utf-8')).hexdigest())
# assert("1de0b75cbb473b0c3ea2dd146dc4af28" == hashlib.md5(open('test.csv', 'r').read().encode('utf-8')).hexdigest())

print("Reproducing data succeeded!")
