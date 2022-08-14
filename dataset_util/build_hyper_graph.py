import numpy as np
import pandas as pd
import time
import os
import sys
from scipy.sparse import csr_matrix
import scipy.sparse as sp

# version_1: 根据behavior log构建hypergraph, 超边是商品属性, 节点是用户属性, 统计下稀疏度; 
# 每条数据采用历史七天内的行为序列构建超图;
# 1. 统计训练数据集和测试数据集中的用户和商品集合;
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
raw_sample = raw_sample.sort_values(by='time_stamp') #默认升序;

raw_sample_with_ad = pd.merge(raw_sample, right=ad_features_df, on="adgroup_id", how="left").reset_index()
raw_sample_with_features = pd.merge(raw_sample_with_ad, right=user_features_df, on="userid", how="left").reset_index()
time_delta = time.mktime(time.strptime('2017-05-06 08:55:10', "%Y-%m-%d %H:%M:%S")) - 1494032110
assert time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(1494032110 + time_delta)) == "2017-05-06 08:55:10" # make sure the timezone is correct
raw_sample_with_features.loc[:, 'time_stamp'] = raw_sample_with_features.loc[:, 'time_stamp'].map(lambda x: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(x) + time_delta)))
raw_sample_with_features = raw_sample_with_features.reindex(columns=["clk","time_stamp","userid","adgroup_id","pid","cate_id","campaign_id","customer","brand","price",
"cms_segid","cms_group_id","final_gender_code","age_level","pvalue_level","shopping_level","occupation","new_user_class_level"])
print("raw_sample_with_features shape", raw_sample_with_features.shape)

# Split by time as suggested by the original dataset
df = raw_sample_with_features
all_df = df[df['time_stamp'] < '2017-05-13']
train_df = df[(df['time_stamp'] < '2017-05-13') & (df['time_stamp'] >= '2017-05-10')] #筛选近三天的数据;
#train_df = df[(df['time_stamp'] < '2017-05-13')] #筛选近三天的数据;
test_df = df[df['time_stamp'] >= '2017-05-13']
behavior_df = df[df['time_stamp'] < '2017-05-10'] # 构建超图的df;

# user_voc, item_voc
user_voc = pd.unique(train_df['userid'])
item_voc = pd.unique(train_df['adgroup_id'])

# Generate click and no_click sequence
item_set = pd.unique(all_df['userid'])
user_set = pd.unique(all_df['adgroup_id'])

user_voc_set = set(user_voc)
item_voc_set = set(item_voc)

# 2. 根据用户集合从behavior log中筛选出相应的数据;
click_time = ""
item_attributes = []
user_attributes = []
itemAt2id, userAt2id = {}, {}
m_item, m_user = 0, 0
trainDataSize = 0

item_at_type = ["pid","cate_id","campaign_id","customer","brand","price"]
user_at_type = ["cms_segid","cms_group_id","final_gender_code","age_level","pvalue_level","shopping_level","occupation"]

for idx, row in behavior_df.iterrows():
    if idx % 10000 == 0:
        print("Processing {} lines".format(idx))
    if row["time_stamp"] < click_time:
        sys.exit("data not sorted by timestamp!")
    user_id = row["userid"]
    item_id = row["adgroup_id"]
    click = row['clk']

    if item_id in item_voc_set and user_id in user_voc_set:
        pass

    for item_type in item_at_type:
        item_at = row[item_type]
        if type(item_at) != int:
            continue
        else:
            at_item_key = item_type + "_{}".format(item_at)
            if at_item_key not in itemAt2id:
                itemAt2id[at_item_key] = len(itemAt2id) + 1
            # item_attributes.append(itemAt2id[at_key])
            # user_attributes.append(userAt2id[at_key])
    
            for user_type in user_at_type:
                user_at = row[user_type]
                if type(user_at) != int:
                    continue
                else:
                    at_key = user_type + "_{}".format(user_at)
                    if at_key not in userAt2id:
                        userAt2id[at_key] = len(userAt2id) + 1
                    user_attributes.append(userAt2id[at_key])
                    item_attributes.append(itemAt2id[at_item_key])
                    m_item = max(m_item, max(item_attributes))
                    m_user = max(m_user, max(user_attributes))
                    trainDataSize += 1

user_attributes = np.array(user_attributes)
item_attributes = np.array(item_attributes)

print("item number: {}, user number: {}".format(m_item, m_user))
print(f"Graph Sparsity: {(trainDataSize) / m_user / m_item}")

# (users,items), bipartite graph
UserItemNet = csr_matrix((np.ones(len(user_attributes)), (item_attributes, user_attributes)),
                                    shape=(m_user, m_item)) #第一参数: value, 第二个采纳数: index;

# 3. 根据日志构建hypergraph
# a = {'q':1, 'u':3}
# b = {'q':1, 'u':3, 'd':8}
# tmp = (a, b)