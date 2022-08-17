import numpy as np
import pandas as pd
import time
import os
import sys
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import pdb
import pickle

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
#构建测试数据
#= df[df['time_stamp'] < '2017-05-10'] # 构建超图的df;
behavior_df_test = df[(df['time_stamp'] < '2017-05-13') & (df['time_stamp'] >= '2017-05-10')] #筛选近三天的数据;
# select label == 1

# user_voc, item_voc
user_voc = pd.unique(train_df['userid'])
#item_voc = pd.unique(train_df['adgroup_id'])
customer_voc = pd.unique(train_df['customer'])

# Generate click and no_click sequence
item_set = pd.unique(all_df['userid'])
#user_set = pd.unique(all_df['adgroup_id'])
customer_set = pd.unique(all_df['customer'])

user_voc_set = set(user_voc)
#item_voc_set = set(item_voc)
customer_voc_set = set(customer_voc)

#pdb.set_trace()

# 2. 根据用户集合从behavior log中筛选出相应的数据;
click_time = ""
item_attributes = []
user_attributes = []
itemAt2id, userAt2id = {}, {}
m_item, m_user = 0, 0
trainDataSize = 0
interact_items_d = {} #{user_id: item list}

#item_at_type = ["pid","cate_id","campaign_id","customer","brand"] #忽略price;
#user_at_type = ["cms_segid","cms_group_id","final_gender_code","age_level","pvalue_level","shopping_level","occupation"]

#item_at_type = ["adgroup_id"] #忽略price;
item_at_type = ["customer"] #忽略price;
user_at_type = ["userid"]

for idx, row in behavior_df.iterrows():
    if idx % 10000 == 0:
        print("Processing {} lines".format(idx))
    if row["time_stamp"] < click_time:
        sys.exit("data not sorted by timestamp!")
    user_id = row["userid"]
    #item_id = row["adgroup_id"]
    item_id = row["customer"]
    click = row['clk']

    #if item_id in item_voc_set and user_id in user_voc_set:
    if item_id in customer_voc_set and user_id in user_voc_set and click == 1:
        pass
    else:
        continue

    for item_type in item_at_type:
        item_at = row[item_type]
        #if type(item_at) != int:
        #pdb.set_trace()
        #if np.isnan(item_at):
        #if pd.isnull(item_at) or item_at not in item_voc_set:
        if pd.isnull(item_at) or item_at not in customer_voc_set:
            continue
        else:
            at_item_key = item_type + "_{}".format(item_at)
            if at_item_key not in itemAt2id:
                itemAt2id[at_item_key] = len(itemAt2id) + 1
            # item_attributes.append(itemAt2id[at_key])
            # user_attributes.append(userAt2id[at_key])

            for user_type in user_at_type:
                user_at = row[user_type]
                #pdb.set_trace()
                #if type(user_at) != int:
                if pd.isnull(user_at) or user_at not in user_voc_set:
                    continue
                else:
                    at_key = user_type + "_{}".format(user_at)
                    if at_key not in userAt2id:
                        userAt2id[at_key] = len(userAt2id) + 1
                    user_attributes.append(userAt2id[at_key])
                    item_attributes.append(itemAt2id[at_item_key])
                    #m_item = max(m_item, max(item_attributes))
                    #m_user = max(m_user, max(user_attributes))

                    # add dict
                    if userAt2id[at_key] not in interact_items_d:
                        interact_items_d[userAt2id[at_key]] = [itemAt2id[at_item_key]]
                    else:
                        interact_items_d[userAt2id[at_key]].append(itemAt2id[at_item_key])
                    trainDataSize += 1

user_attributes = np.array(user_attributes)
item_attributes = np.array(item_attributes)

#m_item = max(item_attributes) + 1
#m_user = max(user_attributes) + 1

#m_item = len(item_voc)
m_item = len(customer_voc)
m_user = len(user_voc)


print("user num: {}, item_num:{}".format(m_user, m_item))
print(f"Graph Sparsity: {(trainDataSize) / m_user / m_item}")

# save dict
dict_path ="./train_data_dict.npz"
with open(dict_path, "wb") as w_f:
    pickle.dump(interact_items_d, w_f)

pdb.set_trace()

# (users,items), bipartite graph; 0是空;
UserItemNet = csr_matrix((np.ones(len(user_attributes)), (item_attributes, user_attributes)),
                                    shape=(m_item, m_user)) #第一参数: value, 第二个采纳数: index;

print("UserItemNet:", UserItemNet.get_shape())
#pdb.set_trace()
# 3. 根据日志构建hypergraph
sp.save_npz('./user_id_Item_id_net_v2_ori_num_replace_customer_true.npz', UserItemNet)

path = "userItmeDict_v2_ori_num_replace_customer_true.pk"
f=open(path,"wb") #二进制的方式打开，如果不存在创建一个
pickle.dump((userAt2id, itemAt2id),f)#将myList列表写入f文件中
f.close() #关闭文件，关闭时自动写入
