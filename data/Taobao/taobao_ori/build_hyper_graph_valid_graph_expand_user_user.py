import numpy as np
import pandas as pd
import time
import os
import sys
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import pdb
import random
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

#raw_sample = pd.read_csv(os.path.join(datapath, "raw_sample.csv"), dtype=object)
#time_delta = time.mktime(time.strptime('2017-05-06 08:55:10', "%Y-%m-%d %H:%M:%S")) - 1494032110
#raw_sample.loc[:, 'time_stamp'] = raw_sample.loc[:, 'time_stamp'].map(lambda x: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(x) + time_delta)))
#df = raw_sample
#train_df = df[(df['time_stamp'] < '2017-05-13') & (df['time_stamp'] >= '2017-05-10')] #筛选近三天的数据;
#
##pdb.set_trace()
#user_voc = pd.unique(train_df['user'])
#user_voc_set = set(user_voc)
#with open('./user_voc_set.pk', "wb") as f: #二进制的方式打开，如果不存在创建一个
#    pickle.dump(user_voc_set, f)#将myList列表写入f文件中


#raw_sample = pd.read_csv(os.path.join(datapath, "raw_sample.csv"), dtype=object)
##raw_sample = pd.read_csv(os.path.join(datapath, "raw_sample_test.csv"), dtype=object)
#raw_sample.rename(columns={'user':'userid'}, inplace=True)
#print("raw_sample shape", raw_sample.shape)
#raw_sample = raw_sample.sort_values(by='time_stamp') #默认升序;
#
#raw_sample_with_ad = pd.merge(raw_sample, right=ad_features_df, on="adgroup_id", how="left").reset_index()
#raw_sample_with_features = pd.merge(raw_sample_with_ad, right=user_features_df, on="userid", how="left").reset_index()
#time_delta = time.mktime(time.strptime('2017-05-06 08:55:10', "%Y-%m-%d %H:%M:%S")) - 1494032110
#assert time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(1494032110 + time_delta)) == "2017-05-06 08:55:10" # make sure the timezone is correct
#raw_sample_with_features.loc[:, 'time_stamp'] = raw_sample_with_features.loc[:, 'time_stamp'].map(lambda x: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(x) + time_delta)))
#raw_sample_with_features = raw_sample_with_features.reindex(columns=["clk","time_stamp","userid","adgroup_id","pid","cate_id","campaign_id","customer","brand","price",
#"cms_segid","cms_group_id","final_gender_code","age_level","pvalue_level","shopping_level","occupation","new_user_class_level"])
#print("raw_sample_with_features shape", raw_sample_with_features.shape)
#
## Split by time as suggested by the original dataset
#df = raw_sample_with_features
#all_df = df[df['time_stamp'] < '2017-05-13']
#train_df = df[(df['time_stamp'] < '2017-05-13') & (df['time_stamp'] >= '2017-05-10')] #筛选近三天的数据;
##train_df = df[(df['time_stamp'] < '2017-05-13')] #筛选近三天的数据;
#test_df = df[df['time_stamp'] >= '2017-05-13']
#behavior_df = df[df['time_stamp'] < '2017-05-10'] # 构建超图的df;
#
## user_voc, item_voc
#user_voc = pd.unique(train_df['userid'])
##item_voc = pd.unique(train_df['adgroup_id'])
#customer_voc = pd.unique(train_df['customer'])
#
#user_voc_set = set(user_voc)
##item_voc_set = set(item_voc)
#customer_voc_set = set(customer_voc)
#
#
## 2. 根据用户集合从behavior log中筛选出相应的数据;
#click_time = ""
#item_attributes = []
#user_attributes = []
#itemAt2id, userAt2id = {}, {}
#m_item, m_user = 0, 0
#trainDataSize = 0
#interact_items_d = {} #{user_id: item list}
#
##item_at_type = ["pid","cate_id","campaign_id","customer","brand"] #忽略price;
##user_at_type = ["cms_segid","cms_group_id","final_gender_code","age_level","pvalue_level","shopping_level","occupation"]
#
##item_at_type = ["adgroup_id"] #忽略price;
#item_at_type = ["customer"] #忽略price;
#user_at_type = ["userid"]
#
#for idx, row in behavior_df.iterrows():
#    if idx % 10000 == 0:
#        print("Processing {} lines".format(idx))
#    if row["time_stamp"] < click_time:
#        sys.exit("data not sorted by timestamp!")
#    user_id = row["userid"]
#    #item_id = row["adgroup_id"]
#    item_id = row["customer"]
#    click = row['clk']
#
#    #if item_id in item_voc_set and user_id in user_voc_set:
#    #if item_id in customer_voc_set and user_id in user_voc_set:
#    #if item_id in customer_voc_set and user_id in user_voc_set and int(click) == 1: #只构建CTR标签为1的这部分数据
#    if item_id in customer_voc_set and user_id in user_voc_set and int(click) == 1: #只构建CTR标签为1的这部分数据
#    #if item_id in customer_voc_set and int(click) == 1: #只构建CTR标签为1的这部分数据, 可利用未出现过users的属性信息
#        pass
#    else:
#        continue
#
#    for item_type in item_at_type:
#        item_at = row[item_type]
#        #if type(item_at) != int:
#        #pdb.set_trace()
#        #if np.isnan(item_at):
#        #if pd.isnull(item_at) or item_at not in item_voc_set:
#        if pd.isnull(item_at) or item_at not in customer_voc_set:
#        #if pd.isnull(item_at):
#            continue
#        else:
#            at_item_key = item_type + "_{}".format(item_at)
#            if at_item_key not in itemAt2id:
#                itemAt2id[at_item_key] = len(itemAt2id) + 1
#            # item_attributes.append(itemAt2id[at_key])
#            # user_attributes.append(userAt2id[at_key])
#
#            for user_type in user_at_type:
#                user_at = row[user_type]
#                #pdb.set_trace()
#                #if type(user_at) != int:
#                if pd.isnull(user_at) or user_at not in user_voc_set:
#                #if pd.isnull(user_at):
#                    continue
#                else:
#                    at_key = user_type + "_{}".format(user_at)
#                    if at_key not in userAt2id:
#                        userAt2id[at_key] = len(userAt2id) + 1
#                    user_attributes.append(userAt2id[at_key])
#                    item_attributes.append(itemAt2id[at_item_key])
#                    #m_item = max(m_item, max(item_attributes))
#                    #m_user = max(m_user, max(user_attributes))
#
#                    # add dict, 用于构建训练数据;
#                    if userAt2id[at_key] not in interact_items_d:
#                        interact_items_d[userAt2id[at_key]] = [itemAt2id[at_item_key]]
#                    else:
#                        interact_items_d[userAt2id[at_key]].append(itemAt2id[at_item_key])
#                    trainDataSize += 1
#
#user_attributes = np.array(user_attributes)
#item_attributes = np.array(item_attributes)
#
##m_item = max(item_attributes) + 1
##m_user = max(user_attributes) + 1
#
##m_item = len(item_voc)
#m_item = len(customer_voc)
#m_user = len(user_voc)
#
#
#print("user num: {}, item_num:{}".format(m_user, m_item))
#print(f"Graph Sparsity: {(trainDataSize) / m_user / m_item}")
#
## save dict
#dict_path ="./train_data_dict_only_pos_c_u.npz"
#with open(dict_path, "wb") as w_f:
#    pickle.dump(interact_items_d, w_f)
#
#pdb.set_trace()
##根据共现频次过滤低频的pair, 保证矩阵稀疏度;
#
## (users,items), bipartite graph; 0是空;
#UserItemNet = csr_matrix((np.ones(len(user_attributes)), (item_attributes, user_attributes)),
#                                    shape=(m_item, m_user)) #第一参数: value, 第二个采纳数: index;
#
#print("UserItemNet:", UserItemNet.get_shape()) #(I, U)
#pdb.set_trace()
## 3. 根据日志构建hypergraph
#sp.save_npz('./user_id_Item_id_net_v2_ori_num_replace_customer_true_only_pos_c_u.npz', UserItemNet)
#
#path = "userItmeDict_v2_ori_num_replace_customer_true_only_pos_c_u.pk"
#f=open(path,"wb") #二进制的方式打开，如果不存在创建一个
#pickle.dump((userAt2id, itemAt2id),f)#将myList列表写入f文件中
#f.close() #关闭文件，关闭时自动写入


#========================
#with open(dict_path, "wb") as w_f:
#    pickle.dump(interact_items_d, w_f)

c_u_mat = sp.load_npz('./user_id_Item_id_net_v2_ori_num_replace_customer_true_only_pos.npz')
path = "userItmeDict_v2_ori_num_replace_customer_true_only_pos.pk"
f=open(path,"rb") #二进制的方式打开，如果不存在创建一个
user2id, item2id = pickle.load(f)#将myList列表写入f文件中
f.close() #关闭文件，关闭时自动写入

with open('./user_voc_set.pk', "rb") as f: #二进制的方式打开，如果不存在创建一个
    user_voc_set = pickle.load(f)#将myList列表写入f文件中


print("finish writing c_u_mat.")
print("user2id begfore length: ", len(user2id))
# start contruct u_u_mat
user_at_type = ["cms_segid","cms_group_id","final_gender_code","age_level","pvalue_level","shopping_level","occupation"]
at2id = {}
#user2id = {}
user_list = []
at_list = []

#pdb.set_trace()
assert len(user_voc_set) == c_u_mat.shape[1]


try:
    a_u_mat = sp.load_npz('./a_u_mat.npz')
    #sp.load_npz()
    with open('./user2id_item2id_c_u_mat.pk', "rb") as f: #二进制的方式打开，如果不存在创建一个
        user2id, item2id = pickle.load(f)#将myList列表写入f文件中
    print("user2id after length: ", len(user2id))
    m_user = c_u_mat.get_shape()[1]

except:
    for idx, row in user_features_df.iterrows():
        if idx % 10000 == 0:
            print("Processing {} lines".format(idx))
        #if row["time_stamp"] < click_time:
        #    sys.exit("data not sorted by timestamp!")
        #user_id = int(row["userid"])
        user_id = row["userid"]
        #item_id = row["adgroup_id"]
        #item_id = row["customer"]
        #click = row['clk']
        #if user_id in user_voc_set and "userid_{}".format(user_id) in user2id:

        # 只在正样本对应的用户集合上, 没有实现正样本扩充的效果;
        #if "userid_{}".format(user_id) in user2id:
        if user_id in user_voc_set:
            pass
        else:
            continue

        for a_type in user_at_type:
            user_at = row[a_type]
            if pd.isnull(user_at):
                continue
            else:
                at_key = a_type + "_{}".format(user_at)
                if at_key not in at2id:
                    at2id[at_key] = len(at2id) + 1
                    #itemAt2id[at_item_key] = len(itemAt2id) + 1
                user_key = "userid" + "_{}".format(user_id)
                #if user_key not in user2id:
                #    user2id[user_key] = len(user2id) + 1
                #user_list.append(int(user_id))
                #user_list.append(user2id[user_key])


                if user_key in user2id:
                    user_id_v = user2id[user_key]
                else:
                    user2id[user_key] = len(user2id) + 1
                    user_id_v = user2id[user_key]
                #user_list.append(user2id[user_key])

                user_list.append(user_id_v)
                at_list.append(at2id[at_key])


    print("user2id after length: ", len(user2id))
    #pdb.set_trace() # user2id是否增加呢?

    # user2id, at2id
    user_arr = np.array(user_list)
    at_arr = np.array(at_list)

    m_a = max(at_arr) + 1
    m_user = c_u_mat.get_shape()[1]
    #m_user = max(user_attributes) + 1

    # (users,items), bipartite graph; 0是空;
    a_u_mat = csr_matrix((np.ones(len(user_arr)), (at_arr, user_arr)),
                                        shape=(m_a, m_user)) #第一参数: value, 第二个采纳数: index;

    # save a_u_mat
    sp.save_npz('./a_u_mat.npz',a_u_mat)


# convert to U-U matrix
u_a_mat = a_u_mat.T

u_a_mat = u_a_mat.toarray()

u_sum = u_a_mat.sum(1) #(u)
a_sum = u_a_mat.sum(0) #(a)


a_u_mat_arr = u_a_mat.T
# a_u_mat.T.toarray()
#pdb.set_trace()


# 归一化
u_a_mat_norm = u_a_mat / u_sum[:, None] / a_sum[None, :] #(U, A)
u_a_mat_norm[np.isinf(u_a_mat_norm)] = 0.
u_a_mat_norm[np.isnan(u_a_mat_norm)] = 0.
a_u_mat_arr_norm = u_a_mat_norm.T


batch_size = 1000
exp_num = 2/6 * 1.5

iter_num = len(u_a_mat) // batch_size
row_all_list = []
col_all_list = []

for i in range(iter_num):
    if i % 10 == 1:
        print("{}/{}".format(i, iter_num))
        #break
    begin = i * batch_size
    if (i + 1) * batch_size > len(u_a_mat):
        end = len(u_a_mat)
    else:
        end = (i + 1) *batch_size


    batch_u_a_mat = u_a_mat[begin:end, :] #(b, dim)

    batch_u_u_mat = batch_u_a_mat.dot(a_u_mat_arr) #(b, U)
    # 添加过滤条件
    n_filter = 6 #保留属性均相同的用户对
    batch_u_u_mat_mask = batch_u_u_mat > n_filter
    #batch_uu_csr = sp.csr_matrix(batch_u_u_mat_mask)

    batch_u_a_mat_norm = u_a_mat_norm[begin:end, :] #(b, dim)

    batch_u_u_mat_norm = batch_u_a_mat_norm.dot(a_u_mat_arr_norm) #(b, U)

    batch_u_u_mat_mask = batch_u_u_mat_mask * batch_u_u_mat_norm

    #batch_u_u_mat_mask = batch_u_u_mat #(b, U)
    # 归一化
    #batch_u_u_norm = batch_u_u_mat_mask / a_sum[:, None] / u_sum[None, :] #(b, U)
    #bathc_u_u_norm[np.isinf(batch_u_u_norm)] = 0.

    #row_idx, col_idx = batch_uu_csr.nonzero()
    row_idx, col_idx = np.nonzero(batch_u_u_mat_mask)

    val = batch_u_u_mat_mask[(row_idx, col_idx)]
    #val = batch_u_u_mat_mask[row_idx, col_idx]

    #row_idx_reindex = row_idx + begin
    #rand_idx =list(range(len(col_idx)))
    #random.shuffle(rand_idx)


    #top_k = int(batch_size * len(u_a_mat) * 1e-6)
    top_k = min(int(batch_size * exp_num), len(col_idx))

    #pdb.set_trace()
    row_idx_reindex = row_idx + begin
    sel_val = val.argsort()[-top_k:]


    #sub_row_idx = row_idx_reindex[rand_idx][:top_k]
    #sub_col_idx = col_idx[rand_idx][:top_k]

    sub_row_idx = row_idx_reindex[sel_val]
    sub_col_idx = col_idx[sel_val]

    #row_all_list.extend(row_idx_reindex)
    #col_all_list.extend(col_idx)
    row_all_list.extend(sub_row_idx)
    col_all_list.extend(sub_col_idx)

#pdb.set_trace()
u_u_mat_csr = csr_matrix((np.ones(len(row_all_list)), (row_all_list, col_all_list)),
                                    shape=(m_user, m_user)) #第一参数: value, 第二个采纳数: index;

sp.save_npz('./att_based_u_u_mat_norm_topk.npz',first_hop_c_u_mat)
# mat mul得到扩散后的c_u_mat
pdb.set_trace()

first_hop_c_u_mat = c_u_mat.dot(u_u_mat_csr)

#add origin and mat
first_hop_c_u_mat = c_u_mat + first_hop_c_u_mat

row_col_idxs = first_hop_c_u_mat.nonzero()

merge_1_hop_c_u_mat = csr_matrix((np.ones(len(row_col_idxs[0])), row_col_idxs),
                                    shape=c_u_mat.get_shape()) #第一参数: value, 第二个采纳数: index;

#sp.save_npz('./first_hop_c_u_mat.npz',first_hop_c_u_mat)
sp.save_npz('./first_hop_c_u_mat_norm_topk.npz',first_hop_c_u_mat)
with open('./user2id_item2id_c_u_mat.pk', "wb") as f: #二进制的方式打开，如果不存在创建一个
    pickle.dump((user2id, item2id),f)#将myList列表写入f文件中

#exit(0)
