import random
import pickle
import numpy as np
import pandas as pd

random.seed(1234)

with open('../raw_data/reviews.pkl', 'rb') as f:
    reviews_df = pickle.load(f)
    reviews_df = reviews_df[['reviewerID', 'asin', 'unixReviewTime']]
with open('../raw_data/meta.pkl', 'rb') as f:
    meta_df = pickle.load(f)
    meta_df = meta_df[['asin', 'categories', 'price', 'brand']]
    meta_df['categories'] = meta_df['categories'].map(lambda x: x[-1][-1])

# 根据rate显性设置正负样本; 根据overall>=4设置为正样本, 其余设置为负样本;


# concat df, ['reviewerID', 'asin', 'unixReviewTime', 'categories', 'price', 'brand']
merged_left = pd.merge(left=reviews_df, right=meta_df, how='left', left_on='asin', right_on='asin')

# sort by time
merged_left = merged_left.sort_values(['reviewerID', 'unixReviewTime'])
merged_left = merged_left.reset_index(drop=True)

print(merged_left.head())

item_count = len(merged_left["asin"].unique())

# get the history sequence: [(review_id, asin, cate_str_list, label)]
pos_tuple_list = []
neg_tuple_list = []

for reviewerID, hist in merged_left.groupby('reviewerID'):
    pos_list = hist['asin'].tolist()
    def gen_neg():
        neg = pos_list[0]
        while neg in pos_list:
            neg = random.randint(0, item_count-1)
        return neg
    
    neg_list = [gen_neg() for i in range(len(pos_list))]

    for i in range(1, len(pos_list)):
        # if i == 0:
        #     # pos_tuple_list.append((reviewerID, pos_list[i], '', 1)) # 忽略没有用户行为序列的日志数据;
        #     pass
        # else:
        if type(pos_list[:i]) != list:
            click_seq = [pos_list[:i]]
        else:
            click_seq = pos_list[:i]
        pos_tuple_list.append((reviewerID, pos_list[i], '^'.join(click_seq), 1))

        # valid set and test set; only set test set; todo;

    
    # sample negative items for constructing negative sequence;
    # for i in range(1, len(neg_list)):
    #     # if i == 0:
    #     #     # neg_tuple_list.append((reviewerID, pos_list[i], '', 0))
    #     #     pass
    #     # else:
    #     if type(neg_list[:i]) != list:
    #         click_neg_seq = [neg_list[:i]]
    #     else:
    #         click_neg_seq = neg_list[:i]
    #     neg_tuple_list.append((reviewerID, pos_list[i], '^'.join(click_neg_seq), 0))

        # only set test set; todo;
        
# merge and generate
pos_df = pd.DataFrame(data=pos_tuple_list, columns=['reviewerID', 'asin', 'click_sequence', 'label'])
neg_df = pd.DataFrame(data=neg_tuple_list, columns=['reviewerID', 'asin', 'click_sequence', 'label'])

final_df = pd.merge(merged_left, pos_df, how='inner', on=['reviewerID', 'asin'])
final_df = pd.merge(final_df, neg_df, how='inner', on=['reviewerID', 'asin'])

print("final head:", final_df.head())


# generate 


# def build_map(df, col_name):
#   key = sorted(df[col_name].unique().tolist())
#   m = dict(zip(key, range(len(key))))
#   df[col_name] = df[col_name].map(lambda x: m[x])
#   return m, key

# asin_map, asin_key = build_map(meta_df, 'asin')
# cate_map, cate_key = build_map(meta_df, 'categories')
# revi_map, revi_key = build_map(reviews_df, 'reviewerID')

# user_count, item_count, cate_count, example_count =\
#     len(revi_map), len(asin_map), len(cate_map), reviews_df.shape[0]
# print('user_count: %d\titem_count: %d\tcate_count: %d\texample_count: %d' %
#       (user_count, item_count, cate_count, example_count))

# meta_df = meta_df.sort_values('asin')
# meta_df = meta_df.reset_index(drop=True)
# reviews_df['asin'] = reviews_df['asin'].map(lambda x: asin_map[x])
# reviews_df = reviews_df.sort_values(['reviewerID', 'unixReviewTime'])
# reviews_df = reviews_df.reset_index(drop=True)
# reviews_df = reviews_df[['reviewerID', 'asin', 'unixReviewTime']]

# cate_list = [meta_df['categories'][i] for i in range(len(asin_map))]
# cate_list = np.array(cate_list, dtype=np.int32)


# with open('../raw_data/remap.pkl', 'wb') as f:
#   pickle.dump(reviews_df, f, pickle.HIGHEST_PROTOCOL) # uid, iid
#   pickle.dump(cate_list, f, pickle.HIGHEST_PROTOCOL) # cid of iid line
#   pickle.dump((user_count, item_count, cate_count, example_count),
#               f, pickle.HIGHEST_PROTOCOL)
#   pickle.dump((asin_key, cate_key, revi_key), f, pickle.HIGHEST_PROTOCOL)


import random
import pickle

random.seed(1234)

# with open('../raw_data/remap.pkl', 'rb') as f:
#   reviews_df = pickle.load(f)
#   cate_list = pickle.load(f)
#   user_count, item_count, cate_count, example_count = pickle.load(f)

train_set = []
test_set = []
for reviewerID, hist in reviews_df.groupby('reviewerID'):
  pos_list = hist['asin'].tolist()
  def gen_neg():
    neg = pos_list[0]
    while neg in pos_list:
      neg = random.randint(0, item_count-1)
    return neg
  neg_list = [gen_neg() for i in range(len(pos_list))]

  for i in range(1, len(pos_list)):
    hist = pos_list[:i]
    if i != len(pos_list) - 1:
      train_set.append((reviewerID, hist, pos_list[i], 1))
      train_set.append((reviewerID, hist, neg_list[i], 0))
    else:
      label = (pos_list[i], neg_list[i])
      test_set.append((reviewerID, hist, label))

# random.shuffle(train_set)
# random.shuffle(test_set)

# assert len(test_set) == user_count
# assert(len(test_set) + len(train_set) // 2 == reviews_df.shape[0])

# with open('dataset.pkl', 'wb') as f:
#   pickle.dump(train_set, f, pickle.HIGHEST_PROTOCOL)
#   pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)
#   pickle.dump(cate_list, f, pickle.HIGHEST_PROTOCOL)
#   pickle.dump((user_count, item_count, cate_count), f, pickle.HIGHEST_PROTOCOL)
