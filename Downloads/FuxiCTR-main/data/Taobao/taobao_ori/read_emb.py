import h5py
import pdb
#file_name = 'user_emb_hypergraph_dim10_final_epoch_20.h5'
#file_name = 'user_emb_hypergraph_dim10_final.h5' # taobao_x1_sub_seq_only_user_min_1_pretrain_total
#file_name = 'user_emb_hypergraph_dim10_final_output_epoch_20.h5' #user_emb_hypergraph_dim10_final_output_epoch_20
file_name = 'user_emb_hypergraph_dim10_final_output_epoch_20_add_norm.h5'
file_name = 'user_emb_hypergraph_dim10_final_output_epoch_20_ori_num.h5'
file_name = 'user_emb_hypergraph_dim10_final_output_epoch_20_ori_num_output.h5'
#file_name = 'user_emb_hypergraph_dim10_final_output_epoch_20_ori_num_input_v2.h5'


# 加载到ctr模型的user embeddings
#file_name = "../../../data/taobao_x1_sub_seq_only_user_min_1_pretrain_output_total_epoch_20/pretrained_userid.h5"
#file_name = "../../../data/taobao_x1_sub_seq_only_user_min_1_pretrain_total/pretrained_userid.h5"

mode = 'r'
f = h5py.File(file_name, mode)

for key in f.keys():
    print(key) #Names of the root level object names in HDF5 file - can be groups or datasets.
    print(type(f[key])) # get the object type: usually group or dataset
    # f['value'][100:120]
    # f['userid'][100:120]
    pdb.set_trace()
