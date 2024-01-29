import os
from turtle import shape
import scipy.sparse as sp
import scipy.io as sio
import numpy as np

def to_implict(mat, negs=False):
    '''
        The ratings above the average rating w.r.t each user is converted to 1s and others are converted to 1s or 0s.

        Other strategy, such as treating all interacted items as 1s, is ommitted here.
    '''
    
    # calculate average rating for each user
    (user_num, item_num) = mat.shape
    mat = mat.tocsr()
    avg_score = mat.sum(1).A.squeeze() / (mat!=0).sum(1).A.squeeze()

    user_ind_list = []
    item_ind_list = []

    for i in range(user_num):
        ratings = mat.data[mat.indptr[i]:mat.indptr[i+1]]
        indices = mat.indices[mat.indptr[i]:mat.indptr[i+1]]
        items = indices[ratings >= avg_score[i]]

        user_ind_list += [i] * len(items)
        item_ind_list += items.tolist()
    
    user_ind_list = np.array(user_ind_list)
    item_ind_list = np.array(item_ind_list)
    value_list = np.ones_like(user_ind_list)
    im_mat = sp.csc_matrix((value_list, (user_ind_list, item_ind_list)), shape=mat.shape)
    import pdb; pdb.set_trace()


    # converted_rat_list = np.zeros(mat.nnz)
    # for i in range(user_num):
    #     ratings = mat.data[mat.indptr[i]:mat.indptr[i+1]]
    #     indices = mat.indices[mat.indptr[i]:mat.indptr[i+1]]
        
    #     # pos_items = indices[ratings >= avg_score[i]]
    #     # neg_items = indices[ratings < avg_score[i]]
    #     new_rat = np.zeros_like(ratings)
    #     new_rat[ratings >= avg_score[i]] = 1.0
    #     new_rat[ratings < avg_score[i]] = -1.0
    #     converted_rat_list[mat.indptr[i]:mat.indptr[i+1]] = new_rat
    
    # im_mat = sp.csr_matrix((converted_rat_list, mat.indices, mat.indptr), shape=mat.shape)
    return im_mat
        


if __name__ == '__main__':
    data_dir = 'datasets/raw_data'
    data_name = 'ml-100k'

    filter_flag = False

    if filter_flag:
        mat_name = 'filter_mat' 
        save_mat_name = ''
    else:
        mat_name = 'raw_mat'

    file_path = os.path.join(data_dir, data_name + '.mat')
    
    mat = sio.loadmat(file_path)[mat_name]

    implict_mat = to_implict(mat)

    sio.savemat('datasets/implicit_data/' + data_name + '.mat', {'data':implict_mat})
