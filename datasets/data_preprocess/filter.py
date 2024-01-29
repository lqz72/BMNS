import os, yaml, re
import numpy as np
import scipy.io as sio
import scipy.sparse as sp


def filter_data(mat:sp.spmatrix, min_user_inter:int, min_item_inter:int):
    mat = mat.tocsc()

    while(True):
        (user_num, item_num) = mat.shape
        rows, cols = np.arange(user_num), np.arange(item_num)
        col_num = (mat!=0).sum(0).A.squeeze()
        col_ind = col_num >= min_item_inter
        if col_ind.sum() > 0:
            cols = cols[col_ind]
            mat = mat[:, cols]
        print(col_ind.shape, cols.shape)
        
        row_num = (mat!=0).sum(1).A.squeeze()
        row_ind = row_num >= min_user_inter
        
        if row_ind.sum() > 0:
            rows = rows[row_ind]
            mat = mat[rows, :]
        print(row_ind.shape, rows.shape)
        
        if ( (~col_ind).sum() == 0 ) and ( (~row_ind).sum() == 0) :
            print('finish')
            break
        print('success')
    return mat

if __name__ == '__main__':
    datamat = sio.loadmat('datasets/clean_data/lastfm.mat')['data']

    datamat = filter_data(datamat, 5, 5)

    sio.savemat('lastfm5.mat', {'data':datamat})