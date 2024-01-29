import os, random
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
    

class RatMixData(object):
    # The class fits in those rating datas, with ratings ranging from different level of values.
    # For other datasets, such as CTR datasets, refer to other class.
    def __init__(self, dir:str, data:str):
        self.dir_path = dir
        self.data_name = data
        self.file_path = os.path.join(dir, data + '.mat')

    def split_train_test(self, split_ratio=0.8, valid=True):
        mat = sio.loadmat(self.file_path)['data'] # the rating matrix
        # print(mat.nnz)
        # Split train/val/test data
        
        mat = mat.tocsr()  # followed by rows(users)
        m, n = mat.shape  # m=users n=items
        train_data_indices, train_indptr = [], [0] * (m+1)
        valid_data_indices, valid_indptr = [], [0] * (m+1)
        test_data_indices, test_indptr = [], [0] * (m+1)
        tr_inter, val_inter, test_inter = 0, 0, 0
        
        # for each users
        for i in range(m):
            # row[j] = (jth item index, ranking)
            row = [(mat.indices[j], int(mat.data[j])) for j in range(mat.indptr[i], mat.indptr[i+1])]
            train_idx = random.sample(range(len(row)), round(split_ratio * len(row)))

            valid_idx = []
            if valid:
                valid_idx = train_idx[-round(0.1 * len(train_idx))-1:]
                # _valid_idx = random.sample(range(len(train_idx)), round(0.1 * len(train_idx)))
                # valid_idx = [train_idx[i] for i in _valid_idx]
            
            train_binary_idx = np.full(len(row), False)  # [False] * rows
            train_binary_idx[train_idx] = True
            test_idx = (~train_binary_idx).nonzero()[0]
            # train data
            for idx in train_idx:
                if idx in valid_idx:
                    val_inter += 1
                    valid_data_indices.append(row[idx])
                else:
                    tr_inter += 1
                    train_data_indices.append(row[idx]) 
            train_indptr[i+1] = len(train_data_indices)
            valid_indptr[i+1] = len(valid_data_indices)
            # test data
            for idx in test_idx:
                test_inter += 1
                test_data_indices.append(row[idx])
            test_indptr[i+1] = len(test_data_indices)

        [train_indices, train_data] = zip(*train_data_indices)
        [test_indices, test_data] = zip(*test_data_indices)
        
        train_mat = sp.csr_matrix((train_data, train_indices, train_indptr), (m,n))
        test_mat = sp.csr_matrix((test_data, test_indices, test_indptr), (m,n))
        
        if valid:
            [valid_indices, valid_data] = zip(*valid_data_indices)
            valid_mat = sp.csr_matrix((valid_data, valid_indices, valid_indptr), (m,n))
        else:
            valid_mat = None
        
        print('Average train sample: %.2f' % (tr_inter / m))
        print('Average valid sample: %.2f' % (val_inter / m))
        print('Average test  sample: %.2f' % (test_inter / m))
        return train_mat, valid_mat, test_mat    


class UserHisData(Dataset):
    """ Dataset for model training
    """
    def __init__(self, train_mat:sp.spmatrix):
        super().__init__()
        self.train = train_mat.tocoo()

    def __len__(self):
        return self.train.nnz
    
    def __getitem__(self, idx):
        return self.train.row[idx].astype(np.int64), self.train.col[idx].astype(np.int64) + 1


class UserEvalData(Dataset):
    """ Dataset for model evaluation
    """
    def __init__(self, train_mat, valid_mat=None, test_mat=None):
        # the max_test_num is always smaller than the number of users. Maybe modified.
        super().__init__()
        self.train, self.valid, self.test = train_mat, valid_mat, test_mat

    def __len__(self):
        return self.train.shape[0]
    
    def __getitem__(self, index):
        user_id = torch.LongTensor([index])
        user_his = self.train[index].nonzero()[1] + 1
        
        if self.test is None:
            eval_data = self.valid
        else:
            if self.valid is not None:
                user_valid = self.valid[index].nonzero()[1] + 1
                user_his = np.concatenate([user_his, user_valid], axis=-1)
            eval_data = self.test
            
        start, end = eval_data.indptr[index], eval_data.indptr[index + 1]
        user_eval = eval_data.indices[start:end] + 1
        user_rating = eval_data.data[start:end]

        return user_id, torch.LongTensor(user_his), torch.LongTensor(user_eval), torch.Tensor(user_rating)


def pad_collate_valid(batch):
    (user, user_his, items, user_rating) = zip(*batch)
    return torch.LongTensor(user), pad_sequence(user_his, batch_first=True), pad_sequence(items, batch_first=True), pad_sequence(user_rating, batch_first=True)


if __name__ == '__main__':
    dir = 'datasets/clean_data'
    data = 'gowalla'

    mldata = RatMixData(dir, data)
    train_mat, test_mat = mldata.get_train_test()
    
    train_data = UserHisData(train_mat)
    
    train_loader = DataLoader(train_data, batch_size=2048, num_workers=8, shuffle=True, pin_memory=True)

    import time
    start_time = time.time()

    # for _ in range(1):
    #     for batch_data in train_loader:
    #         user_id, neg_id = batch_data
    #         # if neg_items is None:
    #             # if neg_ is false, neg_items is None
    #             # if neg_items is not None, add the negatives into the negative set
    #             # pass

    # end_time = time.time()
    # print(end_time - start_time)

    test_data = UserEvalData(train_mat, test_mat)
    test_loader = DataLoader(test_data, batch_size=11, collate_fn=pad_collate_valid)
    for batch in test_loader:
        user_id, user_his, user_cand, user_rating = batch
        import pdb; pdb.set_trace()
