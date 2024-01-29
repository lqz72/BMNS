import os, yaml, re
import numpy as np
import scipy.io as sio
import scipy.sparse as sp

def parser_yaml(config_path):
    loader = yaml.FullLoader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(
            u'''^(?:
            [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X
        ), list(u'-+0123456789.')
    )
    with open(config_path, 'r', encoding='utf-8') as f:
        ret = yaml.load(f.read(), Loader=loader)
    return ret

def load_data(file_name:str, sep:str):
    user_indices = []
    item_indices = []
    rat_values = []
    with open(file_name, 'r') as fn:
        lines = fn.readlines()
        max_user = -1
        max_item = -1
        for line in lines[1:]:
            u, i, v = line.split(sep)[:3]
            u, i = int(u) - 1, int(i) - 1
            user_indices.append(u)
            item_indices.append(i)
            rat_values.append(float(v))
            max_user = u if u > max_user else max_user
            max_item = i if i > max_item else max_item
    
    mat = sp.coo_matrix((np.array(rat_values), (np.array(user_indices), np.array(item_indices))), shape=(max_user+1, max_item+1))
    return mat

def load_data_im(file_name:str, sep:str):
    user_indices = []
    item_indices = []
    with open(file_name, 'r') as fn:
        lines = fn.readlines()
        max_user = -1
        max_item = -1
        for line in lines[1:]:
            u, i = line.split(sep)[:2]
            u, i = int(u), int(i)
            user_indices.append(u)
            item_indices.append(i)
            max_user = u if u > max_user else max_user
            max_item = i if i > max_item else max_item
    mat = sp.coo_matrix((np.ones_like(user_indices), (np.array(user_indices), np.array(item_indices))), shape=(max_user+1, max_item+1))
    return mat

        
def explicit2implict(mat:sp.spmatrix):
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
    im_mat = sp.csr_matrix((value_list, (user_ind_list, item_ind_list)), shape=mat.shape)
    return im_mat

    
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

    # ============== rating data ==========

    data_dir = 'datasets/raw_data'  # absolute path
    data_name ='ml-20m' # for change

    current_dir = os.path.join(data_dir, data_name)
    yaml_path = os.path.join(current_dir, data_name + '.yaml')

    # read yaml files
    config = parser_yaml(yaml_path)

    inter_file = os.path.join(current_dir, config['inter_feat_name'])

    datamat = load_data(inter_file, config['field_separator'])
    # datamat = filter_data(datamat, 1, 1)
    
    filtered_mat = filter_data(datamat, int(config['min_user_inter']), int(config['min_item_inter']))
    print(datamat.shape, filtered_mat.shape)
    
    # import pdb; pdb.set_trace()
    mat_name = os.path.join('datasets/clean_data', data_name + '.mat' )

    im_mat = explicit2implict(filtered_mat)
    sio.savemat(mat_name, {'data':im_mat})
    
    # ========= implicit data ===============
    # data_dir = 'datasets/raw_data'  # absolute path
    # data_name ='gowalla' # for change

    # current_dir = os.path.join(data_dir, data_name)
    # yaml_path = os.path.join(current_dir, data_name + '.yaml')

    # # read yaml files
    # config = parser_yaml(yaml_path)

    # inter_file = os.path.join(current_dir, config['inter_feat_name'])

    # datamat = load_data_im(inter_file, config['field_separator'])
    # # datamat = filter_data(datamat, 1, 1)
    
    # filtered_mat = filter_data(datamat, int(config['min_user_inter']), int(config['min_item_inter']))
    # print(datamat.shape, filtered_mat.shape)
    
    # mat_name = os.path.join('datasets/clean_data', data_name + '.mat' )

    # sio.savemat(mat_name, {'data':filtered_mat})