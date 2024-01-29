import logging, os, datetime, time, sys, math
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.beta import Beta
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.uniform import Uniform
from tqdm import tqdm
from line_profiler import LineProfiler

ROOT_PATH = os.path.abspath(os.path.dirname(__file__)).split('framework')[0] + 'framework'
sys.path.append(ROOT_PATH)
from dataloader import RatMixData, UserHisData, UserEvalData, pad_collate_valid
from model import TowerModel, MFModel
from debias import Base_Debias, Pop_Debias, EstPop_Debias, MixNeg_Debias, ReSample_Debias, BatchMixup_Debias
import eval as eval


def get_logger(filename, verbosity=1, name=None):
    filename = filename
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger
    

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'])
        self.init_logger()
        self.set_seed()

    def init_logger(self):
        if not os.path.exists(self.config['log_path']):
            os.makedirs(self.config['log_path'])

        ISOTIMEFORMAT = '%m%d-%H%M%S'
        timestamp = str(datetime.datetime.now().strftime(ISOTIMEFORMAT))
        seed = 'seed' + str(self.config['seed'])

        sampled_flag = 'sampled_' + str(self.config['sample_size']) if self.config[
                                                                           'sample_from_batch'] is True else 'full'
        log_name = '_'.join((self.config['data_name'], str(self.config['batch_size']), str(self.config['debias']),
                             sampled_flag, str(self.config['learning_rate']), seed, timestamp))
        os.makedirs(os.path.join(self.config['log_path'], log_name))
        log_file_name = os.path.join(self.config['log_path'], log_name)
        self.writer = SummaryWriter(log_dir=log_file_name)

        logname = log_file_name + '/log.txt'
        self.logger = get_logger(logname)
        self.logger.info(self.config)

    def set_seed(self):
        if self.config['fix_seed']:
            import os
            seed = self.config['seed']
            os.environ['PYTHONHASHSEED'] = str(seed)

            import random
            random.seed(seed)
            np.random.seed(seed)

            import torch
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True

    def load_dataset(self, valid=True):
        mldata = RatMixData(self.config['data_dir'], self.config['data_name'])

        train_mat, val_mat, test_mat = mldata.split_train_test(self.config['split_ratio'], valid)
        (M, N) = train_mat.shape
        self.logger.info('Number of Users/Items, {}/{}'.format(M, N))
        self.item_num = N + 1
        return train_mat, val_mat, test_mat

    def model_init(self, train_mat):
        (user_num, item_num) = train_mat.shape
        if self.config['model'].lower() == 'mf':
            return MFModel(user_num, item_num, self.config['emb_dim']).to(self.device)
        else:
            raise ValueError('Not supported model types')

    def config_optimizers(self, parameters, lr, wd):
        if self.config['optim'].lower() == 'adam':
            return optim.Adam(parameters, lr=lr, weight_decay=wd)
        elif self.config['optim'].lower() == 'sgd':
            return optim.SGD(parameters, lr=lr, weight_decay=wd)
        else:
            raise NotImplementedError

    def topk(self, model, query, k, user_h=None):
        more = user_h.size(1) if user_h is not None else 0
        score, topk_items = torch.topk(model.scorer(query, model.item_encoder.weight[1:]), k + more)
        if user_h is not None:
            topk_items += 1
            existing, _ = user_h.sort()
            idx_ = torch.searchsorted(existing, topk_items)
            idx_[idx_ == existing.size(1)] = existing.size(1) - 1
            score[torch.gather(existing, 1, idx_) == topk_items] = -float('inf')
            score1, idx = score.topk(k)
            return score1, torch.gather(topk_items, 1, idx)
        else:
            return score, topk_items  # B X K, B x K

    def _test_step(self, model, test_data, eval_metric, cutoffs):
        user_id, user_his, user_cand, user_rating = test_data
        user_id, user_his, user_cand, user_rating = user_id.to(self.device), user_his.to(self.device), user_cand.to(
            self.device), user_rating.to(self.device)
        rank_m = eval.get_rank_metrics(eval_metric)
        topk = self.config['topk']
        bs = user_id.size(0)
        query = model.construct_query(user_id)
        score, topk_items = self.topk(model, query, topk, user_his)
        if user_cand.dim() > 1:
            target, _ = user_cand.sort()
            idx_ = torch.searchsorted(target, topk_items)  # B x K
            idx_[idx_ == target.size(1)] = target.size(1) - 1
            label = torch.gather(target, 1, idx_) == topk_items
            pos_rating = user_rating
        else:
            label = user_cand.view(-1, 1) == topk_items
            pos_rating = user_rating.view(-1, 1)
        return [func(label, pos_rating, cutoff) for cutoff in cutoffs for name, func in rank_m], bs

    def evaluate(self, model, test_loader):
        model.eval()
        eval_metric = self.config['metrics']
        cutoffs = self.config['cutoffs']
        out_res = []
        for batch_idx, test_data in enumerate(test_loader):
            outputs = self._test_step(model, test_data, eval_metric, cutoffs)
            out_res.append(outputs)

        metric, bs = zip(*out_res)
        metric = torch.tensor(metric)
        bs = torch.tensor(bs)
        out = (metric * bs.view(-1, 1)).sum(0) / bs.sum()
        metrics = [f"{v}@{c}" for c in cutoffs for v in eval_metric]
        out = dict(zip(metrics, out))
        return out

    def _train_step(self, epoch, user_id, item_id, model: TowerModel, debias: Base_Debias, **kwargs):
        # Embedding
        query = model.construct_query(user_id)
        item_emb = model.item_encoder(item_id)
        # B = batch_size
        B = user_id.shape[0]

        scores = torch.matmul(query, item_emb.T)
        log_pos_prob = debias(item_id)
        pos_rat = torch.diag(scores)

        if self.config['sample_from_batch']:
            # Only resample method have sample_from_batch
            sample_size = min(self.config['sample_size'], B)
            # assert 1 < sample_size < B, ValueError('The number of samples must be greater than 1 and smaller than '
            #                                        'batch_size')
            # Actually, 'replacement=True'
            IndM = torch.randint(B, size=(B, sample_size), device=self.device)
            log_neg_prob = log_pos_prob[IndM]
            neg_rat = torch.gather(scores, 1, IndM)
        else:
            sample_size = B
            log_neg_prob = log_pos_prob.view(1, -1).repeat(B, 1)
            neg_rat = scores

        loss = model.loss(pos_rat, log_pos_prob, neg_rat, log_neg_prob, epoch, batch_idx=kwargs['batch_idx'])

        return loss

    def _fit(self, model: TowerModel, debias: Base_Debias, train_loader: DataLoader, valid_loader=None, test_loader=None):
        num_epoch = self.config['epoch']
        optimizer = self.config_optimizers(model.parameters(), self.config['learning_rate'],
                                           self.config['weight_decay'])

        if self.config['steprl']:
            scheduler = optim.lr_scheduler.StepLR(optimizer, self.config['step_size'], self.config['step_gamma'])

        # init mixing weight distribution
        if self.config['sample_dist'] == 'beta':
            beta_alpha = torch.tensor(self.config['beta_alpha'], device=self.device)
            beta_beta = torch.tensor(self.config['beta_beta'], device=self.device)
            sampler = Beta(beta_alpha, beta_beta)
        elif self.config['sample_dist'] == 'uniform':
            low = torch.tensor(0.0, device=self.device)
            high = torch.tensor(1.0, device=self.device)
            sampler = Uniform(low, high)
        else:
            assert 'No initial sampler'
            
        for epoch in range(num_epoch):
            loss_ = 0.0
            for batch_idx, batch_data in enumerate(tqdm(train_loader)):
                model.train()
                debias.train()
                optimizer.zero_grad()
                
                user_id, item_id = batch_data  # (tensor Bx1, tensor Bx1)
                user_id, item_id = user_id.to(self.device), item_id.to(self.device)

                if self.config['sample_dist'] in ['beta', 'uniform']:
                    loss = self._train_step(epoch, user_id, item_id, model, debias, sampler=sampler, batch_idx=batch_idx)
                
                else:
                    loss = self._train_step(epoch, user_id, item_id, model, debias, batch_idx=batch_idx)

                loss_ += loss.detach()
                loss.backward()
                optimizer.step()
                
            if self.config['steprl']:
                scheduler.step()
            self.writer.add_scalar("Train/Loss", loss_ / (batch_idx + 1.0), epoch)
            self.logger.info('Epoch {}'.format(epoch))
            self.logger.info('***************Train loss {:.8f}'.format(loss_))

            # validation
            if ((epoch % self.config['valid_interval']) == 0) or (epoch >= num_epoch - 1):
                with torch.no_grad():
                    out = self.evaluate(model, valid_loader)

                for k in out.keys():
                    self.writer.add_scalar("Evaluate/{}".format(k), out[k], epoch)
                ress = (', ').join(["{} : {:.6f}".format(k, out[k]) for k in out.keys()])

                self.logger.info('***************Eval_Res ' + ress)

            self.writer.flush()

        # test
        with torch.no_grad():
            out = self.evaluate(model, test_loader)

            for k in out.keys():
                self.writer.add_scalar("Evaluate/{}".format(k), out[k], epoch)
            ress = (', ').join(["{} : {:.6f}".format(k, out[k]) for k in out.keys()])

            self.logger.info('***************Test_Res ' + ress)
            
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
    def fit(self, train_mat, valid_mat, test_mat):
        train_data = UserHisData(train_mat=train_mat)
        train_loader = DataLoader(train_data, batch_size=self.config['batch_size'],
                                  num_workers=self.config['num_workers'], shuffle=True, pin_memory=True)
        
        test_data = UserEvalData(train_mat=train_mat, valid_mat=valid_mat, test_mat=test_mat)
        test_loader = DataLoader(test_data, batch_size=self.config['eval_batch_size'], collate_fn=pad_collate_valid,
                                 num_workers=self.config['num_workers'])
        
        model = self.model_init(train_mat=train_mat)

        # =========================================
        # Define bias mmodule
        # Base debias : uniform, Pop debias : pop
        if self.config['debias'] == 1:
            """ base debias, uniform sampling  """
            debias_module = Base_Debias(train_mat.shape[1], self.device, mode=self.config['pop_mode'])
        elif self.config['debias'] == 2:
            """ debias with popularity   """
            pop_count = train_mat.sum(axis=0).A.squeeze()
            debias_module = Pop_Debias(pop_count, self.device, mode=self.config['pop_mode'])
        elif self.config['debias'] == 3:
            debias_module = EstPop_Debias(train_mat.shape[1], self.device, self.config['alpha'], mode=self.config['pop_mode'])
        elif self.config['debias'] == 4:
            pop_count = train_mat.sum(axis=0).A.squeeze()
            debias_module = MixNeg_Debias(pop_count, self.device, mode=self.config['pop_mode'])
        elif self.config['debias'] == 5:
            pop_count = train_mat.sum(axis=0).A.squeeze()
            debias_module = BatchMixup_Debias(pop_count, self.device, mode=self.config['pop_mode'])
        elif self.config['debias'] == 6:
            pop_count = train_mat.sum(axis=0).A.squeeze()
            debias_module = ReSample_Debias(pop_count, self.device, mode=self.config['pop_mode'])
        else:
            raise NotImplementedError

        debias_module = debias_module.to(self.device)

        if valid_mat is not None:  
            valid_data = UserEvalData(train_mat=train_mat, valid_mat=valid_mat, test_mat=None)
            valid_loader = DataLoader(valid_data, batch_size=self.config['eval_batch_size'], collate_fn=pad_collate_valid,
                                    num_workers=self.config['num_workers'])        
            self._fit(model, debias_module, train_loader, valid_loader, test_loader)
        else:
            self._fit(model, debias_module, train_loader, test_loader, test_loader)
        


class Trainer_MixNeg(Trainer):
    """
        Mixed Negative Sampling for Learning Two-tower Neural Networks in Recommendations
    """
    def __init__(self, config):
        super().__init__(config)
    
    def _train_step(self, epoch, user_id, item_id, model: TowerModel, debias: MixNeg_Debias, **kwargs):
        query = model.construct_query(user_id)
        item_emb = model.item_encoder(item_id)

        sample_size = self.config['sample_size']
        # Sampling negative samples from global corpus using uniform distribution
        mixed_items = torch.randint(self.item_num, size=(sample_size,), device=self.device)
        mixed_item_emb = model.item_encoder(mixed_items)

        items = torch.cat([item_id, mixed_items], dim=-1)

        # logQ
        log_pos_prob = debias.get_pop_bias(item_id)
        ratio = (sample_size * 1.0) / (sample_size + self.config['batch_size'])

        log_neg_prob = debias(items, ratio=ratio)
        pop_scores = torch.matmul(query, item_emb.T)
        uni_scores = torch.matmul(query, mixed_item_emb.T)

        pos_rat = torch.diag(pop_scores)
        neg_rat = torch.cat([pop_scores, uni_scores], dim=-1)

        loss = model.loss(pos_rat, log_pos_prob, neg_rat, log_neg_prob, epoch, batch_idx=kwargs['batch_idx'])
        return loss
    

class Trainer_Resample(Trainer):
    def __init__(self, config):
        super().__init__(config)
    
    def _train_step(self, epoch, user_id, item_id, model: TowerModel, debias: ReSample_Debias, **kwargs):
        query = model.construct_query(user_id)
        item_emb = model.item_encoder(item_id)

        # generate the index matrix of items
        B = user_id.shape[0]
        # B = self.config['batch_size']
        if self.config['sample_from_batch']:
            sample_size = min(B, self.config['sample_size'])
        else:
            sample_size = B
        log_pop_bias = debias.get_pop_bias(item_id)

        
        scores = torch.matmul(query, item_emb.T)
        log_pos_prob, IndM, log_neg_prob = debias.resample(scores, log_pop_bias, sample_size)

        pos_rat = torch.diag(scores)
        neg_rat = torch.gather(scores, 1, IndM)
        loss = model.loss(pos_rat, log_pos_prob, neg_rat, log_neg_prob, epoch, batch_idx=kwargs['batch_idx'])
        return loss
    
    
class Trainer_BatchMix(Trainer):
    """
        Batch-Mix Sampling
        This trainer is based on sampled softmax loss
    """

    def __init__(self, config):
        super().__init__(config)

    def _train_step(self, epoch, user_id, item_id, model: TowerModel, debias: BatchMixup_Debias, sampler, **kwargs):
        # Embedding
        query = model.construct_query(user_id)
        item_emb = model.item_encoder(item_id)
        # real batch size
        B = user_id.shape[0]
        # batch index
        batch_idx = kwargs['batch_idx']

        # K => the number of generated negatives 
        mix_neg_num = self.config['mix_neg_num']
        
        weight_mat = sampler.sample((mix_neg_num, B))  # K x B
    
        # score function: inner product
        scores = torch.matmul(query, item_emb.T)
        
        # positive scores and correct logits
        pos_score = torch.diag(scores)
        log_pos_prob = debias.get_pop_bias(item_id)
        
        # inbatch negative correct logits
        log_neg_prob = log_pos_prob.view(1, -1).repeat(B, 1)
    
        # inbatch sampling
        if self.config['sample_from_batch']:
            sample_size = min(self.config['sample_size'], B)
            IndM = torch.randint(B, size=(B, sample_size), device=self.device)
            batch_neg_score = torch.gather(scores, 1, IndM)  # B x M
            log_batch_neg_prob = log_pos_prob[IndM]  # B x M
        else:
            batch_neg_score = scores
            log_batch_neg_prob = log_neg_prob

        weight_mat = weight_mat - log_pos_prob.unsqueeze(0)  # [K x B] - [1 x B]

        sample_weight = torch.softmax(weight_mat / self.config['temp'], dim=-1).detach()  # K x B

        # mixup_neg_score: B x B, K x B => B x K
        mixup_neg_score = torch.matmul(batch_neg_score, sample_weight.T)
            
        # calcluate inbatch data loss
        data_loss = model.loss(pos_score, log_pos_prob, batch_neg_score, log_batch_neg_prob, epoch, batch_idx)

        # calcluate mixup data loss
        aug_loss = model.aug_loss(pos_score, log_pos_prob, mixup_neg_score, epoch, batch_idx)

        loss = model.mixure_loss(data_loss, aug_loss, self.config['loss_gamma'], epoch, batch_idx)

        return loss
