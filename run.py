from xml.dom import NotSupportedErr
from framework.trainer import Trainer, Trainer_MixNeg, Trainer_Resample
from framework.trainer import Trainer_BatchMix
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Initialize Parameters!')
    parser.add_argument('--device', default='cuda', type=str, help='cpu or cuda')
    parser.add_argument('--model', default='MF')
    parser.add_argument('--log_path',default='logs', type=str, help='path for log files')
    parser.add_argument('--data_name', default='ml-100k', type=str, help='name of dataset')
    parser.add_argument('--data_dir', default='datasets/clean_data', type=str, help='data dir')
    parser.add_argument('--split_ratio', default=0.8, type=float)
    parser.add_argument('--num_workers', default=8, type=int) 
    parser.add_argument('--fix_seed', action='store_false', help='whether to fix the seed values')
    parser.add_argument('--seed', default=10, type=int, help='random seeds')
    parser.add_argument('--optim', default='Adam', type=str, help='optimizers')
    parser.add_argument('--emb_dim', default=32, type=int)
    parser.add_argument('--epoch', default=50, type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--weight_decay', default=1e-3, type=float)
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--eval_batch_size', default=256, type=int)
    parser.add_argument('--metrics', default=['ndcg', 'recall'])
    parser.add_argument('--valid_interval', default=5, type=int)
    parser.add_argument('--topk', default=100, type=int, help='cutoff for evaluators')
    parser.add_argument('--cutoffs', default=[10, 20, 50], nargs='+', type=int)
    parser.add_argument('--steprl', action='store_false', help='whether to use steprl, default true')
    parser.add_argument('--step_size', default=5, type=int, help='step size for stepRL')
    parser.add_argument('--step_gamma', default=0.95, type=float, help='step discount for stepRL')
    parser.add_argument('--debias', default=2, type=int, help='the debias method')
    parser.add_argument('--sample_from_batch', action='store_true', help='indicate whether sampling from batch')
    parser.add_argument('--sample_from_corpus', action='store_true', help='indicate whether sampling from corpus')
    parser.add_argument('--sample_size', default=2048, type=int)
    parser.add_argument('--lambda', default=0.5, type=float, help='the coefficient to controll the cache')
    parser.add_argument('--alpha', default=1e-4, type=float, help='the lr of the streaming frequency estimation algorithm')
    parser.add_argument('--pop_mode', default=1, type=int, help='the mode of pop normalization')
    parser.add_argument('--sample_dist', default='none', type=str, help='the distribution of mixing weight sampler')
    parser.add_argument('--beta_alpha', default=0.5, type=float, help='the parameter of beta distribution')
    parser.add_argument('--beta_beta', default=0.5, type=float, help='the parameter of beta distribution')
    parser.add_argument('--temp', default=1.0, type=float, help='the temperature of softmax function')
    parser.add_argument('--mix_neg_num', default=2048, type=int, help='the number of generated negatives')
    parser.add_argument('--loss_gamma', default=1.0, type=float, help='the weight of augmentation data loss')
    parser.add_argument('--top_k', default=64, type=int)
    
    config = vars(parser.parse_args())

    if config['debias'] in [1,2,3]:
        trainer = Trainer(config)
    elif config['debias'] in [4]:
        trainer = Trainer_MixNeg(config)
    elif config['debias'] in [5]:
        trainer = Trainer_BatchMix(config)
    elif config['debias'] in [6]:
        trainer = Trainer_Resample(config)
    else:
        raise NotSupportedErr
    
    # train and validation
    # train_mat, valid_mat, test_mat = trainer.load_dataset(valid=True)
    # trainer.fit(train_mat, valid_mat, test_mat)
    
    # test
    train_mat, _, test_mat = trainer.load_dataset(valid=False)
    trainer.fit(train_mat, _, test_mat)

