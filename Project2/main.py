import argparse
import random
import numpy as np
import torch
from models import ridge, svr, mlp, gru
from utils.plot import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='ridge', help='model to run')
    parser.add_argument('--dataset', type=str, default='commodity', help='choose dataset: commodity, electricity and traffic')
    parser.add_argument('--horizon', type=int, default=3)
    parser.add_argument('--seed', type=int, default=54321, help='random seed')
    parser.add_argument('--n_jobs', type=int, default=1, help='number of workers')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout applied to layers (0 = no dropout)')
    args = parser.parse_args()
    print('Training configs: {}'.format(args))
    model = args.model
    dataset = args.dataset
    if dataset not in ('commodity', 'electricity', 'traffic'):
        print('No dataset')
        raise ValueError
    horizon = args.horizon
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    n_jobs = args.n_jobs
    dropout = args.dropout
    if model == 'ridge':
        ridge.train_and_predict('datasets/' + dataset + '.txt', horizon=horizon, n_jobs=n_jobs)
    elif model == 'svr':
        svr.train_and_predict('datasets/' + dataset + '.txt', horizon=horizon, n_jobs=n_jobs)
    elif model == 'mlp':
        mlp.train_and_predict('datasets/' + dataset + '.txt', horizon=horizon, dropout=dropout)
    elif model == 'gru':
        gru.train_and_predict('datasets/' + dataset + '.txt', horizon=horizon, dropout=dropout)
    else:
        raise NotImplementedError
