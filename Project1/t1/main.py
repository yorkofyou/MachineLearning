import argparse
from models import linear_regression, decision_tree_regression, xg_boost
from utils.plot import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='commodity', help='choose dataset: commodity, electricity and traffic')
    parser.add_argument('--tau', type=int, default=10, help='choose tau for sliding windows')
    args = parser.parse_args()
    print('Training configs: {}'.format(args))
    dataset = args.dataset
    tau = args.tau
    if dataset not in ('commodity', 'electricity', 'traffic'):
        print('No dataset')
        raise ValueError
    if tau < 1:
        print('Tau error')
        raise ValueError
    if dataset == 'electricity':
        tau = 24
    elif dataset == 'traffic':
        tau = 12
    p1, labels = linear_regression.train_and_predict(dataset + '.txt', tau=tau)
    p2, _ = decision_tree_regression.train_and_predict(dataset + '.txt', tau=tau)
    p3, _ = xg_boost.train_and_predict(dataset + '.txt', tau=tau)
    plot_results(title='traffic', predictions=[p1, p2, p3], labels=labels)
