from models import linear_regression, decision_tree_regression, xg_boost
from utils.plot import *


if __name__ == '__main__':
    p1, labels = linear_regression.train_and_predict('commodity.txt', tau=7)
    p2, _ = decision_tree_regression.train_and_predict('commodity.txt', tau=7)
    p3, _ = xg_boost.train_and_predict('commodity.txt', tau=7)
    plot_results(title='commodity', predictions=[p1], labels=labels)
