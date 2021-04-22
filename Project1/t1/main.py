import pandas as pd
from models import random_forest
from models import xg_boost
from models import neural_network


if __name__ == '__main__':
    random_forest.train_and_predict('./commodity.txt')
    # xg_boost.feature_extraction('commodity.txt')
    # xg_boost.train_and_predict('./commodity.txt')
    # neural_network.train_and_predict('./commodity.txt')
