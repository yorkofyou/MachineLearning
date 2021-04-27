import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_results(title: str, predictions: list, labels: np.ndarray):
    data = pd.DataFrame({
        'labels': labels,
        'linear regression': predictions[0],
        'decision tree regression': predictions[1],
        'xgboost': predictions[2]
    }, index=range(len(labels)))
    sns.lineplot(data=data).set(title=title, xlabel='time', ylabel='value')
    plt.show()
