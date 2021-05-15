import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_results(title: str, predictions: list, labels: np.ndarray):
    data = pd.DataFrame({
        'labels': labels[:110],
        'linear regression': predictions[0][:110],
        'decision tree regression': predictions[1][:110],
        'xgboost': predictions[2][:110]
    }, index=range(110))
    sns.lineplot(data=data).set(title=title, xlabel='time', ylabel='value')
    plt.show()
