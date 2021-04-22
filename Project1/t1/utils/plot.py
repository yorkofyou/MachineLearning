import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_results(predictions: np.ndarray, labels: np.ndarray):
    print(predictions.shape)
    sns.relplot(
        data=predictions[0]
    )
    sns.relplot(
        data=labels[0]
    )
    plt.show()
