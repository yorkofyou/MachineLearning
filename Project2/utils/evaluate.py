import numpy as np


def get_rse(y_predict, y_label):
    assert y_predict.shape == y_label.shape
    label_rse = np.std(y_label) * np.sqrt((y_label.shape[0] - 1.) / y_label.shape[0])
    return np.sqrt(np.mean((y_predict - y_label) ** 2)) / label_rse
    # return np.sqrt(np.mean((y_predict.reshape(-1) - y_label.reshape(-1)) ** 2)) / np.std(y_predict, axis=None)


def get_corr(y_predict, y_label):
    sigma_p = (y_predict).std(axis=0)
    sigma_g = (y_label).std(axis=0)
    mean_p = y_predict.mean(axis=0)
    mean_g = y_label.mean(axis=0)
    index = (sigma_g != 0)
    correlation = ((y_predict - mean_p) * (y_label - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
    correlation = (correlation[index]).mean()
    return correlation
    # assert y_predict.shape == y_label.shape
    # y_label_mean = np.mean(y_label, axis=1, keepdims=True)
    # y_predict_mean = np.mean(y_predict, axis=1, keepdims=True)
    # numerator = np.sum((y_label - y_label_mean)*(y_predict - y_predict_mean), axis=1)
    # denominator = np.sqrt(np.sum((((y_label - y_label_mean)*(y_predict - y_predict_mean))**2), axis=1))
    # return np.mean(numerator / denominator)

