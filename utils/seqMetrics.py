import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression


def plot_acf(sequence, dim, lags=20):
    fig, axes = plt.subplots(dim, 1, figsize=(10, 5 * dim))
    
    for i in range(dim):
        sm.graphics.tsa.plot_acf(sequence[:, i], lags=lags, ax=axes[i])
        axes[i].set_title(f'ACF for Column {i+1}')
    
    plt.tight_layout()
    plt.show()


def plot_mutual_information(sequence, dim, top_n=5, lags=None, plot=False):
    seq_len, dim = sequence.shape

    if lags == None:
        lags = seq_len - 1

    mi_matrix = np.zeros((dim, lags))
    
    for col in range(dim):
        for lag in range(1, lags + 1):
            target = sequence[lag:, col]
            features = sequence[:-lag, col]
            mi = mutual_info_regression(features.reshape(-1, 1), target)[0]
            mi_matrix[col, lag-1] = mi

    high_mi_indices = []
    
    for col in range(dim):
        top_indices = np.argsort(mi_matrix[col, :])[-top_n:]
        high_mi_indices.append((col, top_indices, mi_matrix[col, top_indices]))

    if plot and seq_len < 150:
        fig, axes = plt.subplots(dim, 1, figsize=(10, 5 * dim))
        
        for i in range(dim):
            axes[i].plot(range(1, lags + 1), mi_matrix[i, :])
            axes[i].set_title(f'Mutual Information for Column {i+1}')
            axes[i].set_xlabel('Lag')
            axes[i].set_ylabel('Mutual Information')
        
        plt.tight_layout()
        plt.show()

    return mi_matrix

