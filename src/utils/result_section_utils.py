import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.models.GGRBF import GGRBF
from src.models.RBF import RBF
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from typing import Type, Union


def two_classes_scatter(
    X: np.ndarray, y: np.ndarray, col_1: str = "blue", col_2: str = "red"
) -> None:
    """Plots a scatter for binary bidimensional data (used in results section).

    ### Args:
        `X (np.ndarray)`: Input data.
        `y (np.ndarray)`: Labels.
        `col_1 (str, optional)`: Class 1 color. Defaults to "blue".
        `col_2 (str, optional)`: Class 2 color. Defaults to "red".
    """
    y = 1 * (y == 1)

    class_1_points = X[y == 0]
    class_2_points = X[y == 1]

    plt.scatter(
        class_1_points[:, 0],
        class_1_points[:, 1],
        color=col_1,
        label="Amostra da classe 1",
    )
    plt.scatter(
        class_2_points[:, 0],
        class_2_points[:, 1],
        color=col_2,
        label="Amostra da classe 2",
    )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_decision_surface(X: pd.DataFrame, y: np.ndarray, model: Type[GGRBF]) -> None:
    """Plot the decision surface based on a trained model (used in results section).

    ### Args:
        `X (pd.DataFrame)`: Input features.
        `y (np.ndarray)`: Labels.
        `model (class GGRBF or class RBF)`: Trained model.
    """
    x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
    y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    df_grid = np.c_[xx.ravel(), yy.ravel()]
    df_grid = pd.DataFrame(df_grid)
    Z = model.predict(df_grid, classification=True)
    Z = Z.reshape(xx.shape)

    plt.contour(xx, yy, Z, levels=[0], linewidths=1, colors="black")

    class_1 = y == 1
    class_2 = y == -1
    plt.scatter(
        X.iloc[class_1, 0],
        X.iloc[class_1, 1],
        c="blue",
        edgecolors="k",
        label="Amostra da classe 1",
    )
    plt.scatter(
        X.iloc[class_2, 0],
        X.iloc[class_2, 1],
        c="red",
        edgecolors="k",
        label="Amostra da classe 2",
    )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.legend()
    plt.show()


def GGRBF_K_Fold_Performance(
    X: pd.DataFrame,
    y: np.ndarray,
    k: int = 10,
    perf_metric: str = "accuracy",
    sigma: Union[int, float] = 1,
) -> tuple[float, float]:
    """
    Computes the k-fold cross-validation performance metric for the GGRBF model (used in results section).

    ### Args:
        - `X (pd.DataFrame)`: Input data.
        - `y (np.ndarray)`: Labels.
        - `k (int, optional)`: The number of folds for cross-validation. Defaults to 10.
        - `perf_metric (str, optional)`: The performance metric to use, either "accuracy" or "auc". Defaults to "accuracy".
        - `sigma (int or float, optional)`: The sigma value for the model. Defaults to 1.

    ### Returns:
        - `mean (float)`: The mean performance metric.
        - `sd (float)`: The standard deviation of the performance metric.
    """
    skf = StratifiedKFold(k, shuffle=True)
    perf_vec = []
    for train_idx, test_idx in skf.split(X, y):
        X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
        y_train_fold, y_test_fold = y[train_idx], y[test_idx]

        model = GGRBF(sigma=sigma)
        model.fit_model(X_train_fold, y_train_fold, classification=True)

        y_hat_fold = model.predict(X_test_fold, classification=True)

        if perf_metric == "accuracy":
            perf_fold = accuracy_score(y_test_fold, y_hat_fold)
        elif perf_metric == "auc":
            perf_fold = roc_auc_score(y_test_fold, y_hat_fold)

        perf_vec.append(perf_fold)
    mean = np.mean(perf_vec)
    sd = np.std(perf_vec)
    return mean, sd
