import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import src.Gabriel_Graph as GG
from src.RBF import RBF
from scipy.spatial.distance import cdist
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.metrics import roc_auc_score, accuracy_score
from statistics import mode


def sign(x):
    """Sign function

    ### Args:
        `x (int or float)`: Argument

    ### Returns:
        `int`: 1 if x is positive or zero, -1 if x is negative.
    """
    return 2 * (x >= 0) - 1


def two_classes_scatter(X, y, col_1="blue", col_2="red"):
    """Plots a scatter for binary bidimensional data

    ### Args:
        `X (array like)`: input data
        `y (array like)`: labels
        `col_1 (str, optional)`: Class 1 color. Defaults to "blue".
        `col_2 (str, optional)`: Class 2 color. Defaults to "red".
    """
    X = np.array(X)
    y = 1 * (y == 1)
    colors = y == 1
    colors = [col_1 if color == False else col_2 for color in colors]
    plt.scatter(X[:, 0], X[:, 1], color=colors)
    plt.show()


def make_gaussian(N1, M1, S1, N2, M2, S2, seed=42):
    """Generates two gaussian distributions on plane.

    ### Args:
        `N1 (int)`: Samples on class 1.
        `M1 (int or array like)`: Center of class 1.
        `S1 (int or float)`: Standard deviation of class 1.
        `N2 (int)`: Samples on class 2.
        `M2 (int or array like)`: Center of class 2.
        `S2 (int or float)`: Standard deviation of class 2.
        `seed (int or float, optional)`: Seed for reproductibility. Defaults to 42.

    ### Returns:
        `ndarray`: Data (X) and labels (y).
    """
    Xc1 = np.random.normal(loc=M1, scale=S1, size=(N1, 2))
    Xc2 = np.random.normal(loc=M2, scale=S2, size=(N2, 2))
    y_c1 = np.full(N1, 0).reshape(-1, 1)
    y_c2 = np.full(N2, 1).reshape(-1, 1)
    Xy_c1 = np.hstack((Xc1, y_c1))
    Xy_c2 = np.hstack((Xc2, y_c2))
    Xy = np.vstack((Xy_c1, Xy_c2))
    np.random.shuffle(Xy)

    X = Xy[:, :-1]
    y = Xy[:, -1]
    y = y.astype(int)
    return X, y


def plot_decision_surface(X, y, model):
    """Plot the decision surface based on a trained model.

    ### Args:
        `X (array like)`: Input features.
        `y (array like)`: Labels.
        `model (object)`: Trained model.
    """
    X = pd.DataFrame(X)
    x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
    y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()], classification=True)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.6)
    colors = ["blue" if i == 0 else "red" for i in y]
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=colors, edgecolors="k")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()


def GGRBF_K_Fold_Performance(
    X,
    y,
    K_kfold=10,
    wilson_editing=True,
    K_wilson=10,
    perf_metric="accuracy",
    sigma=1,
    random_state=42,
):
    X = pd.DataFrame(X)
    skf = StratifiedKFold(K_kfold, shuffle=True, random_state=random_state)
    perf_vec = []
    for train_idx, test_idx in skf.split(X, y):
        X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
        y_train_fold, y_test_fold = y[train_idx], y[test_idx]

        graph = GG.Gabriel_Graph(X_train_fold, y_train_fold)
        graph.build(wilson_editing=wilson_editing, k=K_wilson)
        rbf_centers = graph.calculate_centers()

        model = RBF()
        model.fit_model(
            X_train_fold, y_train_fold, rbf_centers, sigma, classification=True
        )

        y_hat_fold = model.predict(X_test_fold, classification=True)

        if perf_metric == "accuracy":
            perf_fold = accuracy_score(y_test_fold, y_hat_fold)
        elif perf_metric == "auc":
            perf_fold = roc_auc_score(y_test_fold, y_hat_fold)

        perf_vec.append(perf_fold)
    mean = np.mean(perf_vec)
    sd = np.std(perf_vec)
    return mean, sd
