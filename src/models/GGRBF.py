import numpy as np
import pandas as pd
from src.models.RBF import RBF
from src.models.graphs import Gabriel_Graph as GG
from typing import Union


class GGRBF(RBF):
    def __init__(self, sigma: Union[int, float]) -> None:
        """
        Instantiates a Gabriel Graph based Radial Basis Function network (GGRBF).

        ### Args:
        -   `sigma (int or float)`: Width parameter for the RBF kernel.

        ### Attributes:
        -   `n_centers (int)`: Number of centers.
        -   `centers (pd.DataFrame)`: Centers.
        -   `sigma (int or float)`: Width parameter for the RBF kernel.
        -   `w (np.ndarray)`: Weights.

        ### Example:
        ```
        # Importing necessary libraries
        >> from src.models.GGRBF import GGRBF
        >> from sklearn.datasets import load_breast_cancer
        >> from sklearn.metrics import roc_auc_score
        >> from sklearn.model_selection import train_test_split
        >> from sklearn.preprocessing import MinMaxScaler

        # Loading the dataset
        >> X, y = load_breast_cancer(return_X_y=True)

        # Preprocessing
        >> scaler = MinMaxScaler()
        >> X = scaler.fit_transform(X)
        >> X = pd.DataFrame(X)
        >> y = 2 * (y == 1) - 1
        >> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Fitting the model
        >> model = GGRBF(sigma=1)
        >> model.fit_model(X_train, y_train)
        >> y_hat = model.predict(X_test)

        # Evaluating the performance using the ROC AUC score
        >> print(roc_auc_score(y_test, y_hat))
        ```
        """
        self._n_centers: int = None
        self._centers: pd.DataFrame = None
        self._sigma: Union[int, float] = sigma
        self._w: np.ndarray = None

    def _search_centers(self, X_train: pd.DataFrame, y_train: np.ndarray) -> None:
        """
        Searches for the SSVs (centers) of the Gabriel Graph based Radial Basis Function network (GGRBF) using the training data.

        ### Args:
            `X_train (pd.DataFrame)`: Training features.
            `y_train (np.ndarray)`: Training labels.
        """
        graph = GG.Gabriel_Graph(X_train, y_train)
        graph.build()
        rbf_centers = graph.calculate_centers()

        self._n_centers = rbf_centers.shape[0]
        self._centers = rbf_centers

    def fit_model(
        self, X_train: pd.DataFrame, y_train: np.ndarray, classification: bool = False
    ) -> None:
        # Refer to the documentation of the RBF class
        self._search_centers(X_train, y_train)
        super().fit_model(X_train, y_train, classification=classification)

    def predict(self, X_test: pd.DataFrame, classification: bool = False) -> np.ndarray:
        # Refer to the documentation of the RBF class
        return super().predict(X_test, classification=classification)
