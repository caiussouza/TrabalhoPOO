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
            `sigma (int or float)`: sigma value.
        """
        self.n_centers = None
        self.centers = None
        self.sigma = sigma
        self.w = None

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

        self.n_centers = rbf_centers.shape[0]
        self.centers = rbf_centers

    def fit_model(
        self, X_train: pd.DataFrame, y_train: np.ndarray, classification: bool = False
    ) -> None:
        # Refer to the documentation of the RBF class
        self._search_centers(X_train, y_train)
        super().fit_model(X_train, y_train, classification=classification)

    def predict(self, X_test: pd.DataFrame, classification: bool = False) -> np.ndarray:
        # Refer to the documentation of the RBF class
        return super().predict(X_test, classification=classification)
