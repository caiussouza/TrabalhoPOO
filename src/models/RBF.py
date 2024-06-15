import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from typing import Union


class RBF:
    def __init__(self, centers: pd.DataFrame, sigma: Union[int, float]) -> None:
        """
        Instantiates a Radial Basis Function based network (RBF).

        ### Args:
            `center (pd.DataFrame)`: RBF centers.
            `sigma (int or float)`: sigma value.

        For more information on Radial Basis Function Networks (RBFNs), reach [Radial Basis Function Networks(RBFNs)](https://mohamedbakrey094.medium.com/radial-basis-function-networks-rbfns-be2ec324d8fb)
        """
        self.n_centers = centers.shape[0]
        self.centers = centers.values
        self.sigma = sigma
        self.w = None

    def _gaussian_rbf(
        self,
        x: Union[int, float, pd.DataFrame],
        center: Union[int, float, pd.DataFrame],
        sigma: Union[int, float],
    ) -> Union[float, pd.DataFrame]:
        """Gaussian kernel.

        Refer to: [Gaussian kernel](https://towardsdatascience.com/radial-basis-function-rbf-kernel-the-go-to-kernel-acf0d22c798a)

        ### Args:
            `x (int, float or pd.DataFrame)`: Argument.
            `center (int, float or pd.DataFrame)`: Center.
            `sigma (int or float)`: Sigma (width parameter).

        ### Returns:
            `float or pd.DataFrame`: Output.
        """
        return np.exp(-cdist([x], [center]) / (2 * sigma**2))

    def fit_model(
        self, X_train: pd.DataFrame, y_train: np.ndarray, classification: bool = False
    ) -> None:
        """Fits the model based on features (X) and labels (y) values.

        ### Args:
            `X_train (pd.DataFrame)`: Training features.
            `y_train (np.ndarray)`: Training labels.
            `classification (bool, optional)`: If True, fit the model based on a classification binary problem. Defaults to False.
        """
        if classification:
            # Making sure that the labels are in a {+1, -1} range
            y_train = 2 * (y_train == 1) - 1
        N = X_train.shape[0]

        Phi = np.zeros((N, self.n_centers + 1))
        for lin in range(N):
            Phi[lin, 0] = 1
            for col in range(self.n_centers):
                Phi[lin, col + 1] = self._gaussian_rbf(
                    X_train.iloc[lin, :], self.centers.iloc[col, :], self.sigma
                )
        w = np.linalg.pinv(Phi.T @ Phi) @ Phi.T @ y_train

        self.w = w

    def predict(self, X_test: pd.DataFrame, classification: bool = False) -> np.ndarray:
        """Performs a prediction based on learned parameters.

        ### Args:
            `X_test (pd.DataFrame)`: Testing features.
            `classification (bool, optional)`: If True, performs a classification prediction. Defaults to False.

        ### Returns:
            `y_hat (np.ndarray)`: Array containing predictions.
        """
        N = X_test.shape[0]
        pred = np.repeat(self.w[0], N)
        for j in range(N):
            for k in range(len(self.centers)):
                pred[j] += self.w[k + 1] * self._gaussian_rbf(
                    X_test.iloc[j, :], self.centers.iloc[k, :], self.sigma
                )
        if classification:
            pred = np.sign(pred)
        return pred
