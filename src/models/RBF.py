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

        ### Attributes:
        -   `n_centers (int)`: Number of centers.
        -   `centers (pd.DataFrame)`: Centers.
        -   `sigma (int or float)`: Width parameter for the RBF kernel.
        -   `w (np.ndarray)`: Weights.

        ### Example:
        ```
        # Importing necessary libraries
        >> from src.models.RBF import RBF
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

        # centers = [matrix of centers calculated by some method, like K-means]
        >> model = RBF(sigma=1, centers=centers)
        >> model.fit_model(X_train, y_train)
        >> y_hat = model.predict(X_test)

        # Evaluating the performance using the ROC AUC score
        >> print(roc_auc_score(y_test, y_hat))
        ```

        For more information on Radial Basis Function Networks (RBFNs), reach [Radial Basis Function Networks(RBFNs)](https://mohamedbakrey094.medium.com/radial-basis-function-networks-rbfns-be2ec324d8fb)
        """
        self._n_centers: int = centers.shape[0]
        self._centers: pd.DataFrame = centers.values
        self._sigma: Union[int, float] = sigma
        self._w: np.ndarray = None

    def _rbf_kernel(
        self,
        x: Union[int, float, pd.DataFrame],
        center: Union[int, float, pd.DataFrame],
        sigma: Union[int, float],
    ) -> Union[float, pd.DataFrame]:
        """Gaussian kernel.

        Refer to: [RBF kernel](https://towardsdatascience.com/radial-basis-function-rbf-kernel-the-go-to-kernel-acf0d22c798a)

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

        Phi = np.zeros((N, self._n_centers + 1))
        for lin in range(N):
            Phi[lin, 0] = 1
            for col in range(self._n_centers):
                Phi[lin, col + 1] = self._rbf_kernel(
                    X_train.iloc[lin, :], self._centers.iloc[col, :], self._sigma
                )
        w = np.linalg.pinv(Phi.T @ Phi) @ Phi.T @ y_train

        self._w = w

    def predict(self, X_test: pd.DataFrame, classification: bool = False) -> np.ndarray:
        """Performs a prediction based on learned parameters.

        ### Args:
            `X_test (pd.DataFrame)`: Testing features.
            `classification (bool, optional)`: If True, performs a classification prediction. Defaults to False.

        ### Returns:
            `y_hat (np.ndarray)`: Array containing predictions.
        """
        N = X_test.shape[0]
        pred = np.repeat(self._w[0], N)
        for j in range(N):
            for k in range(len(self._centers)):
                pred[j] += self._w[k + 1] * self._rbf_kernel(
                    X_test.iloc[j, :], self._centers.iloc[k, :], self._sigma
                )
        if classification:
            pred = np.sign(pred)
        return pred
