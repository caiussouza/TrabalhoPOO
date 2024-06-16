from src.models.GGRBF import GGRBF
from src.models.RBF import RBF
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np


class GGRBF_BC_Classifier(GGRBF, RBF):
    def __init__(self, sigma: float = 1) -> None:
        GGRBF.__init__(self, sigma=sigma)

    def _search_centers(self) -> None:
        GGRBF._search_centers(self, self.X_train, self.y_train)

    def fit_model(self) -> None:
        self._preprocessing()
        self._search_centers()
        RBF.fit_model(self, self.X_train, self.y_train, classification=True)

    def read_sample(self, sample: pd.DataFrame) -> None:
        self.sample = sample

    def predict(self) -> int:
        self.sample = (self.sample - min(self.sample)) / (
            max(self.sample) - min(self.sample)
        )
        return GGRBF.predict(self, self.sample, classification=True)[0]

    def _preprocessing(self) -> None:
        X, y = load_breast_cancer(return_X_y=True)
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
        y = 2 * (y == 1) - 1
        X = pd.DataFrame(X)
        y = np.array(y)
        self.X_train = X
        self.y_train = y
