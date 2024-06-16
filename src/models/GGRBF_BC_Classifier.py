from src.models.GGRBF import GGRBF
from src.models.RBF import RBF
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np


class GGRBF_BC_Classifier(GGRBF, RBF):
    def __init__(self, sigma: float = 1) -> None:
        """
        Initializes a new instance of the GGRBF_BC_Classifier class.

        ### Args:
        -   `sigma (float, optional)`: The width value used for the RBF kernel. Defaults to 1.

        ### Attributes:
            Refer to the documentation of the GGRBF class initializer.
        -   `_sample (pd.DataFrame)`: Sample containing feature values for a cell.

        ### Example:
        ```
        # Import the class
        >> from src.models.GGRBF_BC_Classifier import GGRBF_BC_Classifier
        # Instantiate the model
        >> model = GGRBF_BC_Classifier()
        # Fit the model
        >> model.fit_model()
        # Read a sample
        # sample = np.array([...]) (Array containing feature values for a cell)
        >> model.read_sample(sample)
        # Predict the diagnosis
        >> diagnosis = 'Malignant' if x == 1 else 'Benign' if x == -1
        >> print(diagnosis)

        ```
        """
        GGRBF.__init__(self, sigma=sigma)
        self._sample: pd.DataFrame = None

    def _search_centers(self) -> None:
        """Search for centers using the GGRBF algorithm.
        Refer to the documentation of the GGRBF class for more information.
        """
        GGRBF._search_centers(self, self.X_train, self.y_train)

    def fit_model(self) -> None:
        """Fits a GGRBF model using breast cancer data from scikit-learn."""
        self._preprocessing()
        self._search_centers()
        RBF.fit_model(self, self.X_train, self.y_train, classification=True)

    def read_sample(self, sample: pd.DataFrame) -> None:
        """Reads a sample from the user interface. Used for message passing between model and interface.

        ### Args:
        -   `sample (pd.DataFrame)`: Sample containing feature values for a cell.
        """
        self._sample = sample

    def predict(self) -> int:
        """Predicts the diagnosis of the sample from the user interface.

        Returns:
            int: Binary value: 1 for malignant, -1 for benign.
        """
        self._sample = (self._sample - min(self._sample)) / (
            max(self._sample) - min(self._sample)
        )
        return GGRBF.predict(self, self._sample, classification=True)[0]

    def _preprocessing(self) -> None:
        """Preprocesses the training data (Min Max scaling and label correction to range [-1, 1]).
        The preprocessing is done only once. It uses the whole dataset.
        """
        X, y = load_breast_cancer(return_X_y=True)
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
        y = 2 * (y == 1) - 1
        X = pd.DataFrame(X)
        y = np.array(y)
        self.X_train = X
        self.y_train = y
