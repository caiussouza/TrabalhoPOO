import tkinter as tk
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from src.models.GGRBF_BC_Classifier import GGRBF_BC_Classifier
from typing import Type


class Breast_Cancer_APP:
    def __init__(self, model: Type[GGRBF_BC_Classifier]) -> None:
        """Instantiates the Breast_Cancer_APP class.

        ### Args:
            `model (Type[GGRBF_BC_Classifier])`: Trained model.
        """
        self._model: Type[GGRBF_BC_Classifier] = model
        self._feature_names: np.ndarray[str] = load_breast_cancer().feature_names
        self._root: Type[tk.Tk] = tk.Tk()
        self._entries: list[float] = []
        self._result_label: Type[tk.Label] = None

    def start_app(self) -> None:
        """Starts the application and displays the interface."""
        self._root.title("Classificador de Células")
        instruction = tk.Label(
            self._root,
            text="Digite as características morfológicas da célula analisada:",
        )

        instruction.grid(row=0, columnspan=2)

        for i, feature in enumerate(self._feature_names):
            tk.Label(self._root, text=feature).grid(row=i + 1, column=0)
            entry = tk.Entry(self._root)
            entry.grid(row=i + 1, column=1)
            self._entries.append(entry)

        capture_button = tk.Button(
            self._root, text="Enviar características!", command=self._send_vals
        )
        capture_button.grid(row=len(self._feature_names) + 1, columnspan=2)

        self._result_label = tk.Label(self._root, text="")
        self._result_label.grid(row=len(self._feature_names) + 2, columnspan=2)

        self._root.mainloop()

    def _send_vals(self) -> None:
        """Sends the values entered by the user to the model."""
        sample = pd.DataFrame([float(entry.get()) for entry in self._entries]).T
        self._model.read_sample(sample)
        prediction = self._model.predict()
        self._show_results(prediction)

    def _show_results(self, prediction: int) -> None:
        """Shows the results of the prediction.

        Args:
            `prediction (int)`: Prediction returned by the model.
        """
        if prediction == 1:
            result_text = "Maligno"
        elif prediction == -1:
            result_text = "Benigno"
        self._result_label.config(text=f"Resultado da classificação: {result_text}")
