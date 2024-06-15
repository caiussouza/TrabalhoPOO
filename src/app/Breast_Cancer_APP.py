import tkinter as tk
import pandas as pd
from sklearn.datasets import load_breast_cancer
from src.models.GGRBF_BC_Classifier import GGRBF_BC_Classifier
from typing import Type


class Breast_Cancer_APP:
    def __init__(self, model: Type[GGRBF_BC_Classifier]) -> None:
        self.model = model
        self._feature_names = load_breast_cancer().feature_names
        self._root = tk.Tk()
        self._entries = []
        self.label = None

    def start_app(self) -> None:
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
            self._root, text="Enviar características!", command=self.send_vals
        )
        capture_button.grid(row=len(self._feature_names) + 1, columnspan=2)

        self.result_label = tk.Label(self._root, text="")
        self.result_label.grid(row=len(self._feature_names) + 2, columnspan=2)

        self._root.mainloop()

    def send_vals(self) -> None:
        sample = pd.DataFrame([float(entry.get()) for entry in self._entries]).T
        self.model.read_sample(sample)
        prediction = self.model.predict()
        self.show_results(prediction)

    def show_results(self, prediction: int) -> None:
        if prediction == 1:
            result_text = "Maligno"
        else:
            result_text = "Benigno"
        self.result_label.config(text=f"Resultado da classificação: {result_text}")
