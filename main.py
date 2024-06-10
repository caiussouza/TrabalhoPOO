import src.Gabriel_Graph as GG
from src.utils import (
    two_classes_scatter,
    plot_decision_surface,
    GGRBF_K_Fold_Performance,
    make_gaussian,
)
from src.RBF import RBF
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.datasets import make_circles, make_moons, make_classification
import matplotlib.pyplot as plt

X, y = make_moons(200, noise=0.3, random_state=42)
X = pd.DataFrame(X)
two_classes_scatter(X, y)


# Divisão entre dados de treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Criação do grafo de Gabriel com dados de treinamento
grafo = GG.Gabriel_Graph(X_train, y_train)

# Para desativar a edição de Wilson, desabilite o parâmetro wilson_editing
grafo.build(wilson_editing=True, k=2)

# Alocação dos centros do grafo
rbf_centers = grafo.calculate_centers()

# Plot do grafo (se os dados forem bidimensionais)
grafo.plot(label=True, show_centers=True)

# Como a saída da RBF para classificação está no range {+1, -1}, os rótulos de teste também devem estar
y_test = 2 * (y_test == 1) - 1

# Construção e ajuste do modelo
model = RBF()
model.fit_model(X_train, y_train, rbf_centers, 1, classification=True)

# Predição em dados não vistos
y_hat = model.predict(X_test, classification=True)

# Métricas de desempenho do modelo
acc = accuracy_score(y_test, y_hat)
auc = roc_auc_score(y_test, y_hat)

print(f"A acurácia do modelo é {acc*100:.2f}%")
print(f"A AUC do modelo é {auc:.2f}")

# Visualização da superfície de decisão
plot_decision_surface(X, y, model)

# Conversão do range para {+1, -1} para se adequar às saídas da RBF
y = 2 * (y == 1) - 1

# Avaliação por meio de KFold
mean, sd = GGRBF_K_Fold_Performance(
    X, y, K_kfold=10, wilson_editing=True, K_wilson=3, perf_metric="accuracy"
)
print(f"Acurácia K-Fold: {mean*100:.2f} +- {sd*100:.2f}")
