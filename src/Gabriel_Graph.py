from src.utils import sign
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import seaborn as sns
from statistics import mode

class Gabriel_Graph:
    def __init__(self, X, y, index=None, dist_method="euclidean", palette="bright"):
        """Initialize the Gabriel Graph.

        Args:
            X (pd.DataFrame or np.ndarray): Input matrix containing features (without labels).
            y (pd.DataFrame or np.ndarray): Label vector.
            index (array, optional): Index array for the input matrix. Defaults to None.
            dist_method (str, optional): Distance metric for calculating distances between samples. Defaults to 'euclidean'.
            palette (str, optional): Color palette for node representation. Defaults to 'bright'.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        self.X_ = X

        if not isinstance(y, np.ndarray):
            y = np.array(y)
        self.y_ = y

        self.index_ = index if index is not None else X.index

        assert isinstance(dist_method, str), "dist_method must be a string."
        self.dist_method_ = dist_method

        assert isinstance(palette, str), "palette must be a string."
        self.palette_ = palette

        self.centers_ = None
        self.GGraph_ = None
        self.node_locations_ = None
        self.node_colors_ = None
        self.node_ids_ = None
        self.Vp_ = None

    def build(self, wilson_editing=False, k=1):
        """Builds the Gabriel Graph.

        Args:
            wilson_editing (bool, optional): Whether to apply Wilson editing for noise reduction. Defaults to False.
            k (int, optional): The k parameter for Wilson editing. Defaults to 1.
        """
        if wilson_editing:
            self._apply_wilson_editing(k)

        self._construct_graph()

    def plot(self, label=True, show_centers=False):
        """Plots the Gabriel Graph.

        Args:
            label (bool, optional): Whether to show labels. Defaults to True.
            show_centers (bool, optional): Whether to highlight centers. Defaults to False.
        """
        if show_centers:
            assert self.centers_ is not None, "Centers were not calculated yet."
            node_colors = self._get_center_highlighted_colors()
        else:
            node_colors = self.node_colors_

        nx.draw(
            self.GGraph_,
            pos=self.node_locations_,
            node_color=node_colors,
            labels=self.node_ids_,
            with_labels=label,
            alpha=0.8,
            edgecolors="black",
        )
        plt.show()

    def adjacency_matrix(self, sparse=False):
        """Returns the adjacency matrix of the graph.

        Args:
            sparse (bool, optional): Whether to return a sparse matrix. Defaults to False.

        Returns:
            pd.DataFrame or scipy sparse matrix: Adjacency matrix representation.
        """
        adj_mat = nx.adjacency_matrix(self.GGraph_)

        if sparse:
            return adj_mat
        else:
            adj_mat = pd.DataFrame(adj_mat.toarray())
            return adj_mat

    def calculate_centers(self):
        """Calculates the centers (SSVs).

        Returns:
            pd.DataFrame: Centers (SSVs).
        """
        edges = list(self.GGraph_.edges())

        node_positions = [self.GGraph_.nodes[i]["pos"] for i in self.GGraph_.nodes]
        node_positions = pd.DataFrame(node_positions)

        centers = []
        for i in range(len(edges)):
            x1 = edges[i][0]
            x2 = edges[i][1]
            if self.GGraph_.nodes[x1]["label"] != self.GGraph_.nodes[x2]["label"]:
                centers.append(node_positions.iloc[x1, :])
                centers.append(node_positions.iloc[x2, :])
        centers = pd.DataFrame(centers).drop_duplicates()
        self.centers_ = centers
        return centers

    def _apply_wilson_editing(self, k):
        """Applies Wilson editing for noise reduction."""
        D = cdist(self.X_, self.X_, metric=self.dist_method_)
        Vp = []

        for i in range(len(D)):
            dist_vet = D[i, :]
            idx_knn = np.argsort(dist_vet)[:k + 1]
            idx_knn = np.delete(idx_knn, 0)
            k_nearest_classes = self.y_[idx_knn]
            moda = mode(k_nearest_classes)
            i_pred = moda
            if self.y_[i] == i_pred:
                Vp.append(np.hstack((self.X_.iloc[i, :].values, self.y_[i])))

        Vp = pd.DataFrame(Vp)
        self.Vp_ = Vp
        self.X_ = Vp.iloc[:, :-1]
        self.y_ = Vp.iloc[:, -1].astype(int)

    def _construct_graph(self):
        """Constructs the Gabriel Graph."""
        D = cdist(self.X_, self.X_, metric=self.dist_method_)

        GG = nx.Graph()

        for i in range(len(self.X_)):
            label = self.y_[i]
            GG.add_node(
                i,
                pos=self.X_.iloc[i, :],
                label=label,
                id=self.X_.index[i],
                color=sns.color_palette(self.palette_)[label],
            )

        for i in range(len(self.X_)):
            for j in range(i + 1, len(self.X_)):
                is_GG = True
                for k in range(len(self.X_)):
                    if i != j and j != k and i != k:
                        if D[i, j] ** 2 > (D[i, k] ** 2 + D[j, k] ** 2):
                            is_GG = False
                            break
                if is_GG:
                    GG.add_edge(i, j)

        self.GGraph_ = GG
        self.node_locations_ = nx.get_node_attributes(GG, "pos")
        self.node_colors_ = list(nx.get_node_attributes(GG, "color").values())
        self.node_ids_ = nx.get_node_attributes(GG, "id")

    def _get_center_highlighted_colors(self):
        """Returns node colors with centers highlighted."""
        color_aux_list = self.node_colors_[:]
        for i in self.GGraph_.nodes:
            if i in self.centers_.index:
                rgb_val = list(color_aux_list[i])
                rgb_val[1] += 0.5
                color_aux_list[i] = tuple(rgb_val)
        return color_aux_list
