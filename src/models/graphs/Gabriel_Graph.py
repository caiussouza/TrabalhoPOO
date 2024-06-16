import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from src.models.graphs.Graph import Graph
import seaborn as sns


class Gabriel_Graph(Graph):
    def __init__(self, X: pd.DataFrame, y: np.ndarray) -> None:
        """
        ## Initializer for Gabriel Graph.

        ### Args:
        - `X (pd.DataFrame)`: Input matrix.
        - `y (np.ndarray)`: Label vector.

        ### Attributes:
        - `_X (pd.DataFrame)`: DataFrame containing feature values.
        - `_y (np.ndarray)`: Numpy array containing label values.
        - `_index (pd.Index)`: Index array for the input matrix.
        - `_centers (pd.DataFrame)`: Structural Support Vectors (SSV) or "Vetores de Suporte
           Estrutuais (VSE)". Read more on [Dissertação Matheus Salgado](https://repositorio.ufmg.br/bitstream/1843/RAOA-BCFHQJ/1/matheus_salgado_dissertacao__1_.pdf)
        - `_GG (nx.Graph)`: The Gabriel Graph itself represented by a networkx graph
           object.
        - `_node_locations (dict[float, float])`: Feature vector describing each sample (node) in
           the graph.
        - `_node_colors (list[tuple[float, float, float]])`: Color of each node. It is used for distinguishing
           classes or SSVs in cases when plotting is necessary.
        - `_node_ids (dict[int, int])`: Name of each node.

        ### Example:

        Creating a Gabriel Graph.

        ```
        # Importing the package
        >> import Gabriel_Graph as GG
        # Loading a dataset
        >> X, y = make_moons(200, noise=0.1, random_state=42)
        # Instantiating the graph
        >> graph = GG.Gabriel_Graph(X, y)
        # Building the graph
        >> graph.build()
        # Calculating the centers (SSVs)
        >> graph.calculate_centers()
        # Plotting the graph (only if data is binary, bidimensional and
        # labels are in a {1, 0} range)
        >> graph.plot(label=True, show_centers=True)
        ```
        """

        assert isinstance(X, pd.DataFrame), "X must be a pandas DataFrame"
        self._X: pd.DataFrame = X
        assert isinstance(y, np.ndarray), "y must be a numpy ndarray"
        self._y: np.ndarray = y

        self._index: pd.Index = X.index

        self._centers: pd.DataFrame = None
        self._GG: nx.Graph = None
        self._node_locations: dict[float, float, int] = None
        self._node_colors: list[tuple[float, float, float]] = None
        self._node_ids: dict[int, int] = None

    def build(self) -> None:
        """
        Constructs a Gabriel Graph, a proximity graph representing
        the relationships between points in the input matrix (X). If
        labels (y) are provided, it can be used for supervised learning
        tasks. Distances between points are calculated based on the method
        specified in the initializer.
        """

        D = cdist(self._X, self._X, metric="euclidean")

        GG = nx.Graph()

        palette = sns.color_palette("bright")

        for i in range(len(self._X)):
            label = self._y[i]
            GG.add_node(
                i,
                pos=self._X.iloc[i, :],
                label=label,
                id=self._X.index[i],
                color=palette[label],
            )

        for i in range(len(self._X)):
            for j in range(i + 1, len(self._X)):
                for k in range(len(self._X)):
                    is_GG = True
                    if (i != j) and (j != k) and (i != k):
                        if D[i, j] ** 2 > (D[i, k] ** 2 + D[j, k] ** 2):
                            is_GG = False
                            break
                if is_GG:
                    GG.add_edge(i, j)

        self._GG = GG
        self._node_locations = nx.get_node_attributes(GG, "pos")
        self._node_colors = list(nx.get_node_attributes(GG, "color").values())
        self._node_ids = nx.get_node_attributes(GG, "id")

    def plot(self, label: bool = True, show_centers: bool = False) -> None:
        """Plots a 2D graph if data is binary, bidimensional and labels are in
           an {1, 0} range.

        ### Args:
            - `label (bool, optional)`: Presence of labels. Defaults to True.
            - `show_centers (bool, optional)`: Flag to show centers. Defaults
               to False.
        """
        if show_centers:
            assert self._centers is not None, "Centers were not calculated."
            color_aux_list = self._node_colors.copy()
            for i in self._GG.nodes:
                if i in self._centers.index:
                    rgb_val = list(color_aux_list[i])
                    rgb_val[1] += 0.5
                    color_aux_list[i] = rgb_val

        nx.draw(
            self._GG,
            self._node_locations,
            node_color=color_aux_list if show_centers else self._node_colors,
            labels=self._node_ids,
            with_labels=label,
            alpha=0.8,
            edgecolors="black",
        )
        plt.show()

    def adjacency_matrix(self) -> pd.DataFrame:
        """Adjacency matrix representation of the graph. Is useful for
        for visualizing graphs with dimensions greater than 2 or 3.

        ### Returns:
            - `adj_mat (pandas DataFrame)`: Adjacency matrix.
        """
        adj_mat = nx.adjacency_matrix(self._GG)
        adj_mat = pd.DataFrame(adj_mat.toarray())
        return adj_mat

    def calculate_centers(self) -> pd.DataFrame:
        """Calculates and returns the Structural Support Vectors (SSV) or "Vetores de Suporte Estrutuais (VSE)". Read more on
           [Dissertação Matheus Salgado](https://repositorio.ufmg.br/bitstream/1843/RAOA-BCFHQJ/1/matheus_salgado_dissertacao__1_.pdf)

        ### Returns:
            - `centers (pd.DataFrame)`: Vector of centers (SSVs)
        """
        edges = list(self._GG.edges())

        node_pos = []
        node_labels = []
        for i in self._GG.nodes:
            node_pos.append(self._GG.nodes[i]["pos"])
            node_labels.append(self._GG.nodes[i]["label"])
        node_pos = pd.DataFrame(node_pos)
        node_labels = pd.DataFrame(node_labels)

        centers = []
        for i in range(len(edges)):
            x1 = edges[i][0]
            x2 = edges[i][1]
            if self._GG.nodes[x1]["label"] != self._GG.nodes[x2]["label"]:
                centers.append(node_pos.iloc[x1, :])
                centers.append(node_pos.iloc[x2, :])
        centers = pd.DataFrame(centers)
        centers = centers.drop_duplicates()
        self._centers = centers
        return centers
