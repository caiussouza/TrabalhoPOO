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
        - `X_`: DataFrame containing feature values.
        - `y_`: Numpy array containing label values.
        - `centers_`: Structural Support Vectors (SSV) or "Vetores de Suporte
           Estrutuais (VSE)". Read more on [Dissertação Matheus Salgado](https://repositorio.ufmg.br/bitstream/1843/RAOA-BCFHQJ/1/matheus_salgado_dissertacao__1_.pdf)
        - `GG`: The Gabriel Graph itself represented by a networkx graph
           object.
        - `node_locations_`: Feature vector describing each sample (node) in
           the graph.
        - `node_colors_`: Color of each node. It is used for distinguishing
           classes or SSVs in cases when plotting is necessary.
        - `node_ids_`: Name of each node.

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
        self.X_ = X
        assert isinstance(y, np.ndarray), "y must be a numpy ndarray"
        self.y_ = y

        self.index_ = X.index

        self.centers_ = None
        self.GG = None
        self.node_locations_ = None
        self.node_colors_ = None
        self.node_ids_ = None

    def build(self) -> None:
        """
        Constructs a Gabriel Graph, a proximity graph representing
        the relationships between points in the input matrix (X). If
        labels (y) are provided, it can be used for supervised learning
        tasks. Distances between points are calculated based on the method
        specified in the initializer.
        """

        D = cdist(self.X_, self.X_, metric="euclidean")

        GG = nx.Graph()

        palette = sns.color_palette("bright")

        for i in range(len(self.X_)):
            label = self.y_[i]
            GG.add_node(
                i,
                pos=self.X_.iloc[i, :],
                label=label,
                id=self.X_.index[i],
                color=palette[label],
            )

        for i in range(len(self.X_)):
            for j in range(i + 1, len(self.X_)):
                for k in range(len(self.X_)):
                    is_GG = True
                    if (i != j) and (j != k) and (i != k):
                        if D[i, j] ** 2 > (D[i, k] ** 2 + D[j, k] ** 2):
                            is_GG = False
                            break
                if is_GG:
                    GG.add_edge(i, j)

        self.GG = GG
        self.node_locations_ = nx.get_node_attributes(GG, "pos")
        self.node_colors_ = list(nx.get_node_attributes(GG, "color").values())
        self.node_ids_ = nx.get_node_attributes(GG, "id")

    def plot(self, label: bool = True, show_centers: bool = False) -> None:
        """Plots a 2D graph if data is binary, bidimensional and labels are in
           an {1, 0} range.

        ### Args:
            - `label (bool, optional)`: Presence of labels. Defaults to True.
            - `show_centers (bool, optional)`: Flag to show centers. Defaults
               to False.
        """
        if show_centers:
            assert self.centers_ is not None, "Centers were not calculated."
            color_aux_list = self.node_colors_.copy()
            for i in self.GG.nodes:
                if i in self.centers_.index:
                    rgb_val = list(color_aux_list[i])
                    rgb_val[1] += 0.5
                    color_aux_list[i] = rgb_val

        nx.draw(
            self.GG,
            self.node_locations_,
            node_color=color_aux_list if show_centers else self.node_colors_,
            labels=self.node_ids_,
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
        adj_mat = nx.adjacency_matrix(self.GG)
        adj_mat = pd.DataFrame(adj_mat.toarray())
        return adj_mat

    def calculate_centers(self) -> pd.DataFrame:
        """Calculates the Structural Support Vectors (SSV) or "Vetores de Suporte Estrutuais (VSE)". Read more on
           [Dissertação Matheus Salgado](https://repositorio.ufmg.br/bitstream/1843/RAOA-BCFHQJ/1/matheus_salgado_dissertacao__1_.pdf)

        ### Returns:
            - `centers (pd.DataFrame)`: Vector of centers (SSVs)
        """
        edges = list(self.GG.edges())

        node_pos = []
        node_labels = []
        for i in self.GG.nodes:
            node_pos.append(self.GG.nodes[i]["pos"])
            node_labels.append(self.GG.nodes[i]["label"])
        node_pos = pd.DataFrame(node_pos)
        node_labels = pd.DataFrame(node_labels)

        centers = []
        for i in range(len(edges)):
            x1 = edges[i][0]
            x2 = edges[i][1]
            if self.GG.nodes[x1]["label"] != self.GG.nodes[x2]["label"]:
                centers.append(node_pos.iloc[x1, :])
                centers.append(node_pos.iloc[x2, :])
        centers = pd.DataFrame(centers)
        centers = centers.drop_duplicates()
        self.centers_ = centers
        return centers
