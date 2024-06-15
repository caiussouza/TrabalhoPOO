from abc import ABC, abstractmethod


class Graph(ABC):
    """
    Graph generic interface
    """

    @abstractmethod
    def build(self):
        """
        Abstract method for building a graph.
        """
        pass

    @abstractmethod
    def plot(self):
        """
        Abstract method for plotting a graph.
        """
        pass

    @abstractmethod
    def adjacency_matrix(self):
        """
        Abstract method for returning an adjacency matrix.
        """
        pass
