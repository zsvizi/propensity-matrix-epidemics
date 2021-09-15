import networkx as nx
import numpy as np


class ModelGraph:
    def __init__(self, graph_dict):
        # Number of age groups
        self.n_ag = graph_dict["age_groups"] if "age_groups" in graph_dict.keys() else 1
        # List of compartments
        self.nodes = list(graph_dict["nodes"].keys())
        # Indexer for the compartments: c_idx['comp'] = idx
        self.c_idx = {key: idx for idx, key in enumerate(self.nodes)}
        # List of transitions
        self.edges = list(graph_dict["edges"].keys())
        # Dictionary of transition weights: {edge: edge_weight}
        self.edge_weights = {
            edge: np.array([val["weight"]]).flatten()
            for edge, val in graph_dict["edges"].items()}
        # Contact matrix
        if "contact_matrix" in graph_dict.keys():
            self.contact_matrix = np.array(graph_dict["contact_matrix"])
        else:
            self.contact_matrix = np.array([[1.0]])

        # Dictionary from input transmissions:
        # original {(infectious, susceptible, infected): {'param': param_value}}
        # new: {(susceptible, infected): {infectious: param_value}}
        self.transmissions = {
            (tr_tuple[1], tr_tuple[2]): dict()
            for tr_tuple, _ in graph_dict["transmission"].items()
        }
        for tr_tuple, tr_dict in graph_dict["transmission"].items():
            self.transmissions[(tr_tuple[1], tr_tuple[2])].update(
                {tr_tuple[0]: tr_dict["param"]}
            )

        # Dictionary for initial values
        self.initial_values = {
            node: np.array([node_d["init"]]).flatten()
            for node, node_d in graph_dict["nodes"].items()
        }

        # List of age-specific transition graphs
        self.graph = []
        self.get_graphs()

        # Weighted adjacency matrix = matrix of transition parameters between compartments
        self.param_mtx = np.array([nx.to_numpy_matrix(graph) for graph in self.graph])

    def get_graphs(self) -> None:
        """
        Creates the list of age-specific transition graphs
        For each age groups, a separated graph is created and stored with the age-specific transition parameters
        :return: None
        """
        for ag in range(self.n_ag):
            # Create new directed graph
            graph = nx.DiGraph()
            # Nodes = compartments
            graph.add_nodes_from(nodes_for_adding=self.nodes)
            # Collect weighted edges
            weighted_edges = []
            for edge, weight in self.edge_weights.items():
                # Get proper edge_weight
                if isinstance(weight, np.ndarray):
                    if len(weight) == self.n_ag:
                        edge_weight = weight[ag]
                    elif len(weight) == 1:
                        edge_weight = weight[0]
                    else:
                        raise Exception("Weight list has to have 1 or number of age groups elements")
                else:
                    edge_weight = weight
                # Append (from, to, weight) tuple to the list of weighted edges for the actual graph
                weighted_edges.append(edge + (edge_weight,))
            # Weighted nodes = age-specific transitions
            graph.add_weighted_edges_from(ebunch_to_add=weighted_edges)
            # Append new graph to the list of graphs
            self.graph.append(graph)
