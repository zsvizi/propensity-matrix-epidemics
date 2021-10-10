import json

import networkx as nx
import numpy as np


class ModelGraph:
    def __init__(self, graph_dict):
        # List of compartments
        self.nodes = list(graph_dict["nodes"].keys())
        # Indexer for the compartments: c_idx['comp'] = idx
        self.c_idx = {key: idx for idx, key in enumerate(self.nodes)}
        # Number of age groups
        self.n_ag = graph_dict["age_groups"] if "age_groups" in graph_dict.keys() else 1
        # List of transitions
        self.edges = []
        self.__create_edges(graph_dict)
        # Dictionary of transition weights
        self.edge_weights = dict()
        self.__create_edge_weights(graph_dict)
        # Dictionary from input transmissions
        self.transmissions = dict()
        self.__create_transmissions(graph_dict)
        # List of age-specific transition graphs
        self.graph = []
        self.__get_graphs()
        # Weighted adjacency matrix = matrix of transition parameters between compartments
        self.param_mtx = np.array([nx.to_numpy_matrix(graph) for graph in self.graph])
        # Contact matrix
        if "contact_matrix" in graph_dict.keys():
            self.contact_matrix = np.array(graph_dict["contact_matrix"])
        else:
            self.contact_matrix = np.array([[1.0]])  # only one age group and no contact specification
        # Dictionary for initial values
        self.initial_values = {
            node: np.array([node_d["init"]]).flatten()
            for node, node_d in graph_dict["nodes"].items()
        }

    def __create_edges(self, graph_dict: dict) -> None:
        """
        Create list of edges represented as tuples (from, to)
        :param dict graph_dict: input graph specification
        :return: None
        """
        for key in graph_dict["edges"].keys():
            edge_tuple = self.convert_to_tuple(str_key=key)
            self.edges.append(edge_tuple)

    def __create_edge_weights(self, graph_dict: dict) -> None:
        """
        Create dictionary of edge weights: {edge: edge_weight}
        :param dict graph_dict: input graph specification
        :return: None
        """
        for edge_key, val in graph_dict["edges"].items():
            edge = self.convert_to_tuple(str_key=edge_key)
            self.edge_weights.update({edge: np.array([val["weight"]]).flatten()})
    
    def __create_transmissions(self, graph_dict: dict) -> None:
        """
        Create dictionary of transmissions: {(susceptible, infected): {infectious: param_value}}
        considering {(infectious, susceptible, infected): {'param': param_value}} from the graph specification
        :param dict graph_dict: input graph specification
        :return: None
        """
        # Initialize dictionary with keys
        for tr_key, _ in graph_dict["transmission"].items():
            tr_tuple = self.convert_to_tuple(str_key=tr_key)
            self.transmissions.update({
                (tr_tuple[1], tr_tuple[2]): dict()
            })
        # Fill values in the dictionary
        for tr_key, tr_dict in graph_dict["transmission"].items():
            tr_tuple = self.convert_to_tuple(str_key=tr_key)
            self.transmissions[(tr_tuple[1], tr_tuple[2])].update(
                {tr_tuple[0]: tr_dict["param"]}
            )
    
    def __get_graphs(self) -> None:
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

    @staticmethod
    def convert_to_tuple(str_key: str) -> tuple:
        """
        Convert string "x,y,..." to tuple of strings ("x", "y",...)
        :param str str_key: sequence given as a string
        :return tuple: tuple of strings
        """
        if isinstance(str_key, str):
            tuple_key = tuple(str_key.split(","))
        else:
            tuple_key = str_key
        return tuple_key


def main():
    graph_dict = json.load(open("../params/graph_dict.json"))
    model = ModelGraph(graph_dict=graph_dict)
    print(model.param_mtx)


if __name__ == '__main__':
    main()
