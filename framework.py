import networkx as nx
import numpy as np


class ModelGraph:
    def __init__(self, graph_dict):
        self.graph_dict = graph_dict
        self.n_ag = graph_dict["age_groups"] if "age_groups" in graph_dict.keys() else 1
        self.nodes = list(graph_dict["nodes"].keys())
        self.edges = list(graph_dict["edges"].keys())
        self.edge_weights = {
            edge: np.array([val["weight"]]).flatten()
            for edge, val in graph_dict["edges"].items()}

        self.transmissions = {
            (tr_tuple[1], tr_tuple[2]): dict()
            for tr_tuple, tr_dict in graph_dict["transmission"].items()
        }
        for tr_tuple, tr_dict in graph_dict["transmission"].items():
            self.transmissions[(tr_tuple[1], tr_tuple[2])].update(
                {tr_tuple[0]: tr_dict["param"]}
            )

        self.initial_values = {
            node: np.array([node_d["init"]]).flatten()
            for node, node_d in graph_dict["nodes"].items()
        }

        self.graph = []
        self.get_graphs()

        self.matrix = [nx.to_numpy_matrix(graph) for graph in self.graph]

    def get_graphs(self):
        for ag in range(self.n_ag):
            graph = nx.DiGraph()
            graph.add_nodes_from(nodes_for_adding=self.nodes)
            weighted_edges = []
            for edge, weight in self.edge_weights.items():
                if isinstance(weight, np.ndarray):
                    if len(weight) == self.n_ag:
                        edge_weight = weight[ag]
                    else:
                        edge_weight = weight[0]
                else:
                    edge_weight = weight
                weighted_edges.append(edge + (edge_weight,))
            graph.add_weighted_edges_from(ebunch_to_add=weighted_edges)
            self.graph.append(graph)


def main():
    graph_dict_1 = {
        "nodes": {
            "S": {"init": 1000},
            "I": {"init": 1},
            "R": {"init": 0}
        },
        "edges": {
            ("S", "I"): {
                "weight": 0.01
            },
            ("I", "R"): {
                "weight": 1 / 5.0
            }
        },
        "transmission": {
            ("I", "S", "I"):
                {"param": 1.0}
        }
    }
    model = ModelGraph(graph_dict=graph_dict_1)
    print(model.matrix)

    graph_dict_2 = {
        "age_groups": 3,
        "nodes": {
            "S": {"init": [1000, 2000, 1000]},
            "I": {"init": [1, 0, 1]},
            "R": {"init": [0, 1, 0]}
        },
        "edges": {
            ("S", "I"): {
                "weight": 0.01
            },
            ("I", "R"): {
                "weight": [1 / 5.0, 1 / 4.0, 1 / 6.0]
            }
        },
        "transmission": {
            ("I", "S", "I"):
                {"param": 1.0}
        }
    }
    model = ModelGraph(graph_dict=graph_dict_2)
    print(model.matrix)


if __name__ == '__main__':
    main()
