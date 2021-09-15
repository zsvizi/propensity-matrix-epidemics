import numpy as np

from model_graph import ModelGraph
from simulator import Simulator


def main():
    # Size N of population:
    pop = 100
    # Parameters
    r0 = 3  # reproduction number
    gamma = np.array([0.4, 0.3, 0.1])  # age dependent recovery constant
    alpha = 1 / 5.0  # age-independent latency rate

    graph_dict = {
        "age_groups": 3,
        "nodes": {
            "S": {"init": [pop - 10, pop - 1, pop - 1]},
            "E": {"init": [10, 1, 1]},
            "I": {"init": [0, 0, 0]},
            "R": {"init": [0, 0, 0]}
        },
        # key pair: (state_from, state_to)
        "edges": {
            ("S", "E"): {
                "weight": r0 * gamma / pop
            },
            ("E", "I"): {
                "weight": alpha
            },
            ("I", "R"): {
                "weight": gamma
            }
        },
        # key triplet: (infectious, susceptible, infected)
        "transmission": {
            ("I", "S", "E"):
            # parameter enabling various infectivity
                {"param": 1.0}
        },
        "contact_matrix":
            [[1, 1, 1],
             [1, 1, 1],
             [1, 1, 1]]
    }

    model = ModelGraph(graph_dict=graph_dict)
    print(model.param_mtx)

    sim = Simulator(graph_dict=graph_dict)
    sim.run()


if __name__ == '__main__':
    main()
