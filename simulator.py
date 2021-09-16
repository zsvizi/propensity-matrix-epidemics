import random
from time import time
from typing import Tuple

import numpy as np

from model_graph import ModelGraph

dur_trn = []
dur_trm = []
dur_one = []


class Simulator:
    def __init__(self, graph_dict):
        self.model = ModelGraph(graph_dict=graph_dict)

        # t = 0
        self.time_series = [0.0]  # [0,t1,t2,t3,...]

        # variables derived from model
        # initial values
        self.init_vals = np.array(list(self.model.initial_values.values()))

        # state variables
        self.state_var = [self.init_vals]

        # last state
        self.last_state = self.init_vals

        # number of age groups
        self.n_ag = self.model.n_ag

        # number of compartments
        self.n_comp = len(self.model.nodes)

        # compartment indexer
        self.c_idx = self.model.c_idx
        self.idx_c = {val: key for key, val in self.c_idx.items()}

        self.edge_dict = dict()
        self.create_edge_dict()

        # Transition part of propensity matrix
        self.P_trn = None
        self.create_transition_matrix()
        # Transmission part of propensity matrix
        self.P_trm = None
        self.create_transmission_matrix()

    def run(self):
        run_start = time()

        # Run first step
        start = time()
        age, c_from, c_to = self.run_one_step()
        end = time()
        dur_one.append(end - start)
        i = self.state_var[-1][self.c_idx["I"]]

        # Halting condition: (i = 0) <- used in while cycle
        # cntr = 0
        while np.sum(i) > 0:
            # Update propensity matrices
            # Transition
            start = time()
            self.update_transition_matrix(age, c_from, c_to)
            end = time()
            dur_trn.append(end - start)
            # Transmission
            start = time()
            self.create_transmission_matrix()
            end = time()
            dur_trm.append(end - start)

            # Run one step for actual states
            start = time()
            age, c_from, c_to = self.run_one_step()
            end = time()
            dur_one.append(end - start)

            i = self.state_var[-1][self.c_idx["I"]]

            # 6. Halting conditions are used in while cycle
            # ---------------------------------------------#
        self.state_var = np.array(self.state_var)
        run_end = time()

        if len(dur_trn) > 0:
            print("Average transition", sum(dur_trn) / len(dur_trn))
            print("Average transmission", sum(dur_trm) / len(dur_trm))
            print("Average run one step", sum(dur_one) / len(dur_one))
            print("Full run:", run_end - run_start)
        else:
            print("There was no step in the simulation!")

    def run_one_step(self) -> Tuple[int, int, int]:
        # Propensity matrix
        p_mtx = self.P_trn * self.P_trm
        a = float(np.sum(p_mtx))

        # 3. Decide when will the next reaction occur:
        # --------------------------------------------#
        tau = random.expovariate(a)  # by using the implemented pseudo random generator

        # 4. Decide which reaction occurs:
        # --------------------------------#
        r2 = random.uniform(0, 1)
        # for mu in range(len(propensities)):
        #   p = 0
        #   p += propensities[mu]
        #   if r2 * a < p: break
        # Get cumulated sum of the propensity matrix
        cumsum_3d_array = np.cumsum(p_mtx).reshape(p_mtx.shape)

        # Find triplet, where cumulated sum exceeds r2 * a
        index_arrays = np.where(cumsum_3d_array > r2 * a)
        age, c_from, c_to = index_arrays[0][0], index_arrays[1][0], index_arrays[2][0]

        # 5. Update time and state variables:
        # -----------------------------------#
        self.time_series.append(self.time_series[-1] + tau)
        # Copy last state for changing it
        changed = np.copy(self.state_var[-1])
        changed[c_from, age] -= 1
        changed[c_to, age] += 1
        # if cntr < 100:
        #   print(changed)
        #   cntr += 1
        self.state_var.append(changed)
        self.last_state = changed
        return age, c_from, c_to

    def create_edge_dict(self):
        graph = self.model.graph[0]
        for node in self.model.nodes:
            self.edge_dict.update(
                {node: list(map(
                    lambda x: (self.c_idx[x[0]], self.c_idx[x[1]]),
                    list(graph.out_edges(node))))}
            )

    def create_transition_matrix(self):
        # 2. Calculate propensity matrix:
        # --- propensity matrix has 3 dimensions: (age, comp_from, comp_to) ---
        # ----------------------------------#
        # transition are defined as: state_mtx[a] * param_mtx[a], e.g. gamma[a] * I[a] for age 'a'
        state_mtx = np.zeros((self.n_ag, self.n_comp, self.n_comp))
        for n_from, n_to in self.model.edges:
            # get indices for nodes 'from' and 'to'
            i_from, i_to = self.c_idx[n_from], self.c_idx[n_to]
            # fill 'from' compartment from last state to the proper element array
            state_mtx[:, i_from, i_to] = self.last_state[i_from]

        self.P_trn = state_mtx * self.model.param_mtx

    def create_transmission_matrix(self):
        # transmission are defined as: state_mtx[a] * param_mtx[a] * P_trm[a],
        # e.g. beta[a] * s[a] * (i1 + i2 + ...).dot(contact_mtx[:, a])
        # for age 'a' and infectious classes i1, i2, ...
        self.P_trm = np.ones((self.n_ag, self.n_comp, self.n_comp))
        for n_from, n_to in self.model.transmissions.keys():
            # get indices for nodes 'from' and 'to'
            i_from, i_to = self.c_idx[n_from], self.c_idx[n_to]

            # for selected (from, to) pair add up corresponding last infectious states w.r.t weights
            inf_states = np.zeros((1, self.n_ag))
            for inf, inf_w in self.model.transmissions[(n_from, n_to)].items():
                # get indices for infectious node
                i_inf = self.c_idx[inf]
                inf_states += inf_w * self.last_state[i_inf]

            self.P_trm[:, i_from, i_to] = inf_states.dot(self.model.contact_matrix).flatten()

    def update_transition_matrix(self, age, c_from, c_to):
        for ch_node in [c_to, c_from]:
            for n_from, n_to in self.edge_dict[self.idx_c[ch_node]]:
                self.P_trn[age, n_from, n_to] = \
                    self.last_state[n_from, age] * self.model.param_mtx[age, n_from, n_to]


def main():
    graph_dict = {
        "age_groups": 1,
        "nodes": {
            "S": {"init": 1000 - 11},
            "E": {"init": 10},
            "I": {"init": 1},
            "R": {"init": 0}
        },
        # key pair: (state_from, state_to)
        "edges": {
            ("S", "E"): {
                "weight": 2.5 * 0.25 / 1000
            },
            ("E", "I"): {
                "weight": 0.2
            },
            ("I", "R"): {
                "weight": 0.25
            }
        },
        # key triplet: (infectious, susceptible, infected)
        "transmission": {
            ("I", "S", "E"):
            # parameter enabling various infectivity
                {"param": 1.0}
        }
    }
    sim = Simulator(graph_dict=graph_dict)
    sim.run()


if __name__ == '__main__':
    main()
