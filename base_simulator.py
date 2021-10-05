import random
from typing import Tuple

import numpy as np

from model_graph import ModelGraph


class BaseSimulator:
    def __init__(self, graph_dict: dict):
        # ---- STATIC MEMBERS ----
        # Create ModelGraph object from the input graph specification
        self.model = ModelGraph(graph_dict=graph_dict)
        # Population vector
        self.pop = np.array(graph_dict["pop"])
        # number of age groups
        self.n_ag = self.model.n_ag
        # number of compartments
        self.n_comp = len(self.model.nodes)
        # compartment indexer
        self.c_idx = self.model.c_idx
        self.idx_c = {val: key for key, val in self.c_idx.items()}
        # Create dictionary for edges used for propensity matrix updates
        self.edge_dict = dict()
        self.__create_edge_dict()
        # ---- DYNAMIC MEMBERS ----
        # t = 0
        self.time_series = [0.0]  # [0,t1,t2,t3,...]
        # initial values
        self.init_vals = np.array(list(self.model.initial_values.values()))
        # state variables
        self.state_var = [self.init_vals]
        # last state
        self.last_state = np.copy(self.init_vals)
        # Transition part of propensity matrix
        self.P_trn = None
        self.create_transition_matrix()
        # Transmission part of propensity matrix
        self.P_trm = None
        self.create_transmission_matrix()

    def simulate(self, age: int, c_from: int, c_to: int) -> Tuple[int, int, int]:
        """
        Executes one simulation step
        :param int age: age for which the previous transition occured
        :param c_from: the compartment from which the previous transition removed an individual
        :param c_to: the compartment to which the previous transition added an individual
        :return Tuple[int, int, int]: current age, c_from and c_to
        """
        # Update propensity matrices
        if age is not None:
            # Transition
            self.__update_transition_matrix(age=age, c_from=c_from, c_to=c_to)
            # Transmission
            self.create_transmission_matrix()
        # Run one step for actual states
        return self.__run_one_step()

    def create_transition_matrix(self) -> None:
        """
        Create transition part of the propensity matrix (propensity = transition * transmission)
        Dimensions of transition matrix: (number of age groups, number of compartments, number of compartments)
        :return: None
        """
        # transitions consist of actual state and transition parameter
        state_mtx = np.zeros((self.n_ag, self.n_comp, self.n_comp))
        for n_from, n_to in self.model.edges:
            # get indices for nodes 'from' and 'to'
            i_from, i_to = self.c_idx[n_from], self.c_idx[n_to]
            # fill 'from' compartment from last state to the proper element array
            state_mtx[:, i_from, i_to] = self.last_state[i_from]
        # transitions are defined as: state_mtx[a] * param_mtx[a], e.g. gamma[a] * I[a] for age 'a'
        self.P_trn = state_mtx * self.model.param_mtx

    def create_transmission_matrix(self) -> None:
        """
        Create transmission part of the propensity matrix (propensity = transition * transmission)
        Dimensions of transmission matrix: (number of age groups, number of compartments, number of compartments)
        :return: None
        """
        # Transmission is taken into account for transitions, which occur as a result of a disease transmission
        # E.g. beta[a] * s[a] * (i1 + i2 + ...).dot(contact_mtx[:, a])
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
            self.P_trm[:, i_from, i_to] = inf_states.dot(self.model.contact_matrix).flatten() / self.pop

    def __create_edge_dict(self) -> None:
        """
        Creates edge dictionary used for updating propensity matrix
        Structure: {"comp_1": [transition_edge_1, transition_edge_2, ...]},
        where transition_edge_x is a tuple (c_from, c_to) associated to a transition in the model
        i.e. an out-edge from node "comp_1" in the transition graph
        :return: None
        """
        graph = self.model.graph[0]
        for node in self.model.nodes:
            self.edge_dict.update(
                {node: list(map(
                    lambda x: (self.c_idx[x[0]], self.c_idx[x[1]]),
                    list(graph.out_edges(node))))}
            )

    def __update_transition_matrix(self, age, c_from, c_to) -> None:
        """
        Updates transition part of the propensity matrix
        based on the transition occured in the previous step (described by (age, c_from, c_to))
        :param int age: age for which the previous transition occured
        :param c_from: the compartment from which the previous transition removed an individual
        :param c_to: the compartment to which the previous transition added an individual
        :return:
        """
        for ch_node in [c_to, c_from]:
            # Update all elements of the transition matrix, which are affected by the change
            # occured in the previous step
            for n_from, n_to in self.edge_dict[self.idx_c[ch_node]]:
                self.P_trn[age, n_from, n_to] = \
                    self.last_state[n_from, age] * self.model.param_mtx[age, n_from, n_to]

    def __run_one_step(self) -> Tuple[int, int, int]:
        """
        Executes steps of one simulation step:
        - sample elapsed time
        - sample occuring transition
        - execute changes
        :return Tuple[int, int, int]: (age, c_from, c_to) describes the currently occured transition
        """
        # Propensity matrix
        p_mtx = self.P_trn * self.P_trm
        a = float(np.sum(p_mtx))
        # Decide when will the next reaction occur
        tau = random.expovariate(a)  # by using the implemented pseudo random generator
        # Decide which reaction occurs based on uniform random sample
        r2 = random.uniform(0, 1)
        # Get cumulated sum of the propensity matrix
        cumsum_3d_array = np.cumsum(p_mtx).reshape(p_mtx.shape)
        # Find triplet, where cumulated sum exceeds r2 * a
        index_arrays = np.where(cumsum_3d_array > r2 * a)
        age, c_from, c_to = index_arrays[0][0], index_arrays[1][0], index_arrays[2][0]
        # Store actual time value
        self.time_series.append(self.time_series[-1] + tau)
        # Copy last state for changing it
        changed = np.copy(self.last_state)
        changed[c_from, age] -= 1
        changed[c_to, age] += 1
        # Store actual state variables
        self.state_var.append(changed)
        self.last_state = changed
        return age, c_from, c_to
