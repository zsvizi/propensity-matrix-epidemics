import os
from typing import Tuple

import numpy as np

from base_simulator import BaseSimulator


class Simulator(BaseSimulator):
    def __init__(self, graph_dict):
        super().__init__(graph_dict=graph_dict)

    def run(self) -> None:
        """
        Basic run function
        :return: None
        """
        # Reset age, c_from, c_to
        age, c_from, c_to = None, None, None
        # initialize variable for halting condition
        actual_time, e, i = self.get_halting_parameters()
        # Loop for one stochastic simulation with halting conditions for state and time
        while np.sum(i + e) > 0 and actual_time < 365:
            # Run simulation
            age, c_from, c_to = self.simulate(age=age, c_from=c_from, c_to=c_to)
            # Calculation for halting condition
            actual_time, e, i = self.get_halting_parameters()
        # Convert list to numpy array for later usage
        self.state_var = np.array(self.state_var)

    def run_after_peak(self, days: int = 10) -> None:
        """
        Run function halting after first peak in the epidemics curve is reached
        :param int days: how many days after peak is still considered
        :return: None
        """
        # Reset age, c_from, c_to
        age, c_from, c_to = None, None, None
        # initialize variable for halting condition
        actual_time, e, i = self.get_halting_parameters()
        # Halting condition
        halting_condition = False
        # Variables for daily sampling
        day = 1
        day_i = [np.sum(i)]
        days_l = [day]
        # Loop for one stochastic simulation with halting conditions for state and time
        while np.sum(i + e) > 0 and actual_time < 365 and not halting_condition:
            # Run simulation
            age, c_from, c_to = self.simulate(age=age, c_from=c_from, c_to=c_to)
            # Calculation for halting condition
            actual_time, e, i = self.get_halting_parameters()
            # Update halting condition
            if len(self.time_series) >= 2:
                prev_time = self.time_series[-2]
                if (actual_time - prev_time) > 1.0:
                    # Fill jumped dates with constant data
                    i_values = np.sum(self.state_var, axis=2)[:, self.c_idx["I"]]
                    for d in range(day, int(actual_time) + 1):
                        days_l.append(d)
                        day_i.append(i_values[-2])
                    day = int(actual_time) + 1
                # Check whether we changed from one day to another
                elif prev_time <= day < actual_time:
                    days_l.append(day)
                    day = int(actual_time) + 1
                    i_values = np.sum(self.state_var, axis=2)[:, self.c_idx["I"]]
                    day_i.append(i_values[-2])
                # calculation for updating the halting condition
                i_max_position = np.argmax(day_i)
                i_max = day_i[i_max_position]
                # update halting condition
                if days_l[-1] >= i_max_position + days:
                    halting_condition = np.all(day_i[i_max_position:i_max_position + days] <= i_max)
        # Convert list to numpy array for later usage
        self.state_var = np.array(self.state_var)

    def run_conf(self) -> int:
        """
        Run function for generating confidence intervals from the simulations
        :return int: number of needed simulations to have enough simulations for evaluations
        """
        # Reset age, c_from, c_to
        age, c_from, c_to = None, None, None
        # Initialize variable for halting condition
        actual_time, e, i = self.get_halting_parameters()
        # Create data directory, if not exists
        os.makedirs("./data", exist_ok=True)
        # Number for minimal number of simulations
        min_n_sim = 100
        # Maximal time until the simulations run (calibrated for parametrization in demo.ipynb)
        max_time = 100
        # Threshold for convergence
        std_threshold = 2.0
        conf_threshold = 0.95
        # Initialize simulation counter
        sim_cnt = 0
        # Create list for peak sizes
        peak_sizes = []
        # Absolute change
        is_converged = True
        # Loop for running multiple simulations
        while sim_cnt < min_n_sim or not is_converged:
            # Loop for one stochastic simulation with halting conditions for state and time
            while actual_time < max_time and np.sum(i + e) > 0:
                # Run simulation
                age, c_from, c_to = self.simulate(age=age, c_from=c_from, c_to=c_to)
                # Calculation for halting condition
                actual_time, e, i = self.get_halting_parameters()
            self.state_var = np.array(self.state_var)
            # Save simulation output
            np.savez_compressed("./data/simulation_" + str(sim_cnt) + ".npz",
                                t=np.array(self.time_series),
                                y=self.state_var,
                                c=np.array(list(self.c_idx.keys())))
            # Save peak size
            i_agg = np.amax(np.sum(self.state_var[:, self.c_idx["I"], :], axis=1))
            peak_sizes.append(i_agg)
            # Increment simulation counter
            sim_cnt += 1
            # Reset for the new simulation
            self.reset_for_new_simulation()
            # Reset age, c_from, c_to
            age, c_from, c_to = None, None, None
            # Initialize variable for halting condition
            actual_time, e, i = self.get_halting_parameters()
            # Calculation for the outer halting condition
            if sim_cnt >= min_n_sim:
                peak_sizes_np = np.array(peak_sizes)
                m = np.mean(peak_sizes_np)  # mean
                s = np.std(peak_sizes_np)  # standard deviation
                peak_sizes_standard = (peak_sizes_np - m) / s  # normalized peak sizes
                # abs value of last min_n_sim number of elements
                abs_peak_sizes = np.abs(peak_sizes_standard[-min_n_sim:])
                # get number of normalized peaks, which are inside [-std_threshold * std, +std_threshold * std]
                n_peaks_inside_std = np.sum(abs_peak_sizes < std_threshold)
                # convergence occurs, if enough peaks are already in the confidence range
                is_converged = n_peaks_inside_std > (conf_threshold * min_n_sim)
                print("Simulation #" + str(sim_cnt) + ":", n_peaks_inside_std)
            else:
                if sim_cnt % 10 == 0:
                    print("Simulation #" + str(sim_cnt))
        return sim_cnt

    def get_halting_parameters(self) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Returns with components needed for calculating the halting condition for the simulation loops
        :return Tuple[float, np.ndarray, np.ndarray]: (actual time, infecteds, expecteds)
        """
        i = self.state_var[-1][self.c_idx["I"]]
        e = self.state_var[-1][self.c_idx["E"]]
        actual_time = self.time_series[-1]
        return actual_time, e, i

    def reset_for_new_simulation(self, init: np.ndarray = None) -> None:
        """
        Reset function for restarting simulation setup
        :param np.ndarray init: new initial value
        :return: None
        """
        if init is None:
            init = np.copy(self.init_vals)
        self.init_vals = init
        self.last_state = init
        self.time_series = [0.0]
        self.state_var = [init]
        self.create_transition_matrix()
        self.create_transmission_matrix()


def generate_daily_data(time_data: np.ndarray,
                        values: np.ndarray,
                        day: int = 1) -> Tuple[list, list]:
    """
    Generates days from input time data array and
    daily sampled values from input values array
    :param int day: starting day of the simulation
    :param np.ndarray time_data: time points
    :param np.ndarray values: values associated with the time points
    :return Tuple[list, list]: (days, daily sampled values)
    """
    # Initialization
    days, day_i = [], []
    t_prev, i_prev = None, None
    # Loop through all time steps taking the aggregated infections
    for t, y in zip(time_data, values):
        # After taking initial data
        if t_prev is not None:
            # If elapsed time is larger than 1 day
            if (t - t_prev) > 1.0:
                # Fill jumped dates with constant data
                # -> needed for average calculations
                for d in range(day, int(t) + 1):
                    days.append(d)
                    day_i.append(i_prev)
                day = int(t) + 1
            # Check whether we changed from one day to another
            elif t_prev <= day < t:
                days.append(day)
                day = int(t) + 1
                day_i.append(i_prev)
        # Taking initial data
        else:
            day_i.append(y)
            days.append(t)
        # Update previously stored data
        t_prev = t
        i_prev = y
    return days, day_i


def main():
    import json

    graph_dict = json.load(open("graph_dict.json"))

    sim = Simulator(graph_dict=graph_dict)
    sim.run_after_peak()


if __name__ == '__main__':
    main()
