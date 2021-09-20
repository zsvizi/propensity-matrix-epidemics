import os

import numpy as np

from base_simulator import BaseSimulator


class Simulator(BaseSimulator):
    def __init__(self, graph_dict):
        super().__init__(graph_dict=graph_dict)

    def run(self):
        # Reset age, c_from, c_to
        age, c_from, c_to = None, None, None

        # initialize variable for halting condition
        i = self.state_var[-1][self.c_idx["I"]]
        e = self.state_var[-1][self.c_idx["E"]]
        actual_time = self.time_series[-1]

        # Halting condition: (i = 0) <- used in while cycle
        while np.sum(i + e) > 0 and actual_time < 365:
            # Run simulation
            age, c_from, c_to = self.simulate(age=age, c_from=c_from, c_to=c_to)

            # Calculation for halting condition
            i = self.state_var[-1][self.c_idx["I"]]
            e = self.state_var[-1][self.c_idx["E"]]
            actual_time = self.time_series[-1]

        self.state_var = np.array(self.state_var)

    def run_after_csocs(self, days: int = 10):
        # Reset age, c_from, c_to
        age, c_from, c_to = None, None, None

        # initialize variable for halting condition
        i = self.state_var[-1][self.c_idx["I"]]
        e = self.state_var[-1][self.c_idx["E"]]
        actual_time = self.time_series[-1]

        # Halting condition: (i = 0) <- used in while cycle
        halting_condition = False
        day = 1
        day_i = [np.sum(i)]
        while np.sum(i + e) > 0 and actual_time < 365 and not halting_condition:
            # Run simulation
            age, c_from, c_to = self.simulate(age=age, c_from=c_from, c_to=c_to)

            # Calculation for halting condition
            i = self.state_var[-1][self.c_idx["I"]]
            e = self.state_var[-1][self.c_idx["E"]]

            if len(self.time_series) >= 2:
                actual_time = self.time_series[-1]
                prev_time = self.time_series[-2]
                if prev_time <= day < actual_time:
                    day += 1
                    aggregated = np.sum(self.state_var, axis=2)
                    i_values = aggregated[:, self.c_idx["I"]]
                    day_i.append(i_values[-2])

                i_max_position = np.argmax(day_i)
                i_max = day_i[i_max_position]

                if len(day_i) >= i_max_position+days:
                    halting_condition = np.all(day_i[i_max_position:i_max_position+days] <= i_max)

        self.state_var = np.array(self.state_var)

    def run_conf(self):
        # Reset age, c_from, c_to
        age, c_from, c_to = None, None, None

        # Initialize variable for halting condition
        i = self.state_var[-1][self.c_idx["I"]]
        e = self.state_var[-1][self.c_idx["E"]]
        actual_time = self.time_series[-1]

        # Create data directory, if not exists
        os.makedirs("./data", exist_ok=True)
        # Number for minimal number of simulations
        min_n_sim = 100
        # Threshold for convergence
        std_threshold = 2.0
        conf_threshold = 0.95
        # Initialize simulation counter
        sim_cnt = 0
        # Create list for peak sizes
        peak_sizes = []
        # Absolute change
        is_converged = True

        while sim_cnt < min_n_sim or not is_converged:
            # Halting condition: (i = 0) <- used in while cycle
            while actual_time < 100 and np.sum(i + e) > 0:
                # Run simulation
                age, c_from, c_to = self.simulate(age=age, c_from=c_from, c_to=c_to)

                # Calculation for halting condition
                i = self.state_var[-1][self.c_idx["I"]]
                e = self.state_var[-1][self.c_idx["E"]]
                actual_time = self.time_series[-1]
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
            i = self.state_var[-1][self.c_idx["I"]]
            e = self.state_var[-1][self.c_idx["E"]]
            actual_time = self.time_series[-1]

            # Calculation for the outer halting condition
            if sim_cnt >= min_n_sim:
                peak_sizes_np = np.array(peak_sizes)
                m = np.mean(peak_sizes_np)
                s = np.std(peak_sizes_np)
                peak_sizes_standard = (peak_sizes_np - m) / s
                abs_peak_sizes = np.abs(peak_sizes_standard[-min_n_sim:])
                n_peaks_inside_std = np.sum(abs_peak_sizes < std_threshold)
                is_converged = n_peaks_inside_std > (conf_threshold * min_n_sim)
                print("Simulation #" + str(sim_cnt) + ":", n_peaks_inside_std)
            else:
                if sim_cnt % 10 == 0:
                    print("Simulation #" + str(sim_cnt))
        return sim_cnt

    def reset_for_new_simulation(self, init=None):
        if init is None:
            init = np.copy(self.init_vals)
        self.init_vals = init
        self.last_state = init
        self.time_series = [0.0]
        self.state_var = [init]
        self.create_transition_matrix()
        self.create_transmission_matrix()


def generate_daily_data(time_series, values):
    # Initialization
    day = 1  # day to check ahead of actual time
    days, day_i = [], []
    t_prev, i_prev = None, None
    # Loop through all time steps taking the aggregated infections
    for t, y in zip(time_series, values):
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
    sim.run_after_csocs()


if __name__ == '__main__':
    main()
