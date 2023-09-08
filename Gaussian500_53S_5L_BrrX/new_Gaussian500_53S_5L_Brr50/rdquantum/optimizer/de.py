import os
import glob
import time

from multiprocessing.connection import wait
from rdquantum.fidelity import fidelity
import numpy as np

from qutip import *

class DE:
    def __init__(self, fidelity, pulses, pulses_range, mu=0.5, xi=0.9, c=15):
        # DE parameters
        self.fidelity = fidelity 
        self.mu = mu    # Mutation factor
        self.xi = xi    # Crossover rate
        self.c = c
        self.k = 0  # Number of control parameters
        for key in pulses:
            self.k += len(pulses[key])  
        self.number_of_populations = self.c * self.k
        # Pulse parameters
        self.populations = []
        self.pulses = pulses
        self.value_range = pulses_range
        # Log data 
        self.log_fidelity = [[]] * self.number_of_populations
        self.log_pulses = [[]] * self.number_of_populations
        self.log_op_fidelity = 0
        self.log_op_pulse = []

    def create_populations(self):
        for i in range(self.number_of_populations):
            population = {}
            for key in self.pulses:
                n_control_parameters = len(self.pulses[key])
                population[key] = np.array(np.random.uniform(self.value_range[key][0], 
                                                             self.value_range[key][1],
                                                             n_control_parameters))
            self.populations.append(population)

    def run(self, itr=60):
        # itr: iteration times
        if len(self.populations) == 0:
            """ To do

            Using raise Error
            """
            print("Please create populations, DE.create_populations(pulses, pulses_range).")
        else:
            i = 0
            while i < itr:
                print("Start differential evolution...")
                print("Number of control parameters: %s" %self.k)
                print("Populations size: %s" %len(self.populations))
                print("==============================")
                print('# %s/%s iteration.' %(i, itr))
                print("Optimized fidelity: %s" %self.log_op_fidelity)
                print("")
                update, update_vector, update_fidelity = parfor(self._iteration, range(self.number_of_populations))
                for j in range(len(update)):
                    self.log_fidelity[j].append(update_fidelity[j])
                    self.log_pulses[j].append(update_vector[j])
                    if update[j]:
                        print("Update population %s" %j)
                        if update_fidelity[j] > self.log_op_fidelity:
                            self.log_op_fidelity = update_fidelity[j]
                            self.log_op_pulse = update_vector[j]
                np.savez("out.npz", 
                         populations=self.populations,
                         fidelity=self.log_fidelity, 
                         pulses=self.log_pulses,
                         op_fidelity=self.log_op_fidelity)
                np.savez("out-op_pulse.npz", **self.log_op_pulse)
                # Remove qobjevo* cache
                cachefile = glob.glob("qobjevo*")
                for filename in cachefile:
                    os.remove(filename)
                i = i + self.number_of_populations
                print('')
                print('==============================')            

    def _sample(self, n, not_include=[]):
        # Randomly sample n populations for one differential evolution iteration
        choosed = []
        np.random.seed()
        for i in range(n):
            s = int(self.number_of_populations * np.random.uniform(0,1))
            while (s in choosed) or (s in not_include):
                s = int(self.number_of_populations * np.random.uniform(0,1))
            choosed.append(s)
        return np.array(choosed)
    
    def _iteration(self, di_index: int):
        """ DE iteration

        di_index: Index of the origianl vector.
        """
        mi_vector = self._mutation(di_index)
        ci_vector = self._crossover(di_index, mi_vector)
        update, update_vector, update_fidelity = self._selection(di_index, ci_vector)
        if update:
            self.populations[di_index] = ci_vector
        return update, update_vector, update_fidelity

    def _mutation(self, di_index: int):
        """ DE mutation

        """
        mi_vector = {}  # Trail vector
        valid_mutation = False
        while not valid_mutation:
            dix_index = self._sample(n=3, not_include=[di_index])
            di1_vector = self.populations[dix_index[0]]
            di2_vector = self.populations[dix_index[1]]
            di3_vector = self.populations[dix_index[2]]
            for key in di1_vector:
                mi_vector[key] = np.array(di1_vector[key]) + self.mu*(np.array(di2_vector[key])-np.array(di3_vector[key]))
                # Check the validation of mi_vector
                invalid = [temp for temp in mi_vector[key] 
                           if temp < self.value_range[key][0] 
                           or temp > self.value_range[key][1]
                           ]
                if len(invalid) != 0:
                    valid_mutation = False
                else:
                    valid_mutation = True
        return mi_vector
    
    def _crossover(self, di_index: int, mi_vector: dict):
        """ DE crossover

        """
        di_vector = self.populations[di_index]
        ci_vector = {}  # Target vector
        for key in di_vector:
            ci_vector[key] = []
            for i in range(len(di_vector[key])):
                r = np.random.uniform(0, 1)
                if r < self.xi:
                    ci_vector[key].append(mi_vector[key][i])
                else:
                    ci_vector[key].append(di_vector[key][i])
        return ci_vector

    def _selection(self, di_index, ci_vector: dict):
        """ DE selection

        """
        di_vector = self.populations[di_index]
        di_fidelity = self.fidelity.get_fidelity(di_vector)
        ci_fidelity = self.fidelity.get_fidelity(ci_vector)
        update = False
        update_vector = None
        update_fidelity = None
        if di_fidelity < ci_fidelity:
            update = True
            update_vector = ci_vector
            update_fidelity = ci_fidelity
        else:
            update = False
        return update, update_vector, update_fidelity
