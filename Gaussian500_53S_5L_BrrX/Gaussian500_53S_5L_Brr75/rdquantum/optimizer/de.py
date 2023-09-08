import os
import glob
import time

from multiprocessing.connection import wait
from IPython.display import clear_output
from rdquantum.fidelity import fidelity
import numpy as np

from qutip import *

class de:
    def __init__(self, fidelity, Pulses, PulsesRange, mu=0.5, xi=0.9, c=15):
        self.fidelity = fidelity        # Fidelity object
        self.mu = mu                    # Mutation factor
        self.xi = xi                    # Crossover rate
        
        self.c = c
        self.K = 0
        for key in Pulses:
            self.K += len(Pulses[key])  # Number of control parameters
        self.Np = self.c * self.K       # Population size
        self.populations = []
        self.Pulses = Pulses
        self.valuerange = PulsesRange

        # Initialize data
        self.data_fidelity = []
        self.data_pulses = []
        self.op_fidelity = 0
        self.op_pulse = []

    def createPopulations(self):
        for i in range(self.Np):
            population = {}
            for key in self.Pulses:
                n = len(self.Pulses[key])
                population[key] = np.array(np.random.uniform(self.valuerange[key][0], self.valuerange[key][1], n))
            self.populations.append(population)

    def sample(self, n):
        # Randomly sample n populations for one differential evolution iteration
        choosed = []
        # np.random.seed()
        for i in range(n):
            s = int(self.Np * np.random.uniform(0,1))
            while s in choosed:
                s = int(self.Np * np.random.uniform(0,1))
            choosed.append(s)

        return np.array(choosed)
    
    def select(self, di, Ci, Di, fci, fdi):
        if fci > fdi:
            update = True
            return Ci, fci, update
        else:
            update = False
            return Di, fdi, update
    
    def iteration(self, batch=1):
        Ci = {}    # Target vector
        batch_Di = []
        batch_Mi = []
        batch_Ci = []
        batch_di = []
        
        # 1) Mutation
        # Mutated population should in the required limit (PulseRange).
        for i in range(batch):
            Di = {}    # Original vector
            Mi = {}    # Trail vector
            validmutation = False
            while validmutation != True:
                di = self.sample(4)             # Sampled population member
                Di = self.populations[di[0]]    # Original vector
                Di1 = self.populations[di[1]]
                Di2 = self.populations[di[2]]
                Di3 = self.populations[di[3]]
                for key in Di:
                    Mi[key] = np.array(Di1[key]) + self.mu*(np.array(Di2[key]) - np.array(Di3[key]))
                    # Check the validation of mutation
                    invalid = [temp for temp in Mi[key]
                    if temp < self.valuerange[key][0] or temp > self.valuerange[key][1]]
                    if len(invalid) != 0:
                        validmutation = False
                        break
                    else:
                        validmutation = True
            batch_Di.append(Di)
            batch_Mi.append(Mi)
            batch_di.append(di)

        # 2) Corssover
        for i in range(batch):
            Ci = {}     # Target vector
            Mi = batch_Mi[i]
            Di = batch_Di[i]
            for key in Mi:
                Ci[key] = []
                for j in range(len(Mi[key])):
                    r = np.random.uniform(0,1)
                    if r < self.xi:
                        Ci[key].append(Mi[key][j])
                    else:
                        Ci[key].append(Di[key][j])
            batch_Ci.append(Ci)

        # 3) Selection
        batch_fci = parfor(self.fidelity.get_fidelity, batch_Ci)
        batch_fdi = parfor(self.fidelity.get_fidelity, batch_Di)
            
        batch_pulses, batch_fidelity, batch_update = parfor(self.select, batch_di, batch_Ci, batch_Di, batch_fci, batch_fdi)
        return batch_pulses, batch_fidelity, batch_di, batch_update

    def start(self, itr = 1000, batch = 10, threshold=0.999):
        # itr: iteration times
        # Check if self.populations is empty
        if len(self.populations) == 0:
            print("Please create populations, de.createPopulations(Pulses, PulsesRange).")
        else:
            i = 0
            while i < itr:
                clear_output(wait=True)
                print("Start differential evolution...")
                print("Number of control parameters: %s" %self.K)
                print("Populations size: %s" %len(self.populations))
                print("==============================")
                print('# %s/%s iteration.' %(i, itr))
                print("Optimized fidelity: %s" %self.op_fidelity)
                print("")
                batch_pulses, batch_fidelity, batch_di, batch_update = self.iteration(batch)
                for j in range(len(batch_fidelity)):
                    di = batch_di[j]
                    update = batch_update[j]
                    if batch_fidelity[j] > threshold or batch_fidelity[j] > self.op_fidelity:
                        self.data_fidelity.append(batch_fidelity[j])
                        self.data_pulses.append(batch_pulses[j])
                    if update == True:
                        self.populations[di[0]] = batch_pulses[j]
                        print("Update population %s" %di[0])
                    # else:
                    #     print("Population %s not changed" %di[0])
                    
                    if batch_fidelity[j] > self.op_fidelity:
                        self.op_fidelity = batch_fidelity[j]
                        self.op_pulse = batch_pulses[j]

                np.savez("out.npz", populations=self.populations
                         , fidelity=self.data_fidelity, pulses=self.data_pulses
                         , op_fidelity=self.op_fidelity)
                np.savez("out-op_pulse.npz", **self.op_pulse)
                
                # Remove qobjevo* cache
                cachefile = glob.glob("qobjevo*")
                for filename in cachefile:
                    os.remove(filename)

                i = i + batch
                print('')
                print('==============================')            
