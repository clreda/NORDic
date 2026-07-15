#coding:utf-8

import pandas as pd
import numpy as np
import subprocess as sb
from tqdm import tqdm

from iterative_network_identification import iterative_network_identification

#############################
## Initialize parameters   ##
#############################
seednb=123456
njobs=1
file_folder="./small_synthetic/"

#############################
## Prior Knowledge Network ##
#############################
## 1. Build the Prior Knowledge Network: identify interactions to keep and those which should be sampled
## Fixed interactions
solution_fixed = pd.DataFrame([], index=["A", "B", "C", "D"], columns=[1])
solution_fixed.loc["A"] = "!D" 
solution_fixed.loc["B"] = "A"
solution_fixed.loc["C"] = "!B"
solution_fixed.loc["D"] = "0"

## Test the current dynamics of the solution
import NORDic
import mpbn
from NORDic.UTILS.utils_sim import BONESIS_SIM
net_sim = BONESIS_SIM(seednb, njobs)
network_fname = "network_initial.bnet"
solution_fixed_format = solution_fixed.copy()
solution_fixed_format[1] = " "+solution_fixed_format[1]
solution_fixed_format.to_csv(network_fname, header=None, sep=",")
net_bn = mpbn.MPBooleanNetwork(mpbn.load(network_fname))
all_gene_list = list(solution_fixed.index)
initial = pd.DataFrame([], columns=["initial"], index=all_gene_list)
final_A0 = pd.DataFrame([], columns=["final_A0"], index=all_gene_list)
final_A0.loc['A'] = 0
final_A0.loc['B'] = 0
final_A0.loc['C'] = 1
final_A0.loc['D'] = 1
final_B1 = pd.DataFrame([], columns=["final_B1"], index=all_gene_list)
final_B1.loc['B'] = 1
final_B1.loc['C'] = 0
final_C0 = pd.DataFrame([], columns=["final_C0"], index=all_gene_list)
final_C0.loc['A'] = 0
final_C0.loc['B'] = 0
final_C0.loc['C'] = 0
final_C0.loc['D'] = 1
final_D0 = pd.DataFrame([], columns=["final_D0"], index=all_gene_list)
final_D0.loc['A'] = 1
final_D0.loc['B'] = 1
final_D0.loc['C'] = 0
final_D0.loc['D'] = 0
final = [final_A0, final_B1, final_C0, final_D0]
verbose = False
all_valid = [False]*4
for ip, pert in enumerate([{"A":0}, {"B":1}, {"C":0}, {"D":0}]):
	net_sim.update_network(network_fname, initial, final[ip], mutation_permanent=pert, verbose=verbose) 
	attrs = net_sim.enumerate_attractors(verbose=verbose)
	is_valid = (~pd.isnull(attrs).values).all()
	all_valid[ip] = bool(is_valid)
assert all(all_valid)

## Putative interactions
net_putative = pd.DataFrame([], index=["regulated","regulator", "sign", "score"]).T
net_putative.loc[0] = ["C","D",1,1]
net_putative.loc[1] = ["A","C",-1,1]
net_putative.loc[2] = ["D","B",-1,1]
net_putative.loc[3] = ["B","D",-1,1]

all_gene_list = list(sorted(list(set(list(solution_fixed.index)+list(net_putative["regulated"])+list(net_putative["regulator"])))))

#############################
## Dynamical constraints   ##
#############################
## 2. Define the Dynamical Constraints: at least the initial state + final (attractor) state for each experiment
## Those are dummy experiments, to be replaced
## arguments: perturbation and initial state and final (attractor) state for each experiment
## returns: dictionary of experiments with perturbation, initial state, and final attractor state
experiments = {}

initial = pd.DataFrame([], columns=["initial"], index=all_gene_list)

pert1 = {"A": 1}
final1 = pd.DataFrame([], columns=["final"], index=all_gene_list)
final1.loc['B'] = 1
final1.loc['C'] = 0
final1.loc['D'] = 0
experiments.update({"exp1": (pert1, initial, final1)})

pert2 = {"C": 0}
final2 = pd.DataFrame([], columns=["final"], index=all_gene_list)
#final2.loc['A'] = 1
#final2.loc['B'] = 1
final2.loc['D'] = 1
experiments.update({"exp2": (pert2, initial, final2)})

pert3 = {"D": 1}
final3 = pd.DataFrame([], columns=["final"], index=all_gene_list)
final3.loc['A'] = 0
final3.loc['B'] = 0
final3.loc['C'] = 1
experiments.update({"exp3": (pert3, initial, final3)})

pert4 = {"B": 1}
final4 = pd.DataFrame([], columns=["final"], index=all_gene_list)
final4.loc['A'] = 1
final4.loc['D'] = 0
final4.loc['C'] = 0
experiments.update({"exp4": (pert4, initial, final4)})

#############################
## Call iterative function ##
#############################
net = iterative_network_identification(seednb, njobs, solution_fixed, net_putative, experiments, limit=3, max_iter=100)
