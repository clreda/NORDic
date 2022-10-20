# coding:utf-8

import imports
import os
import numpy as np
import pandas as pd
from subprocess import call as sbcall
from glob import glob

from NORDic.UTILS.utils_plot import influences2graph
from NORDic.NORDic_PMR.functions import greedy

## Test: we expect X0 and X1 to be outputed, X10 to have no influence on the network because it is isolated
source = list(map(lambda x: "X%d" % x,[0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,3,4,5]))#,0]))
target = list(map(lambda x : "X%d" % x,[2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,6,7,8,9]))#,11]))
genes = list(sorted(list(set(source+target))))+["X10"]

save_folder="minimal/"
sbcall("mkdir -p "+save_folder, shell=True)

## 1. Plot network
influences = np.zeros((len(genes), len(genes)))
for s, t in zip(source, target):
    influences[genes.index(s)][genes.index(t)] = 1
influences = pd.DataFrame(influences, index=genes, columns=genes)
influences2graph(influences, save_folder+"example", optional=False, compile2png=True, engine="sfdp")

## 2. Build network
grfs = {}
for si, s in enumerate(source):
    t = target[si]
    grf = grfs.get(t, [])
    grfs.update(dict([[t, list(set(grf+[s]))]]))
for g in genes:
    if (g not in grfs):
        #grfs.update(dict([[g,[g]]]))
        pass
with open(save_folder+"example.bnet", "w") as f:
    network = []
    for g in genes:
        if (g in grfs):
            network += [g+", "+"&".join(grfs[g])]
        else:
            network += [g+", 1"]
    f.write("\n".join(network))
with open(save_folder+"example.bnet", "r") as f:
    network = f.read()
gene_outputs = [x.split(", ")[0] for x in network.split("\n")[:-1] if (x.split(", ")[1] not in [x.split(", ")[0], "0", "1"])]

## 3. Sample states at random
state_len = 100
states = pd.DataFrame([np.random.choice([0,1], p=[0.5,0.5], size=len(genes)).tolist() for _ in range(state_len)], columns=genes, index=range(state_len)).T
print(states)

seed_number=0
k=2

from multiprocessing import cpu_count
njobs=max(1,cpu_count()-2)

IM_params = {
    "seed": seed_number,
    "njobs": min(5, njobs),
    "gene_inputs": genes, # genes to be perturbed
    "gene_outputs": gene_outputs # genes to be observed
}
SIMU_params = {
    'nb_sims': 100,
    'rates': "fully_asynchronous",
    'thread_count': njobs,
    'depth': "constant_unitary",
}

#########################################
## Test for master regulator detection ##
#########################################

S, spreads = greedy(save_folder+"example.bnet", k, states, IM_params, SIMU_params, save_folder=save_folder)
print("\n\n* ANSWER")
print(S)
print(spreads)
S = [x for s in S for x in s]
assert all([s in ["X0","X1"] for s in S])
