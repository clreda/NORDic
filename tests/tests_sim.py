#coding: utf-8

import NORDic
import pandas as pd
import numpy as np
import os
from multiprocessing import cpu_count
from time import time

taxon_id = 9606
njobs=max(1,cpu_count()-2)
seednb=12345
file_folder="ToyOndine/"

network_fname=file_folder+"solution.bnet"

from IPython.display import Image
Image(filename=file_folder+'inferred_maximal_solution.png') 

import mpbn

net = mpbn.MPBooleanNetwork(mpbn.load(network_fname))
x0 = net.zero() # consider initial state where all gene expression levels are set to 0...
for g in ["ASCL1", "BDNF", "RET"]: #... except for ASCL1 and BDNF and RET
    x0[g] = 1

from NORDic.UTILS.utils_sim import MABOSS_SIM

initial = pd.DataFrame(pd.Series(x0), columns=["initial"])
params = {'sample_count': 10000, 'max_time': 20}
nee = MABOSS_SIM(seednb,njobs)
nee.update_network(network_fname, initial, verbose=False)
probs = nee.generate_trajectories(params=params, outputs=[]) ## generate trajectories
assert all([c=="{<nil>=1}" for c in probs.columns])
assert np.isclose(probs.loc["prob"]["{<nil>=1}"],1)

from NORDic.UTILS.utils_sim import MPBN_SIM

initial = pd.DataFrame(pd.Series(x0), columns=["initial"])
params = {'sample_count': 10000, 'max_time': 20}
nee = MPBN_SIM(seednb,njobs)
nee.update_network(network_fname, initial, verbose=False) ## no mutations, initial state x0
probs = nee.generate_trajectories(params=params, outputs=[]) ## generate trajectories
assert all([c=="{<nil>=1}" for c in probs.columns])
assert np.isclose(probs.loc["prob"]["{<nil>=1}"],1)

from NORDic.UTILS.utils_sim import BONESIS_SIM

initial = pd.DataFrame(pd.Series(x0), columns=["initial"])
final = pd.DataFrame([], index=[g for g in x0], columns=["final"]).fillna(0) 
nee = BONESIS_SIM(seednb,njobs)
nee.update_network(network_fname, initial, final, verbose=False) ## no mutations, initial state x0
attrs = nee.enumerate_attractors(verbose=False) 
assert "StateNotFound" not in attrs.columns[0]

final = pd.DataFrame([[1,0,1,1,1,1]], columns=[g for g in x0], index=["final"]).T 
nee = BONESIS_SIM(seednb,njobs)
nee.update_network(network_fname, initial, final, verbose=False) ## no mutations, initial state x0
attrs = nee.enumerate_attractors(verbose=False) 
assert "StateNotFound" in attrs.columns[0]

initial = pd.DataFrame([[1,0,1,0,1,0]], columns=[g for g in x0], index=["initial"]).T 
nee = MABOSS_SIM(seednb,njobs)
nee.update_network(network_fname, initial, verbose=False)
nee.enumerate_attractors() ## enumerate all attractors
assert nee.attrs.shape[1]==2
assert all(["".join(list(nee.attrs.astype(str)[c])) in ["1"*len(nee.gene_list), "0"*len(nee.gene_list)] for c in nee.attrs])

initial = pd.DataFrame([[1,0,1,0,1,0]], columns=[g for g in x0], index=["initial"]).T 
nee = MPBN_SIM(seednb,njobs)
nee.update_network(network_fname, initial, verbose=False)
nee.enumerate_attractors() ## enumerate all attractors
assert nee.attrs.shape[1]==2
assert all(["".join(list(nee.attrs.astype(str)[c])) in ["1"*len(nee.gene_list), "0"*len(nee.gene_list)] for c in nee.attrs])

C = pd.DataFrame(pd.Series(x0), columns=["initial"])
T = pd.DataFrame([[1,0,1,1,1,1]], columns=[g for g in x0], index=["NotReachable"]).T 
T = T.join(pd.DataFrame([], index=[g for g in x0], columns=["Reachable"]).fillna(0), how="outer")

from NORDic.UTILS.utils_sim import test

_, net_MABOSS_noMut = test(MABOSS_SIM,seednb,njobs,network_fname,C,T,"profiles",verbose=0)
res = pd.DataFrame({"score": net_MABOSS_noMut.max_values}).T
assert np.isclose(res.loc["score"]["initial->Reachable"],1)
assert res.loc["score"]["initial->NotReachable"]<1

_, net_MABOSS_noMut = test(MABOSS_SIM,seednb,njobs,network_fname,C,T,"attractors",verbose=0)
res = pd.DataFrame({"score": net_MABOSS_noMut.max_values}).T
assert np.isclose(res.loc["score"]["initial->Reachable"],1)
assert np.isclose(res.loc["score"]["initial->NotReachable"],0)

_, net_MPBN_noMut = test(MPBN_SIM,seednb,njobs,network_fname,C,T,"profiles",verbose=0)
res = pd.DataFrame({"score": net_MPBN_noMut.max_values}).T
assert np.isclose(res.loc["score"]["initial->Reachable"],1)
assert res.loc["score"]["initial->NotReachable"]<1

_, net_MPBN_noMut = test(MPBN_SIM,seednb,njobs,network_fname,C,T,"attractors",verbose=0)
res = pd.DataFrame({"score": net_MPBN_noMut.max_values}).T
assert np.isclose(res.loc["score"]["initial->Reachable"],1)
assert np.isclose(res.loc["score"]["initial->NotReachable"],0)

_, net_BONESIS_noMut = test(BONESIS_SIM,seednb,njobs,network_fname,C,T,"profiles",verbose=0)
res = pd.DataFrame({"score": net_BONESIS_noMut.max_values}).T
assert np.isclose(res.loc["score"]["initial->Reachable"],1)
assert np.isclose(res.loc["score"]["initial->NotReachable"],0)

_, net_BONESIS_noMut = test(BONESIS_SIM,seednb,njobs,network_fname,C,T,"attractors",verbose=0)
res = pd.DataFrame({"score": net_BONESIS_noMut.max_values}).T
assert np.isclose(res.loc["score"]["initial->Reachable"],1)
assert np.isclose(res.loc["score"]["initial->NotReachable"],0)

nee = MPBN_SIM(seednb,njobs)
nee.update_network(network_fname, initial, mutation_permanent={'PHOX2B': 0}, verbose=False)
assert len(nee.all_mutants)==1 and 'PHOX2B' in nee.all_mutants

nee = MPBN_SIM(seednb,njobs)
nee.update_network(network_fname, initial, mutation_permanent={'PHOX2B': 1}, verbose=False)
assert len(nee.all_mutants)==1 and 'PHOX2B' in nee.all_mutants

nee = MPBN_SIM(seednb,njobs)
nee.update_network(network_fname, initial, mutation_transient={'PHOX2B': 0}, verbose=False)
assert len(nee.all_mutants)==1 and 'PHOX2B' in nee.all_mutants

nee = MPBN_SIM(seednb,njobs)
nee.update_network(network_fname, initial, mutation_transient={'PHOX2B': 1}, verbose=False)
assert len(nee.all_mutants)==1 and 'PHOX2B' in nee.all_mutants

_, net_MPBN_noMut = test(MPBN_SIM,seednb,njobs,network_fname,C,T,"profiles", mutation_permanent={'PHOX2B': 0},verbose=0)
res = pd.DataFrame({"score": net_MPBN_noMut.max_values}).T
assert np.isclose(res.loc["score"]["initial->Reachable"],1)
assert res.loc["score"]["initial->NotReachable"]<1

_, net_MPBN_noMut = test(MPBN_SIM,seednb,njobs,network_fname,C,T,"profiles", mutation_transient={'EDN3': 0},verbose=0)
res = pd.DataFrame({"score": net_MPBN_noMut.max_values}).T
assert np.isclose(res.loc["score"]["initial->Reachable"],1)
assert res.loc["score"]["initial->NotReachable"]<1