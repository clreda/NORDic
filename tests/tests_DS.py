#coding: utf-8

import sys, os
testdir = os.path.dirname(__file__)
srcdir = '../src/'
sys.path.insert(0, os.path.abspath(os.path.join(testdir, srcdir)))

import imports
import pandas as pd
from multiprocessing import cpu_count
import numpy as np

## III. Simulation of drug effect based on a Boolean network (**NORDic DS**)

file_folder="ToyOndine/"
seed_number=12345

np.random.seed(seed_number)
with open(file_folder+"solution.bnet", "r") as f:
    genes = [line.split(", ")[0] for line in f.read().split("\n") if (len(line)>0)]

## Create a set of random "patient" and "control" states (see tests_PMR.py)
states = pd.read_csv(file_folder+"phenotypes.csv", index_col=0)
### Patient profiles
patients = states[[c for c in states.columns if ("Case" in c)]].loc[[i for i in states.index if (i != "annotation")]]

## Generate a set of drugs
targets = {"KO_"+g: {g: -1} for g in genes}
targets.update({"OE_"+g: {g: 1} for g in genes})
targets = pd.DataFrame(targets)
all_genes = pd.DataFrame({"One": {g: 0 for g in genes}})
targets = targets.join(all_genes, how="outer").fillna(0)[[c for c in targets.columns if (c!="One")]].astype(int)
targets.to_csv(file_folder+"targets.csv")

## Drug Scoring
### Define the function which determines whether a state is closer to controls/patients
dfdata = states.loc[[i for i in states.index if (i != "annotation")]]
samples = states.loc["annotation"]

from NORDic.NORDic_DS.functions import compute_frontier
## compute_frontier fits a SVM model to the data
frontier = compute_frontier(dfdata, samples)
score = lambda attrs : (frontier.predict(attrs.values.T)==1).astype(int)

from NORDic.NORDic_DS.functions import simulate
scores = simulate(file_folder+"solution.bnet", targets, patients, score, simu_params=SIMU_params, nbseed=seed_number)

assert scores.shape[0]==patients.shape[1]
assert scores.shape[1]==targets.shape[1]