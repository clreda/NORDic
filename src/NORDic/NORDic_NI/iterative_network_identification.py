#coding:utf-8

import subprocess as sb
import numpy as np
from tqdm import tqdm
import mpbn
import pandas as pd

import NORDic
from NORDic.UTILS.utils_sim import BONESIS_SIM

def sampling_regulatory_functions(solution_fixed, net_changing, network_fname, select_reg, merge_reg):
	if (net_changing.shape[0]==0):
		solution_copy = solution_fixed.copy()
		solution_copy[solution_copy.columns[0]] = " "+solution_copy[solution_copy.columns[0]]
		solution_copy.to_csv(network_fname, header=None, sep=",")
		return network_fname
	solution_changing = net_changing.copy()
	solution_changing["regulator"] = [("!"+x) if (solution_changing.iloc[i,:]["sign"]==-1) else x for i, x in enumerate(solution_changing["regulator"])]
	solution_changing.index = solution_changing["regulated"]
	solution_changing = solution_changing[["regulator"]]
	solution_changing.columns = [1]
	solution_changing[solution_changing.columns[0]] += ","
	grfs = solution_changing.groupby(level=0).sum().to_dict()[solution_changing.columns[0]]
	grfs = {k:[x for x in grfs[k].split(",") if (len(x)>0)] for k in grfs}
	solution_merged = solution_fixed.copy()
	## Select choice : all conditions for cell-to-cell communications
	grfs = {k:select_reg(grfs[k]) for k in grfs}
	for gene in grfs:
		if (str(solution_merged.loc[gene][solution_merged.columns[0]]) in ["0","1"]):
			solution_merged.loc[gene] = grfs[gene]
		else:
			## Merge choice : trigger either the intercell and/or cell-to-cell regulations
			solution_merged.loc[gene] = merge_reg("".join(solution_fixed.loc[gene][solution_fixed.columns[0]].split(" ")), grfs[gene])
	solution_merged[solution_merged.columns[0]] = " "+solution_merged[solution_merged.columns[0]]
	solution_merged.to_csv(network_fname, header=None, sep=",")
	return network_fname

## TODO sampling strategy: start from all interactions and remove greedily = strategy to identify most relevant edges?
def sampling_interactions(net_putative, keep_interactions=None, seed=12324):
	if (keep_interactions is None):
		## First call to sampling interactions: select all edges
		keep_interactions = np.ones(net_putative.shape[0]).astype(bool)
	else: ## random selection TODO guided via the fitness score/constraint check on pairs of genes?
		np.random.seed(seed)
		keep_interactions = np.random.choice([True, False], size=net_putative.shape[0], replace=True, p=[0.5,0.5])
	return net_putative.iloc[keep_interactions], keep_interactions

## Enumerate all possible subsets of interactions
def gray_code(n):
	return ["{0:0{1}b}".format(i^(i>>1),n) for i in range(0, 1<<n)]

## How the Bonesis-based simulator works
## https://github.com/clreda/NORDic/blob/main/notebooks/NORDic%20Network%20Simulations.ipynb
def iterative_network_identification(seednb, njobs, solution_fixed, net_putative, experiments={}, limit=1, verbose=False, max_iter=1000):
	## 3. Loop over subsets of interactions until all experiments can be satisfied 
	## with at least one set of interactions and until the required number of solutions is found
	net_sim = BONESIS_SIM(seednb, njobs)
	## Or use template regulations in https://www.nature.com/articles/npjsba201610
	select_reg = lambda reg_list : "&".join(reg_list)
	merge_reg = lambda inter, intra : f"({inter})|({intra})"
	solutions = [None]*limit
	keep_interactions = None
	seeds = np.random.choice(range(int(max(1e8, max_iter*2))), size=max_iter, p=None, replace=False)
	i_sol = 0
	iter_sol = 0
	while(iter_sol<max_iter and i_sol<limit): ## find several solutions
		exp_valid = [False]*len(experiments)
		constraints_check = {}
		## 4. Sampling + backtracking/optimisation strategy for putative interactions
		## We could rewrite inside Bonesis, but that might cause compatibility issues
		network_fname = f"network{iter_sol}.bnet"
		net_changing, keep_interactions = sampling_interactions(net_putative, keep_interactions, seed=seeds[iter_sol])
		new_solution = sampling_regulatory_functions(solution_fixed, net_changing, network_fname, select_reg, merge_reg)
		net_bn = mpbn.MPBooleanNetwork(mpbn.load(network_fname))
		## Test the network
		for i_exp, exp in enumerate(pbar := tqdm(experiments)):
			pert, initial, final = experiments[exp]
			pbar.set_description(f"Experiment {exp} with perturbation {pert}")
			net_sim.update_network(network_fname, initial, final, mutation_permanent=pert, verbose=verbose) 
			attrs = net_sim.enumerate_attractors(verbose=verbose)
			is_valid = (~pd.isnull(attrs).values).all()
			exp_valid[i_exp] = is_valid
			if (not is_valid):
				## Check the unsatisfied constraints on the final state
				for idx_final in (pbar2 := tqdm(final.index)):
					for idx_final2 in final.index:
						if (not np.isnan(final.loc[idx_final]["final"]) and not np.isnan(final.loc[idx_final2]["final"])):
							pbar2.set_description(f"Checking constraint on {list(set([idx_final,idx_final2]))} in {exp}")
							final_exp = pd.DataFrame([], index=final.index, columns=[f"final_{idx_final}"])
							final_exp.loc[idx_final] = final.loc[idx_final]["final"]
							final_exp.loc[idx_final2] = final.loc[idx_final2]["final"]
							pert_lst = ",".join(list(set([f"{idx_final}={final.loc[idx_final]['final']}", f"{idx_final2}={final.loc[idx_final2]['final']}"])))
							net_sim.update_network(network_fname, initial, final_exp, mutation_permanent=pert, verbose=verbose)
							attrs = net_sim.enumerate_attractors(verbose=verbose)
							is_valid = (~pd.isnull(attrs).values).all()
							constraints_check.update({pert_lst: int(is_valid)})
				print("")
				print((network_fname, i_sol, iter_sol, keep_interactions, exp, constraints_check))
				print("")
		print("")
		print((network_fname, i_sol, iter_sol, keep_interactions, exp_valid))
		print("")
		iter_sol += 1
		if (all(exp_valid)):
			solutions[i_sol] = network_fname
			i_sol += 1
		else:
			proc = sb.Popen(f"rm {network_fname}".split(" "))
			proc.wait()
	print(solutions)
	return solutions

