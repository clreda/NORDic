#coding: utf-8

import pandas as pd
import numpy as np
from joblib import Parallel, delayed

import NORDic.NORDic_DR.bandits as bandits
import NORDic.NORDic_DR.utils as utils

def adaptive_testing(network_name, signatures, targets, score, states, simu_params={}, bandit_args={}, 
        reward_fname=None, quiet=False):
    '''
        Perform adaptive testing and recommends most promising treatments (=maximizing score)
        @param\tnetwork_name\tPython character string: (relative) path to a network .BNET file
        @param\tsignatures\tPandas DataFrame: rows/[features] x columns/[drugs to test]
        @param\ttargets\tPandas DataFrame: rows/[genes] x columns/[drugs to test] (either 1: active expression, -1: inactive expression, 0: undetermined expression)
        @param\tscore\tPython object: scoring of attractors
        @param\tstates\tPandas DataFrame: rows/[gene] x columns/[patient samples] (either 1: activatory, -1: inhibitory, 0: no regulation).
        @param\tsimu_params\tPython dictionary[default={}]: arguments to MPBN-SIM
        @param\tbandit_params\tPython dictionary[default={}]: arguments to the bandit algorithms
        @param\treward_fname\tPython character string[default=None]: path to a reward matrix rows/[patients] x columns/[drugs]
        @param\tquiet\tPython bool[default=False]
        @return\tempirical_rec\tPandas DataFrame: rows/[drugs to test] x column/["Frequency"], the percentage of times 
        across all simulations at the end of which the considered drug is recommended
    '''
    assert signatures.shape[1]==targets.shape[1]
    assert all([c==targets.columns[i] for i, c in enumerate(signatures.columns)])
    assert states.shape[0]==targets.shape[0]
    assert all([c==targets.index[i] for i, c in enumerate(states.index)])
    assert bandit_args.get('bandit', 'LinGapE') in bandits.bandit_types
    assert bandit_args.get('beta', 'heuristic') in bandits.beta_types
    assert bandit_args.get('learner', 'AdaHedge') in bandits.learner_types
    assert bandit_args.get('gain_type', 'empirical') in utils.gain_types
    assert bandit_args.get('tracking_type', 'D') in utils.tracking_types
    assert bandit_args.get("delta", None)
    assert bandit_args.get("m", None)

    delta, m = [bandit_args[x] for x in ["delta", "m"]]
    sigma = bandit_args.get("sigma", 1)
    c = bandit_args.get("c", 0)
    problem_args={ 
        "network_name": network_name,
        "reward_fname": reward_fname,
        "targets": targets,
        "score": score,
        "states": states,
        "simu_params": simu_params,
    }
    np.random.seed(bandit_args.get("seed", 0))
    problem = testing_problem(signatures, problem_args)

    bandit_args.update({
        "beta_linear": eval("bandits."+bandit_args.get('beta', 'heuristic'))(problem.X, delta, sigma, c), "n": m, 
        "tracking_type": bandit_args.get('tracking_type', 'D'), "learner_name": bandit_args.get('learner', 'AdaHedge'),
        "gain_type": bandit_args.get('gain_type', 'empirical'), "subsample": bandit_args.get('subsample', False), 
        "geometric_factor": bandit_args.get('geometric_factor', 1.3)})
    bandit_algorithm=eval("bandits."+bandit_args["bandit"])(bandit_args)

    if (simu_params.get('thread_count', 1)==1 or bandit_args.get("nsimu", 1)==1):  # No parallelization in this case
        active_arms, complexity, running_time = bandit_algorithm.run(problem, bandit_args.get("nsimu", 1))
        active_arms = [[a] for a in active_arms.tolist()] if (bandit_args.get("nsimu", 1)==1) else active_arms
    else:
        # Generate one random seed for each run (best practice with joblib)
        seeds = [np.random.randint(int(1e8)) for _ in range(bandit_args.get("nsimu", 1))]
        # Function to perform a single experiment
        def single_run(id_, seed):
            # joblib replicates the current process, so we need to manually set a different seed for each run
            np.random.seed(seed)
            return bandit_algorithm.run(problem, nsimu=1, run_id=id_)
        # run nsimu simulations over n_jobs parallel processes
        results = Parallel(n_jobs=simu_params.get('thread_count', 1), backend='loky')(delayed(single_run)(id_, seed) for id_, seed in enumerate(seeds))
        # we finally merge the results so as to have them in the same form as for a single process
        active_arms = np.mean(np.array([r[0] for r in results]), axis=0).tolist()
        complexity = [r[1][0] for r in results]
        running_time = [r[2][0] for r in results]

    # results 
    empirical_rec = pd.DataFrame(active_arms, columns=["Frequency"], index=signatures.columns).T
    if (not quiet):
        print("<NORDic DR> Avg. #samples = %d, avg. runtime %s sec (over %d iterations)" % (np.mean(complexity), np.mean(running_time), bandit_args.get("nsimu", 1)))
    return empirical_rec

################################
## TESTING PROBLEM            ##
################################
from NORDic.NORDic_DS.functions import simulate_treatment

class testing_problem(object):
    def __init__(self, signatures, problem_args):
        self.X = signatures.values
        for k in problem_args:
            setattr(self, k, problem_args[k])
        self.memoization = np.nan*np.zeros((signatures.shape[1], self.states.shape[1]))
        self.targets = self.targets[signatures.columns]
        if (self.reward_fname is not None):
            rewards = pd.read_csv(self.reward_fname, index_col=0).loc[self.states.columns][signatures.columns]
            self.memoization = rewards.values.T

    def reward(self, arm):
        patient = np.random.choice(range(self.memoization.shape[1]), 1, p=None, replace=True)
        if (not np.isnan(self.memoization[arm, patient])):
            return float(self.memoization[arm, patient])
        result = float(simulate_treatment(self.network_name, self.targets[[self.targets.columns[arm]]], 
		self.score, self.states.iloc[:,patient], 
		self.simu_params, quiet=False)[0])
        self.memoization[arm, patient] = result
        return result
