#coding: utf-8

import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from scipy.stats.mstats import gmean
from subprocess import call as sbcall
from multiprocessing import cpu_count
from time import time

from mpbn import load, MPBooleanNetwork
from mpbn.simulation import MPSimMemory, step, constant_maximum_depth, uniform_rates, nexponential_depth, nexponential_rates, constant_unitary_depth, fully_asynchronous_rates
from bonesis import BoNesis
import maboss

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from NORDic.UTILS.utils_state import compare_states, binarize_experiments
from NORDic.UTILS.utils_grn import solution2influences, create_grn

import mpbn
import mpbn.simulation as mpbn_sim
from time import time
from sklearn.metrics.pairwise import nan_euclidean_distances as ndist

from numpy.random import choice
from random import seed as rseed
from numpy.random import seed as npseed
from mpbn.simulation import MPBNSim
from pandas import DataFrame

from joblib import Parallel, delayed, wrap_non_picklable_objects
from joblib.externals.loky import set_loky_pickler

import contextlib
@contextlib.contextmanager
def capture():
    import sys
    from io import StringIO
    oldout,olderr = sys.stdout, sys.stderr
    try:
        out=[StringIO(), StringIO()]
        sys.stdout,sys.stderr = out
        yield out
    finally:
        sys.stdout,sys.stderr = oldout, olderr
        out[0] = out[0].getvalue()
        out[1] = out[1].getvalue()

def test(enumerator, seednb, njobs, network_fname, control_profile, treated_profiles, compare_to, mutation_permanent={}, mutation_transient={}, gene_outputs=None, print_boxplot=False, verbose=True):
    assert compare_to in ["profiles","attractors"]
    sims_di = {}
    for c in control_profile.columns:
        sims_c = {}
        for t in treated_profiles.columns:
            net_c = enumerator(seednb,njobs)
            attrs1 = net_c.up_to_attractors(network_fname, control_profile[[c]], treated_profiles[[t]], mutation_permanent, mutation_transient, verbose)
            attrs1.columns = ["Attr%d_%s_1" % (i,c) for i in range(attrs1.shape[1])]
            if (compare_to=="profiles"):
                attrs2 = treated_profiles[[t]]
            else:
                net_t = enumerator(seednb,njobs)
                attrs2 = net_t.up_to_attractors(network_fname, treated_profiles[[t]], treated_profiles[[t]], {}, {}, verbose)
                attrs2.columns = ["Attr%d_%s_2" % (i,t) for i in range(attrs2.shape[1])]
            sims = net_c.attrs_similarity(attrs1, attrs2, gene_outputs=gene_outputs)
            sims_c.setdefault(t, sims)
            if(verbose):
                if (np.isnan(attrs1).values.all()&np.isnan(attrs2).values.all()):
                    print("NaN (not reachable)")
                    print("Similarity %s" % str(sims))
                    continue
                elif (np.isnan(attrs1).values.all()):
                    print("Treated %s (showing 10 first out of %d, Control %s: NaN, 5/%d genes)" % (t, attrs2.shape[1], c, attrs2.shape[0]))
                    #print(attrs2.iloc[:,:10].dropna(how="all").head())
                    if (any([g in gene_outputs for g in mutation])):
                        for g in [g for g in mutation if (g in gene_outputs)]:
                            print("".join(list(attrs2.loc[g].iloc[:,:10].astype(str))))
                    print("Similarity %s" % str(sims))
                    continue
                elif (np.isnan(attrs2).values.all()):
                    print("Control %s (showing 10 first out of %d, Treated %s: NaN, 5/%d genes)" % (c, attrs1.shape[1], t, attrs2.shape[0]))
                    #print(attrs1.iloc[:,:10].dropna(how="all").head())
                    if (any([g in gene_outputs for g in mutation])):
                        for g in [g for g in mutation if (g in gene_outputs)]:
                            print("".join(list(attrs1.loc[g].iloc[:,:10].astype(str))))
                    print("Similarity %s" % str(sims))
                    continue   
                A_set = attrs1.iloc[:,:10].join(attrs2.iloc[:,:10], how="outer").dropna(how="any")
                A_set = A_set.loc[[g for g in gene_outputs if (g in A_set.index)]]         
                print("Control %s, Treated %s (showing 10 first out of %d, resp. 10/%d, %d genes)" % (c,t,attrs1.shape[1],attrs2.shape[1], A_set.shape[0]))
                subst = (A_set-A_set[attrs2.columns[0]]).mean(axis=1)
                if (A_set.shape[1]<=2):
                    print(A_set)
                else:
                    if ((subst!=0).any()):
                        print(A_set.loc[(subst!=0)])
                    else:
                        print("Attractors are equal.")
                mutation = list(set([g for g in mutation_permanent]+[g for g in mutation_transient]))
                if (any([g in A_set.index for g in mutation]) and any([subst.loc[g]==0 for g in mutation])):
                    print(A_set.loc[(subst==0)].loc[[g for g in mutation if (g in A_set.index)]])
                print("Similarity %s" % str(sims))
        sims_di.setdefault(c, sims_c)
    net_c.sims = sims_di
    if (print_boxplot):
        net_c.boxplot()
    net_c.max_values = {"%s->%s"%(c,t):np.max(net_c.sims[c][t]) for c in net_c.sims for t in net_c.sims[c]}
    #gmean = lambda ls: np.prod(np.power(ls,1/len(ls)))
    gmeansim = gmean([net_c.max_values[m]+1 for m in net_c.max_values])-1
    return gmeansim, net_c

## sims = {control: {treated: [[sim]] (or [[sim(a)] for all attractors])}
class BN_SIM(object):
    def __init__(self, seednb=0,njobs=None):
        self.seednb = seednb
        self.initial = None
        self.mutation_permanent = {}
        self.mutation_transient = {}
        self.all_mutants = []
        self.gene_outputs = None
        self.sims = None
        self.attrs = None
        self.network = None
        self.njobs = njobs if (njobs is not None) else max(1,cpu_count()-2)

    def boxplot(self):
        assert self.sims is not None
        f_lst = []
        for im, mut_di in enumerate([self.mutation_transient, self.mutation_permanent]):
            f_lst += [g+(":=" if (im==1) else "~")+("OE" if (gval>0) else "KO") for g, gval in mut_di.items()]
        f = "_".join(f_lst)
        controls = list(self.sims.keys())
        treated = list(self.sims[controls[0]].keys())
        ncontrol, ntreated = len(controls), len(treated)
        fig, axes = plt.subplots(nrows=ntreated, ncols=ncontrol, figsize=(5*(1.5**int(ncontrol>1)),5*(1.5**int(ntreated>1))))
        for ic, c in enumerate(controls):
            for it, t in enumerate(treated):
                if (ntreated==1):
                    ax = axes[ic]
                elif (ncontrol==1):
                    ax = axes[it]
                else:
                    ax = axes[it][ic]
                ticks_loc = ax.get_xticks().tolist()
                ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
                ax.set_xticklabels([f for x in ticks_loc])
                ax.set_ylabel("%s\n(N=%d)||%s(N=%d)" % (c, self.sims[c][t].shape[0], t, self.sims[c][t].shape[1]))
                ax.boxplot(self.sims[c][t].flatten())
                avg = np.mean(self.sims[c][t])
                ax.text(1.1, avg, "mean=%.2f" % avg, color="black", size=10)
                ax.set_title("%s\n%s vs %s" % (f, c, t))
        plt.show()

    def attrs_similarity(self, attrs1, attrs2, gene_outputs=None):
        assert self.network is not None
        if (np.isnan(attrs1.values).all() or np.isnan(attrs2.values).all()):
            return np.array([[0.]])
        if (gene_outputs is None):
            self.gene_outputs = [g for g in self.gene_list if (g not in [m for m in self.mutation_permanent]+[m for m in self.mutation_transient])]
        else:
            self.gene_outputs = gene_outputs
        sims, ngenes = compare_states(attrs1, attrs2, self.gene_outputs)
        if (ngenes!=len(self.gene_outputs)):
            print("/!\ %d/%d genes (initially %d %d)" % (ngenes, len(self.gene_outputs), attrs1.shape[0], attrs2.shape[0]))
        return sims

    def update_network(self, network_fname, initial, final=None, mutation_permanent={}, mutation_transient={}, verbose=True):
        if (final is None):
            final = initial
        assert initial.shape[1]==1 and final.shape[1]==1
        self.initialize_network(network_fname)
        if (verbose):
            print("Initialized network")
        with open(network_fname, "r") as f:
            grf_list = f.read().split("\n")
        grfs = dict([x.split(", ") for x in grf_list if (len(x)>0)])
        input_genes = [g for g in grfs if (grfs[g] in ["0","1"])] ## genes with no regulators
        self.mutation_permanent = {inp: int(initial.loc[inp][initial.columns[0]]) for inp in input_genes if (inp in initial.dropna().index)}
        N = len(self.mutation_permanent)
        self.mutation_permanent.update(mutation_permanent)
        if (verbose):
            print("Initialized %d permanent mutations (%d from initial state)" % (len(self.mutation_permanent), N))
        self.mutation_transient.update(mutation_transient)
        if (verbose):
            print("Initialized %d transient mutations" % len(self.mutation_transient))
        self.all_mutants = list(set([g for g in self.mutation_permanent]+[g for g in self.mutation_transient]))
        if (verbose):
            print("... Total of %d (unique) mutations" % len(self.all_mutants))
        self.add_transient_mutation(self.mutation_transient)
        self.add_permanent_mutation(self.mutation_permanent)
        self.add_initial_states(initial, final)
        if (verbose):
            print("Initialized initial state")
        return None

    def up_to_attractors(self, network_fname, initial, final, mutation_permanent={}, mutation_transient={}, verbose=True):
        self.update_network(network_fname, initial, final, mutation_permanent, mutation_transient, verbose=verbose)
        attrs = self.enumerate_attractors(verbose)
        if (verbose):
            print("Enumerated %d attractors" % attrs.shape[1])
        return attrs

    def initialize_network(self, network_fname):
        raise NotImplemented

    def add_initial_states(self, initial, final=None):
        raise NotImplemented

    def add_transient_mutation(self, mutation):
        raise NotImplemented

    def add_permanent_mutation(self, mutation):
        raise NotImplemented

    def enumerate_attractors(self, verbose=False):
        raise NotImplemented

    def generate_trajectories(self, params={}, outputs=[]):
        raise NotImplemented

############ MPBN VERSION

class MPBN_SIM(BN_SIM):
    def __init__(self, seednb=0,njobs=None):
        super(MPBN_SIM, self).__init__(seednb,njobs)
        self.attrs = []

    def initialize_network(self, network_fname):
        self.network = MPBooleanNetwork(load(network_fname))
        self.gene_list = [g for g in self.network.zero()]

    def add_initial_states(self, initial, final=None):
        assert initial.shape[1]==1
        x0 = self.network.zero()
        for i in initial.loc[~pd.isnull(initial[initial.columns[0]])].index:
            x0[i] = int(initial.loc[i][initial.columns[0]])
        for g, gval in self.mutation_transient.items():
            x0[g] = int(gval)
        self.initial = x0

    def add_transient_mutation(self, mutation):
        pass

    def add_permanent_mutation(self, mutation):
        for i, fi in mutation.items():
            self.network[i] = fi

    def enumerate_attractors(self, max_attrs=-1, verbose=True):
        random.seed(self.seednb)
        np.random.seed(self.seednb)
        #attrs_gen = self.network.fixedpoints(reachable_from=self.initial)
        if (verbose):
            attrs_gen = self.network.attractors(reachable_from=self.initial)
        else:
            with capture() as out:
                attrs_gen = self.network.attractors(reachable_from=self.initial)
        if (max_attrs>0):
            attrs = [None]*max_attrs
            idt = lambda x : x
            for ia, a in (tqdm if (verbose) else idt)(enumerate(attrs_gen)):
                self.attrs.append(a)
                attrs[ia] = pd.DataFrame({"Attr%d"%ia: a})
                if (max_attrs>0 and (ia==max_attrs-1)):
                    break
        else:
            self.attrs = [a for a in (tqdm if (verbose) else idt)(attrs_gen)]
            attrs = [pd.DataFrame({"Attr%d"%ia: a}) for ia, a in enumerate(self.attrs)]
        attrs_df = attrs[0].join(attrs[1:], how="outer")
        self.attrs = attrs_df
        return attrs_df

    def generate_trajectories(self, params={}, outputs=[], show_plot=True):
        rseed(self.seednb)
        npseed(self.seednb)
        nsims = params.get('sample_count', 1000)
        njobs = params.get('thread_count', 1)#self.njobs)
        max_time = params.get("max_time", 1000)
        noutputs = [g for g in (self.gene_list if (len(outputs)==0) else outputs) if (g not in self.all_mutants)]
        seeds = choice(range(int(max(1e4,nsims))), size=nsims)
        name_state = lambda s : " -- ".join(list(sorted([g for g in s if ((s[g]==1) and (g in noutputs))]))) if (any([(s[g]==1) and (g in noutputs) for g in s])) else "<nil>"
        #@delayed
        #@wrap_non_picklable_objects
        def generate_trajectory(snb, net, params, istates, noutputs):
            rseed(snb)
            npseed(snb)
            netsim = MPBNSim(net)
            depth = params.get("depth", constant_maximum_depth)(netsim)
            W = params.get("W", uniform_rates)(netsim)
            n, k, x = len(netsim), 0, istates.copy()
            mem = MPSimMemory(n, n)
            str_state = name_state(x)
            trajectory = [] # [k: state at k]
            trajectory.append([0,str_state,1])
            while (k<=max_time):
                k += 1
                r = step(netsim, mem, x, depth, W)
                str_state = name_state(x)
                trajectory.append([k,str_state,1])
                if (not r):
                    for ki in range(k+1, max_time):
                        trajectory.append([ki,str_state,1])
                    break
            return trajectory
        def plot_trajectory(probs):
            plt.figure(figsize=(8,5))
            plt.xlim((-max_time//20,max_time+max_time//20))
            plt.ylim((-0.1,1.1))
            for c in probs.columns:
                plt.plot(range(max_time), list(probs[c]), label=c)
            plt.xlabel("")
            plt.ylabel("")
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.show()
        if ((njobs==1) or (nsims==1)):
            trajectories = [y for s in seeds for y in generate_trajectory(s, self.network, params, self.initial, noutputs)]
        else:
            set_loky_pickler()
            parallel = Parallel(n_jobs=njobs, backend='loky')
            trajectories = [y for x in parallel(delayed(generate_trajectory)(s, self.network, params, self.initial, noutputs) for s in seeds) for y in x]
        trajectories = pd.DataFrame(trajectories, index=range(nsims*max_time), columns=["step","state","count"])
        probs = pd.pivot_table(trajectories, values="count", index="step", columns="state", aggfunc='count').fillna(0)/nsims
        if (show_plot):
            plot_trajectory(probs)
        table = pd.DataFrame([list(probs.loc[max_time-1])], index=["prob"], columns=["{"+",".join([g+"=1" for g in c.split(" -- ")])+"}" if (c!="<nil>") else c for c in probs.columns])
        table = table[[c for c in table.columns if (float(table.loc["prob"][c])!=0)]]
        return table

############ BONESIS VERSION 

class BONESIS_SIM(BN_SIM):
    def __init__(self, seednb=0,njobs=None):
        super(BONESIS_SIM, self).__init__(seednb,njobs)
        self.nattr = 0

    def initialize_network(self, network_fname):
        with open(network_fname, "r") as f:
            solution = pd.Series(dict([x.split(", ") for x in f.read().split("\n")]))
        influences = solution2influences(solution)
        influences.index = ["_".join(x.split("-")) for x in influences.index]
        influences.columns = ["_".join(x.split("-")) for x in influences.columns]
        self.gene_list = list(set(list(influences.index)+list(influences.columns)))
        self.grn = create_grn(influences, exact=True, quiet=True)
        #self.network = list(BoNesis(self.grn, {}).boolean_networks())[0]

    def add_initial_states(self, initial, final):
        initial_ = pd.DataFrame(initial.values, index=initial.index, columns=["initial"]).dropna().astype(int)
        initial_ = initial_.T
        for g, gval in self.mutation_transient.items():
            initial_[g] = int(gval)
        initial_ = initial_.T
        final_ = pd.DataFrame(final.values, index=final.index, columns=["final"])
        data_df = final_.join(initial_, how="outer")
        data_df.columns = ["exp","init"]
        data_df.index = ["_".join(x.split("-")) for x in data_df.index]
        data_exps = data_df[["exp"]].dropna().to_dict()
        data_exps.update(data_df[["init"]].dropna().to_dict())
        self.network = BoNesis(self.grn, data_exps)
        self.initial = initial_.to_dict()["initial"]
        #self.network = list(BoNesis(self.grn, data_exps).boolean_networks())[0]

    def add_transient_mutation(self, mutation):
        pass

    def add_permanent_mutation(self, mutation):
        pass

    def enumerate_attractors(self, verbose=True):
        #attrs = pd.DataFrame({"Attr%d"%i : attr for i, attr in enumerate(self.network.attractors(reachable_from=self.state))})
        #return attrs
        if (len(self.mutation_permanent)==0):
            final_FP = self.network.fixed(~self.network.obs("exp"))
            ~self.network.obs("init") >= final_FP
            ~self.network.obs("exp") >> "fixpoints" ^ {self.network.obs("exp")};
        else:
            with self.network.mutant(self.mutation_permanent) as m:
                final_FP = m.fixed(~m.obs("exp"))
                ~m.obs("init") >= final_FP
                ~m.obs("exp") >> "fixpoints" ^ {m.obs("exp")};
        if (verbose):
            BNs = list(self.network.boolean_networks(limit=1, njobs=self.njobs))
            nsol = len([bn for bn in tqdm(BNs)])
        else:
            with capture() as out:
                BNs = list(self.network.boolean_networks(limit=1, njobs=self.njobs))
            nsol = len([bn for bn in BNs])
        if (nsol>0):
            attrs = pd.DataFrame({"Attr%d" % self.nattr: self.network.data["exp"]})
        else:
            attrs = pd.DataFrame({"StateNotFound%d" % self.nattr: {"nan":np.nan}})
        self.nattr += 1
        sbcall("rm -f exist_1.zip", shell=True)
        self.attrs = attrs
        return attrs

    def generate_trajectories(self, params={}, outputs=[]):
        raise NotImplemented

############ MABOSS VERSION

class MABOSS_SIM(BN_SIM):
    def __init__(self, seednb=0,njobs=None):
        super(MABOSS_SIM, self).__init__(seednb,njobs)
        self.params = {
            'sample_count': 10000,
            'use_physrandgen': 0,
            'thread_count': self.njobs,
            'max_time': 1000,
            'time_tick': 1,
        }

    def initialize_network(self, network_fname):
        self.network_fname=network_fname
        self.folder=self.network_fname.split("/")[-1].split(".bnet")[0]
        ts = str(int(time()))
        sbcall("cp "+self.network_fname+" network"+ts+".bnet",shell=True)
        sbcall("zip -q -r "+self.folder+".zip network"+ts+".bnet", shell=True)
        self.network = maboss.Ensemble(self.folder+".zip")
        sbcall("rm -f "+self.folder+".zip network"+ts+".bnet", shell=True)
        self.network.param.update(self.params)
        self.network.set_outputs(self.network.nodes)
        self.gene_list = self.network.nodes

    def add_initial_states(self, initial, final=None):
        assert initial.shape[1]==1
        for g in self.network.nodes:
            p_g = 0.5 if (g not in initial.dropna().index) else float(initial.loc[g][initial.columns[0]])
            self.network.set_istate(g, [1.-p_g, p_g]) ## 0: prb(g==0 in initial state), 1: prb(g==1 in initial state)
        for g, p_g in self.mutation_transient.items():
            self.network.set_istate(g, [1.-p_g, p_g])
        self.initial = initial.copy().T
        for g, gval in self.mutation_transient.items():
            self.initial[g] = float(gval)
        self.initial = self.initial.T

    def add_transient_mutation(self, mutation):
        pass

    def add_permanent_mutation(self, mutation):
        self.network.mutations = mutation
        self.network.set_outputs([g for g in self.gene_list if (g not in self.network.mutations)]) 
        mutations_off, mutations_on = [[g for g in mutation if (mutation[g]==v)] for v in [0,1]]
        self.network = maboss.copy_and_mutate(self.network, mutations_off, "OFF")
        self.network = maboss.copy_and_mutate(self.network, mutations_on, "ON")

    def enumerate_attractors(self, verbose=True):
        sbcall("mkdir -p "+self.folder+"/models/",shell=True)
        self.network.write_cfg(self.folder+"/", self.folder+".cfg")
        sbcall("cp "+self.network_fname+" "+self.folder+"/models/"+self.folder+".bnet", shell=True)
        states = self.network.run(workdir=self.folder+"/")._get_raw_states()
        sbcall("rm -rf "+self.folder+"/",shell=True)
        res_mat = np.matrix([[int(g in s) for s in states[-1]] for g in self.gene_list])
        attrs = pd.DataFrame(res_mat, index=self.gene_list, columns=["S%d" % i for i in range(len(states[-1]))]).copy()
        self.attrs = attrs
        return attrs

    def generate_trajectories(self, params={}, outputs=[]):
        self.network.param.update(self.params if (len(params)==0) else params)
        self.network.set_outputs([g for g in (self.gene_list if (len(outputs)==0) else outputs) if (g not in self.network.mutations)])
        sbcall("mkdir -p "+self.folder+"/models/",shell=True)
        self.network.write_cfg(self.folder+"/", self.folder+".cfg")
        sbcall("cp "+self.network_fname+" "+self.folder+"/models/"+self.folder+".bnet", shell=True)
        result = self.network.run(workdir=self.folder+"/")
        result.plot_trajectory()
        table = result.get_last_states_probtraj()
        table.columns = ["{"+",".join([cc+"=1" for cc in list(sorted(c.split(" -- ")))])+"}" if (c!="<nil>") else c for c in table.columns]
        table.index = ["prob"]
        sbcall("rm -rf "+self.folder+"/",shell=True)
        return table
