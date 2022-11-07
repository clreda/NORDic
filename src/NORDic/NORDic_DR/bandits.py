# coding: utf-8

complexity_limit=1e6

import gc
import numpy as np
from time import time

import NORDic.NORDic_DR.utils as utils

bandit_types = ["MisLid", "LinGapE"]
beta_types = ["heuristic", "linear", "lucb1", "misspecified", "gaussian", "subheuristic"]
learner_types = ["AdaHedge", "Greedy"]

####################################################
## Thresholds                                     ##
####################################################

# "Iterated log" inspired threshold [Kaufmann et al., 2015]
def heuristic(X, delta, sigma, c):
    N, K = X.shape
    def f(t, na, x=None):
        return np.log((1+np.log(t+1))/float(delta))
    return f

def subheuristic(X, delta, sigma, c):
    N, K = X.shape
    def f(t, na, x=None):
        return np.log(t+1)/float(delta)
    return f

# This paper's stopping rule for misspecified linear models
def misspecified(X, delta, sigma, c):
    N, K = X.shape
    L = float(np.max(np.linalg.norm(X, axis=0)))
    e = np.exp(1)
    def f(t, na, x=None):
        if (str(x) == "None"):
            x = 2*L
        term_lin = (lambda t : 1+np.log(2/float(delta))+(1+1/np.log(2/float(delta)))*0.5*N*np.log(1+t*L**2/(x*N)*np.log(2/float(delta)))+2*c**2*t)
        term_uns = (lambda t : 2*K*utils.lambert(1/float(2*K)*np.log(2*e/float(delta))+0.5*np.log(8*e*K*np.log(t))))
        return min(term_lin(t), term_uns(t))
    return f

# [Kaufmann et Koolen, 2018]
def gaussian(X, delta, sigma, c):
    ## S subset of arms
    S = range(X.shape[1])
    K = len(S)
    from scipy.special import zeta
    gC = lambda l : 2*l-2*l*np.log(4*l)+np.log(zeta(2*l, q=1))-0.5*np.log(1-l) 
    from scipy.optimize import minimize
    x = np.log(1./float(delta))/float(K)
    fc = lambda l : (gC(l)+x)/float(l)
    lam = minimize(fc, np.array([0.75]), bounds=[(0.511111,0.9999)], tol=1e-6, options={"maxiter":1000}).x
    Cx = fc(lam)
    def f(t, na, x=None):
        return float(2*np.sum(np.log(4+np.log(np.array(na)+np.asarray(np.array(na)==0, dtype=int))))+K*Cx)
    return f

####################################################
## Learners                                       ##
####################################################

# Learner structure
class Learner(object):
	def __init__(self, K, name="UnimplementedLearner"):
		self.K, self.name = K, name
		## Arm playing mixed strategy
		self.p = np.zeros(self.K)
	def act(self):
		raise NotImplemented
	def incur(self, w):
		raise NotImplemented

# Best Response learner
class Greedy(Learner):
	def __init__(self, K):
		super(Greedy, self).__init__(K, name="FTL")
	def act(self):
		p = np.asarray(self.p == np.min(self.p), dtype=float)
		p /= np.sum(p)
		return p
	def incur(self, w):
		self.p[:] = w

# AdaHedge learner (the one used in LinGame)
class AdaHedge(Learner):
	def __init__(self, K):
		self.delta = 0.01
		super(AdaHedge, self).__init__(K, name="AdaHedge")

	def act(self):
		eta = np.log(self.K)/float(self.delta)
		p = np.exp(-eta*(self.p-np.min(self.p)))
		p /= np.sum(p)
		return p

	def incur(self, w):
		p = self.act()
		eta = np.log(self.K)/float(self.delta)
		self.p += w
		m = np.min(w)-1./eta*np.log(p.T.dot(np.exp(-eta*(w-np.min(w)))))
		assert m != float("inf") and p.T.dot(w) >= m-1e-7
		self.delta += p.T.dot(w) - m

####################################################
## Bandit algorithm                               ##
####################################################

## Generic class
class Misspecified(object):
    def __init__(self, method_args):
        for k in method_args:
            setattr(self, k, method_args[k])

    def clear(self):
        self.rewards = []
        self.samples = []
        self.t = 0
        self.na = []

    def sample(self, problem, candidates):
        if (len(self.na)==0):
            self.na = [0]*(problem.X.shape[1])
        for arm in candidates:
            self.rewards.append(problem.reward(arm))
            self.samples.append(arm)
            self.t += 1
            self.na[arm] += 1

    def run(self, problem, nsimu, run_id=None, quiet=False):
        active_arms = np.array([0]*(problem.X.shape[1]))
        complexity = [None]*nsimu
        running_time = [None]*nsimu
        for simu in range(nsimu):
            self.clear()
            starting_time = time()
            active_arms_, complexity_ = self.apply(problem)
            running_time[simu] = time()-starting_time
            complexity[simu] = complexity_
            active_arms[active_arms_] += 1
            C = round(np.mean(complexity[:simu+1]),2)
            id_ = simu + 1 if run_id is None else run_id
            if (not quiet):
                print("It. #%d: %d samples (running avg. %d)" % (id_, complexity_, C))
            gc.collect()
        return active_arms/float(nsimu), complexity, running_time

## For misspecified models
class MisLid(Misspecified):
    def __init__(self, method_args):
        self.name = "MisLid"
        if (method_args["learner_name"] == "AdaHedge"):
            self.learner_type = AdaHedge
        elif (method_args["learner_name"] == "Greedy"):
            self.learner_type = Greedy
        else:
            raise ValueError("Learner '"+method_args["learner_name"]+"' not implemented.")
        self.T_init = 1 
        assert method_args["sigma"] == 1
        assert "beta_linear" in method_args
        method_args["beta"] = method_args["beta_linear"]
        self.m = method_args["m"]
        assert method_args["sigma"] == 1.
        self.lambda_ = method_args["sigma"]/float(20.)
        assert self.lambda_ > 0
        self.constraint = "L_inf"
        assert self.constraint in ["L_inf", "L1"]
        self.multi_learners, self.M = 0, 1
        self.cnorm = lambda x : utils.cnorm(x, norm=self.constraint)
        super(MisLid, self).__init__(method_args)

    def stopping_rule(self, quiet=True):
        if (not quiet):
            print("t="+str(self.t)+" B(t)="+str(self.B)+" thres(t)="+str(self.epsilon)+" stop="+str(self.B > self.epsilon))
        return (self.B > self.epsilon) or (self.t > complexity_limit)

    def best_answer(self, means):
        J = utils.argmax_m(means, self.m)
        return J

#    def update(self, problem, candidates, Vinv, b, Vinv_val=None):
#        return utils.update_misspecified(problem, candidates, Vinv, b, self.c, self.na, self.rewards, Vinv_val=Vinv_val)

    def update(self, problem, candidates, Vinv, b, Vinv_val=None):
        candidates.reverse()
        for ia, a in enumerate(candidates):
            i_a = -(ia+1)
            Vinv = utils.sherman_morrison(Vinv, problem.X[:, a].reshape((problem.X.shape[0], 1)))
            b += self.rewards[i_a]*problem.X[:, a].reshape(b.shape)
        theta = Vinv.dot(b)
        means = np.array(theta.T.dot(problem.X)).flatten().tolist()
        eta = np.zeros((problem.X.shape[1], 1))
        return Vinv, b, means, theta, eta

    def apply(self, problem, precision=1e-7):
        self.B, self.epsilon = -float("inf"), 0.
        N, K = problem.X.shape
        means = [0]*K
        b = np.zeros((N,1))
        learners, sum_w = {}, np.zeros(K)
        stop = False
        L = float(np.max([self.cnorm(problem.X[:,a]) for a in range(K)]))
        self.x = 2*L

        A = problem.X.T
        print("minimum eigenvalue of A^TA: {}".format(np.linalg.eigvals(A.T.dot(A)).min()))
        
        ## Consider a (C-approximate) barycentric spanner
        F = utils.barycentric_spanner(problem.X, C=1, quiet=True)
        ## Uncomment the next two lines to get regularized version of design matrix instead
        #F = range(K)
        #self.T_init = 0
        candidates, maxiter = [], len(F)*(self.T_init+1)
        from scipy.sparse.linalg import eigsh
        get_eigmin = lambda M : eigsh(M, k=1, which="SM")[0][0]
        V = np.zeros((N,N))
        alternative_arms = []  # arms in the recent closest alternatives
        stopping_rule_time = max(5*N, 10)  # next time at which the sampling rule is checked
        while (self.t < maxiter): # t0 initialization phases to check V_t0 >= x Id
            ## Round-Robin procedure
            for a in F:
                self.sample(problem, [a])
                V += problem.X[:,a].dot(problem.X[:,a].T)
            candidates += F
            sign, logdet = np.linalg.slogdet(V-self.x*np.eye(N))
            detVxId = sign*np.exp(logdet)
            sign, logdet = np.linalg.slogdet(V)
            detV = sign*np.exp(logdet)
            if (detVxId>=0 and detV>0):
                break    
        if (self.T_init > 0):
            Vinv = np.linalg.pinv(V)
            Vinv, b, means, theta, eta = self.update(problem, candidates, Vinv, b, Vinv_val=Vinv)
        else:
            ## Regularized version (necessary for invertible design matrices)
            ## In practice, never used
            Vinv = 1/float(self.lambda_**2)*np.eye(N)
            Vinv, b, means, theta, eta = self.update(problem, candidates, Vinv, b)
            V = np.linalg.pinv(Vinv)
            self.x = get_eigmin(V)
        while (not stop):
            S_t = self.best_answer(means)  # current best answer set
            
            if (self.multi_learners):  # Several learners
                learner = learners.get(tuple(S_t), None)  # get step from learner associated with best answer
            else:  # Only one learner
                learner = learners.get(0, None)  # get step from learner associated with best answer
            if (str(learner) == "None"):
                learner = self.learner_type(K)
            
            if (self.learner_name == "AdaHedge"):
                # query the learner
                w_t = learner.act() 
                # get closest alternative to current means - Best-Response
                means_alt, closest, val, alternative_arms = utils.closest_alternative(
                    problem, b, means, theta, eta, w_t, self.c,  S_t,
                    constraint=self.constraint, subsample=self.subsample, alternative_arms=alternative_arms)
            elif (self.learner_name == "Greedy"):
                # get closest alternative to current means - FTL
                means_alt, closest, val, alternative_arms = utils.closest_alternative(
                    problem, b, means, theta, eta, self.na, self.c,  S_t,
                    constraint=self.constraint, subsample=self.subsample, alternative_arms=alternative_arms)
            else:
                raise ValueError("Unknown learner type.")

            # optimism
            mu = np.array(means).flatten()
            lambda_ = np.array(means_alt).flatten()
            delta = -utils.optimistic_gradient(problem, Vinv, mu, lambda_, self.na, self.t, self.M, self.c,
                self.gain_type if (str(self.gain_type)!="None") else "misspecified")
            # update learner
            learner.incur(delta)  
            di_learner = {}
            di_learner.setdefault(tuple(S_t) if (self.multi_learners) else 0, learner)

            if (self.learner_name == "Greedy"):
                # pull argmin of the loss
                w_t = learner.act()

            nb_samples = 1
            if self.subsample:
                nb_samples = N
            for sample_id in range(nb_samples):  # Sample the same arm several times
                sum_w += w_t
                learners.update(di_learner)
                candidate = utils.tracking_rule(w_t, sum_w, self.na, self.t, self.tracking_type if (str(self.tracking_type)!="None") else "S") # default is "S"
                self.sample(problem, candidate)
                # update sum of rewards, estimated parameters and means
                Vinv, b, means, theta, eta = self.update(problem, [candidate], Vinv, b)
            # Stopping rule test
            if (((self.t > stopping_rule_time) and self.subsample) or (not self.subsample)):
                means_alt, closest, val, alternative_arms = utils.closest_alternative(problem, b, means, theta, eta, self.na,
                    self.c, self.best_answer(means), constraint=self.constraint, subsample=False, alternative_arms=alternative_arms)
                self.B = val
                self.epsilon = self.beta(self.t, self.na, x=self.x)
                stop = self.stopping_rule(quiet=(self.t % 1000 != 0))  # check stopping rule
                #stop = self.stopping_rule(quiet=False)
                stopping_rule_time = max(self.t, stopping_rule_time)*self.geometric_factor
        import pandas as pd
        print(pd.DataFrame([
	        means,
	        [np.mean(np.array(self.rewards)[np.array(self.samples)==a]) for a in np.unique(self.samples)]
        ],index=["means", "emp. means"], columns=problem.targets.columns))
        J = self.best_answer(means)
        return J, self.t

class LinGapE(Misspecified):
    def __init__(self, method_args):
        self.T_init = 1
        assert "beta_linear" in method_args
        self.name = "LinGapE"
        method_args["beta"] = method_args["beta_linear"]
        super(LinGapE, self).__init__(method_args)

    def stopping_rule(self):
        if (self.t % 2000 == 0):
            print("t="+str(self.t)+" B(t) = "+str(self.B))
        return (self.B <= self.epsilon) or (self.t > complexity_limit)

    def greedy(self, problem, b_t, c_t, Vinv):
        K = problem.X.shape[1]
        direction = problem.X[:, b_t]-problem.X[:, c_t]
        uncertainty = [float(utils.mahalanobis(direction.reshape((problem.X.shape[0],1)), utils.sherman_morrison(Vinv, problem.X[:,i].reshape((problem.X.shape[0],1))))) for i in range(K)]
        a = utils.randf(uncertainty, 1, np.min)[0]
        return [a]

    def optimized(self, problem, b_t, c_t, Vinv):
        p = self.ratio.get((b_t,c_t))
        if (not len(p)):
            ## https://math.stackexchange.com/questions/1639716/how-can-l-1-norm-minimization-with-linear-equality-constraints-basis-pu
            from scipy.optimize import linprog
            X = problem.X
            Aeq = np.concatenate((X, -X), axis=1)
            beq = (X[:,b_t]-X[:,c_t])
            F = np.ones((2*K, ))
            bounds = [(0, float("inf"))]*(2*K)
            solve = linprog(F, A_eq=Aeq, b_eq=beq, bounds=bounds)
            x = solve.x
            w = x[:K]-x[K:]
            assert solve.status == 0
            p = np.abs(w)
            p /= np.linalg.norm(w, 1)
            self.ratio.setdefault((b_t,c_t), p)
        samplable_arms = [i for i in range(problem.X.shape[1]) if (float(p[i]) > 0)]
        a = samplable_arms[self.randf([float(self.na[i]/float(p[i])) for i in samplable_arms], 1, np.min)]
        return [a]

    def update(self, problem, candidates, Vinv, b):
        candidates.reverse()
        for ia, a in enumerate(candidates):
            i_a = -(ia+1)
            Vinv = utils.sherman_morrison(Vinv, problem.X[:, a].reshape((problem.X.shape[0], 1)))
            b += self.rewards[i_a]*problem.X[:, a].reshape(b.shape)
        theta = Vinv.dot(b)
        means = np.array(theta.T.dot(problem.X)).flatten().tolist()
        return Vinv, b, means

    def apply(self, problem, greedy_sampling=True, lambda_val=1.):
        self.B = float("inf")
        self.ratio = {}
        N, K = problem.X.shape
        Vinv = 1/float(lambda_val**2)*np.matrix(np.eye(N)).reshape((N,N))
        b = np.matrix(np.zeros(N)).reshape((N,1)) 
        means = [0]*K
        self.Cbeta = lambda t, n : np.sqrt(2*self.beta(t,n))
        w = lambda t, c, a, Vinv : float(self.Cbeta(t, self.na[c]+self.na[a])*utils.mahalanobis((problem.X[:,c]-problem.X[:,a]).reshape((problem.X.shape[0],1)), Vinv))
        w_ind = lambda t, a, Vinv : float(self.Cbeta(t, self.na[a])*utils.mahalanobis(problem.X[:,a].reshape(problem.X.shape[0],1), Vinv))
        Bidx = lambda t, i, j, Vinv : float(means[i]-means[j]+w(t, i, j, Vinv))
        stop = False
        for _ in range(self.T_init):
            for a in range(K):
                self.sample(problem, [a])
        if (self.T_init > 0):
            Vinv, b, means = self.update(problem, [a for _ in range(self.T_init) for a in range(K)], Vinv, b)
        while (not stop):
            J = utils.argmax_m(means, self.m)
            notJ = [a for a in range(K) if (a not in J)]
            indices = [[means[c]-means[a]+w(self.t, c, a, Vinv) for c in notJ] for a in J]
            max_idx = utils.randf([np.max(ids) for ids in indices], 1, np.max)[0]
            b_t = J[max_idx]
            max_idx_idx = utils.randf(indices[max_idx], 1, np.max)[0]
            c_t = notJ[max_idx_idx]
            if (greedy_sampling):
                candidate = self.greedy(problem, b_t, c_t, Vinv)
            else:
                candidate = self.optimized(problem, b_t, c_t, Vinv)
            self.sample(problem, candidate)
            Vinv, b, means = self.update(problem, candidate, Vinv, b)
            self.B = float(indices[max_idx][max_idx_idx])
            stop = self.stopping_rule()
        J = utils.argmax_m(means, self.m)
        return J, self.t