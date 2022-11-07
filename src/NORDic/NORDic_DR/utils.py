#coding: utf-8

import numpy as np
from quadprog import solve_qp

gain_types = ["unstructured", "empirical", "linear", "misspecified", "aggressive_misspecified"]
tracking_types = ["C", "D", "S"]

#################
## UTILS       ##
#################

#' @param M squared matrix of size (N, N)
#' @param x vector of size (N, 1)
#' @returns one-step iterative inversion of matrix (M^{-1}+xx^T)
def sherman_morrison(M, x):
    return M-((M.dot(x)).dot(x.T.dot(M))/float(1+mahalanobis(x, M, power=2)))

#' @param M positive definite matrix of size (N, N)
#' @param x vector of size (N, 1)
#' @returns Mahalanobis norm of x wrt M
def mahalanobis(x, M, power=1):
    assert power in [1,2]
    return float(np.sqrt(x.T.dot(M.dot(x))) if (power==1) else x.T.dot(M.dot(x)))

#' @param ls Python list of elements
#' @param m integer
#' @param f function taking a list as argument
#' and returning a value of the same type as the elements of input list
#' @return m random indices i1,i2,...,im of the list which satisfy for all i in i1,i2,...,im, ls[i] == f(ls)
def randf(ls, m, f):
    return np.random.choice(np.array(np.argwhere(np.array(ls) == f(ls))).flatten().tolist(), size=m, replace=False, p=None)

#' @param ls Python list of elements with a total order natively implemented in Python
#' @param m integer
#' @returns m distinct indices of the list which values are the m maximal ones (with multiplicity)
def argmax_m(ls, m):
    assert m <=len(ls)
    allowed = list(range(len(ls)))
    values = [None]*m
    for i in range(m):
        idx = randf([ls[i] for i in allowed], 1, np.max)[0]
        values[i] = allowed[idx]
        del allowed[idx]
    return values

#' @param X matrix of size (N, K)
#' @param C approximation value
#' @param quiet boolean
#' @returns a C-approximate barycentric spanner of the columns of X as done in [Awerbuch et al., 2008]
#' kind of slow, but the only other algorithm ([Amballa et al., 2021] to be published in AAAI) focuses on a specific set
def barycentric_spanner(X, C=1, quiet=True, precision=1e-6):
    N, K = X.shape
    assert K > 0
    def det_(M):
        sign, logdet = np.linalg.slogdet(M)
        return sign*np.exp(logdet)
    if (min(N,K)==K):
        return list(range(K)) #result from Awerbuch et al.
    # basis of set of arm features
    Fx = np.matrix(np.eye(N))
    F = [None]*N
    S = range(K)
    for a in range(N):
        other_ids = [u for u in range(N) if (u != a)]
        # replace Fx[:,a] with X[:,s], s in S
        max_det, max_det_id = -float("inf"), None
        for s in S:
            Xa = np.hstack((X[:,s].reshape((X.shape[0],1)), Fx[:,other_ids]))
            dXa = det_(Xa)
            # keep it linearly independent
            if (dXa > max_det):
                max_det = dXa
                max_det_id = s
        Fx[:,a] = X[:,max_det_id].reshape((X.shape[0],1))
        F[a] = max_det_id
    # transform basis into C-approximate barycentric spanner of size <= d
    done = False
    while (not done):
        found = False
        for s in S:
            for a in range(N):
                other_ids = [u for u in range(N) if (u != a)]
                det_Xs = det_(np.hstack((X[:,s].reshape((X.shape[0],1)), Fx[:,other_ids]))) # |det(x, X_{-a})|
                det_Xa = det_(Fx) # |det(X_a, X_{-a})|
                if ((det_Xs-C*det_Xa) > precision): # due to machine precision, might loop forever otherwise
                    Fx[:,a] = X[:,s].reshape((X.shape[0],1))
                    F[a] = s
                    found = True
        done = not found
    spanner = [f for f in F if (str(f) != F)]
    if (not quiet):
        print("Spanner size d = "+str(len(spanner))+" | K = "+str(K)),
    return F

#' @param x vector
#' @param norm type of norm ||.||
#' @returns ||x||
def cnorm(x, norm="L_inf"):
    assert norm in ["L_inf", "L2", "L1"]
    if (norm == "L_inf"):
        return np.max(np.abs(x))
    elif (norm == "L2"):
        return np.linalg.norm(x)
    elif (norm == "L1"):
        return np.sum(np.abs(x))
    else:
        raise ValueError("Norm not implemented.")

#' @param x input
#' @returns Lambert's function for negative branch y=-1
def lambert(y, approx=False):
    if (approx):
        ## if y >= 1, use the upper bound on W_(y) for computational reasons
        W_ = lambda y : y + np.log(y) + min(0.5, 1/np.sqrt(y)) if (y >= 1.) else 1.
    else:
        from scipy.special import lambertw
        W_ = lambda y : np.real(-lambertw(-np.exp(-y), k=-1))
    return W_(y)

############################
## UPDATE ESTIMATORS      ##
############################

#' @param problem Problem instance as implemented in problems.py
#' @param candidates list of arm ids
#' @param Vinv NumPy Array: inverse of design matrix
#' @param b NumPy Array
#' @param c scale of deviation
#' @param Vinv_val pre-computed Vinv
#' @returns Vinv, b, means, theta, eta using samples from candidates
def update_misspecified(problem, candidates, Vinv, b, c, na, rewards, Vinv_val=None):
    N, K = problem.X.shape
    candidates.reverse()
    for ia, a in enumerate(candidates):
        i_a = -(ia+1)
        b[a,0] += rewards[i_a]
        x_a = problem.X[:, a]
        ## update Vinv
        if (str(Vinv_val) == "None"):
            Vinv = sherman_morrison(Vinv, x_a.reshape((x_a.shape[0],1)))
    if (str(Vinv_val) != "None"):
         Vinv = Vinv_val

    nb_pulls = np.array(na)
    if (np.min(nb_pulls) < 1e-10):
        nb_pulls = nb_pulls + 1e-10
    x_hat = np.diag([1./float(n) for n in nb_pulls]).dot(b)
    theta, eta = projection(Vinv, b, x_hat, problem.X, nb_pulls, c)
    means = np.array(problem.X.T.dot(theta) + eta).flatten()
    return Vinv, b, means, theta, eta

############################
## TRACKING ALLOCATIONS   ##
############################

## minimal distance alternative in all subproblems with best arm a != i_t
def closest_alternative(problem, b, means, theta, eta, w, c, S_t, constraint="L_inf", subsample=False, alternative_arms=[]):
    d, K = problem.X.shape
    if subsample:
        all_other_arms = [a for a in range(K) if (a not in S_t)]
        random_arms = np.random.choice(all_other_arms, d, replace=False)  # the size of the sampled arms is here
        other_arms = [a for a in all_other_arms if ((a in alternative_arms) or (a in random_arms))]
    else:
        other_arms = [a for a in range(K) if (a not in S_t)]
    all_alts = np.zeros((len(S_t), len(other_arms), K+1))
    for i in range(len(S_t)):
        for a in range(len(other_arms)):
            alt = solve_alternative_quadprog(problem, b, theta, eta, other_arms[a], S_t[i], w, c, constraint)
            all_alts[i, a, :-1] = (np.array(alt[0])).flatten()
            all_alts[i, a, -1] = alt[1]
    closest_id = np.unravel_index(all_alts[:, :, -1].argmin(), all_alts[:, :, -1].shape)
    means_alt = all_alts[closest_id[0], closest_id[1], :-1].reshape((K,1))
    val = all_alts[closest_id[0], closest_id[1], -1]
    closest = [S_t[closest_id[0]], other_arms[closest_id[1]]]
    
    closest_arm_in = S_t[closest_id[0]]
    closest_arm_other = other_arms[closest_id[1]]

    if (subsample):
        if len(alternative_arms) > d+len(S_t):
            alternative_arms.pop(0)
        if closest_arm_in not in alternative_arms:
            alternative_arms.append(closest_arm_in)
        if closest_arm_other not in alternative_arms:
            alternative_arms.append(closest_arm_other)

    return means_alt, closest, val, alternative_arms

def solve_alternative_quadprog(problem, b, theta_emp, eta_emp, a, i_t, w, epsilon, constraint=None):

    assert a != i_t

    d, K = problem.X.shape
    A = problem.X.T
    mu_emp = A.dot(theta_emp) + eta_emp
    u = np.zeros(K)
    u[a] = -1.
    u[i_t] = 1.

    P = np.zeros((K+d, K+d))
    q = np.zeros(K+d)
    G = np.zeros((2*K+1, K+d))
    h = np.zeros(2*K+1)

    D = np.diag(w)
    V = A.T.dot(D).dot(A)

    P[:d, :d] = V
    P[d:, d:] = D
    P[d:, :d] = D.dot(A)
    P[:d, d:] = (D.dot(A)).T

    q[:d] = np.array(A.T.dot(D).dot(mu_emp)).flatten()
    q[d:] = np.array(D.dot(mu_emp)).flatten()

    G[0, :] = np.hstack([A.T.dot(u).reshape((1,d)), u.reshape((1,K))])
    G[1:K+1, d:] = np.identity(K)
    G[K+1:, d:] = -np.identity(K)

    h[1:] = epsilon + 1e-5

    theta_eta = quadprog_solve_qp(P, -q, G=G, h=h)

    mu_alt = A.dot(theta_eta[:d]) + theta_eta[d:]
    val = 0.5 * np.linalg.norm(np.sqrt(w)*np.array(mu_alt).flatten() - np.sqrt(w)*np.array(mu_emp).flatten())**2
    return mu_alt, val

#source:https://scaron.info/blog/quadratic-programming-in-python.html
# Wrapper for quadprog module solve_qp function for quadratic problems
def quadprog_solve_qp(P, q, G=None, h=None, A=None, b=None):
    # solve 0.5*x^T P x + q^T x, with Gx <= h and Ax = b
    cste = np.mean(np.abs(P))
    P /= cste
    reg_term = 1e-8
    while (np.linalg.det(P+reg_term*np.identity(len(q))) < 0):
        reg_term *= 10
    P += reg_term*np.identity(len(q))
    q /= np.sqrt(cste)
    G /= np.sqrt(cste)
    qp_G = .5 * (P + P.T)   # make sure P is symmetric
    qp_a = -q
    if A is not None:
        A /= np.sqrt(cste)
        qp_C = -np.vstack([A, G]).T
        qp_b = -np.hstack([b, h])
        meq = A.shape[0]
    else:  # no equality constraint
        qp_C = -G.T
        qp_b = -h
        meq = 0
    return solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0] / np.sqrt(cste)

def projection(Vinv, b, x_hat, X, nb_pulls, c):
    N, K = X.shape
    sqD = np.diag(np.sqrt(nb_pulls))
    ## empirical means
    x_hat = np.diag([1./float(n) for n in nb_pulls]).dot(b)
    X = X.T
    A_N, x_N = sqD.dot(X), sqD.dot(x_hat)
    I = np.identity(K)
    R = I - A_N.dot(Vinv).dot(A_N.T)
    R = R.T.dot(R)

    if c > 0:
        h = np.array(float(c) * np.sqrt(nb_pulls))  
        # norm constraint on eta_N
        P = np.array(R + 1e-8*I)  
        # the identity matrix ensures that R has no negative eigenvalues (which are due to numerical errors)
        q = np.array(- R.dot(x_N)).flatten()
        res = quadprog_solve_qp(P, q, G=np.vstack((I, -I)), h=np.hstack((h, h)))
        eta_N = np.array(res).reshape((K, 1))
    else:
        eta_N = np.zeros_like(x_N)

    theta = Vinv.dot(A_N.T).dot(x_N - eta_N)
    eta = np.diag(1./np.sqrt(nb_pulls)).dot(eta_N)
    return theta.reshape((N,1)), eta.reshape((K, 1))

#' @param w NumPy Array of length K
#' @param sum_w NumPy array of length K
#' @param na NumPy array of length K
#' @param t Python integer
#' @param tracking_type Python string character in ["C", "D", "S"]
#' @returns set of candidates to sample
def tracking_rule(w, sum_w, na, t, tracking_type, forced_exploration=False):
    assert str(na) != "None"
    K =  len(na)
    if (forced_exploration):
        undersampled = np.array(np.array(na) >= (np.sqrt(t)-0.5*K)*np.ones(np.array(na).shape), dtype=int)
        if (np.sum(undersampled)>0):
            w = (undersampled/np.sum(undersampled)).tolist()
    if (tracking_type == "C"):
        sampled = randf([float(na[a]-sum_w[a]) for a in range(K)], 1, np.min)
    elif (tracking_type == "D"):
        sampled = randf([float(na[a]-t*w[a]) for a in range(K)], 1, np.min)
    elif (tracking_type == "S"):
        sampled = np.random.choice(range(K), size=1, p=w, replace=False)
    else:
        raise ValueError("Type of tracking rule not implemented.")
    return sampled

## Upper bound on (mu_k - Proj(mu_k(t)))^2 with high probability
#' @param direction NumPy Array
#' @param problem Problem instance as implemented in problems.py
#' @param na NumPy Array
#' @param t Python integer
#' @param Vinv NumPy matrix: inverse of design matrix
#' @param M upper bound on scores
#' @param c scale of deviation to linearity
#' @param x optional Python float
#' @param confidence_width 
#' @return confidence bound
def c_kt(direction, problem, na, t, Vinv, M, c, x=None, confidence_width=None, cnorm=None):
    if (str(cnorm)=="None"):
        cnorm = lambda x : np.max(np.abs(x)) # ||.||_inf norm
    N, K = problem.X.shape
    L = float(np.max(np.linalg.norm(problem.X, axis=0)))
    if (str(x)=="None"):
        x = 2*L
    X = 5 #number of good events in the analysis
    T_max = 4*M**2
    if (str(confidence_width)=="None"):
        T_lin = 2*np.log(X*t**2)+N*np.log(1+(t*L**2)/(x*N)) # theoretical value
    else:
        T_lin = confidence_width # simplified one
    e = np.exp(1)
    T_uns = 2*K*lambert(1/float(2*K)*np.log(e*X*t**3)+0.5*np.log(8*e*K*np.log(t)))
    Na = sum([1/float(n) for n in na])
    ckt = min(2*T_uns*Na, T_max)
    return min(ckt, 8*c**2+2*T_lin*mahalanobis(direction, Vinv)**2)

#' @param problem Problem instance as described in problems.py
#' @param Vinv Inverse of design matrix
#' @param mu estimated model NumPy Array
#' @param lambda_ alternative model NumPy Array
#' @param na NumPy Array
#' @param t Python integer
#' @param M upper bound on scores
#' @param c scale of deviation to linearity
#' @param gain_type Python string character
#' @x Python float
#' @return gradients of gains to feed to learner
def optimistic_gradient(problem, Vinv, mu, lambda_, na, t, M, c, gain_type, x=None):
    X = problem.X
    N, K = X.shape
    grads = np.zeros(K)
    if ((np.array(na) < 1e-10).any()):
        nb_pulls = np.array(na) + 1e-10
    else:
        nb_pulls = np.array(na)
    for a in range(K):
        ref_value = (mu-lambda_)[a]
        L = float(np.max(np.linalg.norm(X, axis=0)))
        confidence_width = np.log(t) # smaller confidence width than expected theoretically
        #confidence_wdith = 2*np.log(t)+N*np.log(1+(t*L**2)/N) # theoretical one
        if (gain_type == "unstructured"):
            deviation = np.sqrt(2*confidence_width/nb_pulls[a]) # Hoeffding's bounds
        elif (gain_type =="linear"):
            deviation = np.sqrt(2*confidence_width)*float(mahalanobis(X[:,a], Vinv)) # bounds in LinGame
        elif ("misspecified" in gain_type):
            deviation = np.sqrt(c_kt(problem.X[:,a], problem, [nb_pulls[a]], t, Vinv, M, c, x=x, confidence_width=confidence_width)) # bounds in paper
        elif (gain_type == "empirical"):
            deviation = 0.
        else:
            raise ValueError("Unimplemented type of gains: '"+gain_type+"'.")
        # gradient of gains wrt w_a for all a
        if (ref_value > 0):
            grads[a] = 0.5*(ref_value+deviation)**2
            if (gain_type == "aggressive_misspecified"):
                grads[a] = (ref_value**2+deviation**2)
        else:
            grads[a] = 0.5*(ref_value-deviation)**2
            if (gain_type == "aggressive_misspecified"):
                grads[a] = (ref_value**2+deviation**2)
        grads[a] = min(grads[a], confidence_width)
    return grads