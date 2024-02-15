#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.optimize as sp
import matplotlib.pyplot as plt
import math

#should work as presented in the paper, and provide the same results
##################################CONSTANTS####################################

qstar = 0.8 # arg min of cost function
gamma = 1 # scaling factor in the defender's revenue expression 
d = 2 # number of elements in the alphabet set {0;1}
p = 0.5 #H_0
q1 = 0.7 #Left bound of Q
q2 = 0.9 #Right bound of Q
n = 150  # number of observations
maxitr = 15

max_treshold = n
utility_bias = 10 #converges more steadily with all utility values > 0

attacker_rounding_decimals = 6
defender_rounding_decimals = 1
##################################FUNCTIONS####################################

def c(q):
    return np.abs(q-qstar)

def q_func(q, xn): #q(x^n)
    return (q**xn.ones) * ( (1-q) ** (xn.zeros) )

def p_func(xn): #p(x^n)
    return (2)**(-n)

def phi(xn, threshold): #the phi() used in their numerical tests
    if xn.ones >= threshold:
        return 1
    else:
        return 0

def utility_function(q, threshold, xns):
    attack_cost = c(q)
    temporary_sum = 0
    for xn in xns:
        phi_xns = phi(xn, threshold)
        successful_attack = (1 - phi_xns) * q_func(q,xn)
        false_positive = gamma * phi_xns * p_func(xn)
        temporary_sum += (xn.count) * (successful_attack + false_positive)
    return temporary_sum + utility_bias - attack_cost

# CUSTOM FUNCTIONS #
class xn():
    def __init__(self, n, ones):
        self.n = n
        self.ones = ones
        self.zeros = n-ones
        self.count = n_choose_k(n, ones)

    #computes the utility for chosen MIXED attacker strategy and PURE defender strategy
def mixed_utility_function_defender(threshold, qs, distribution, xns):
    temporary_sum = 0
    for i, q in enumerate(qs):
        temporary_sum += distribution[i] * utility_function(q, threshold, xns)
    return temporary_sum
    
    #computes optimal PURE defender strategy
def optimal_defender_move(qs, distribution, xns):
    # this minimize function+method cannot be bounded, but provides most accurate results
    # function, init_guess, method, function args, bounds, options 
    ret = sp.minimize(mixed_utility_function_defender, n , method='Nelder-Mead', args=(qs, distribution, xns), tol=1/(10**defender_rounding_decimals) )
    return ret['x'][0]

    #computes the utility for chosen PURE attacker strategy and MIXED defender strategy
def mixed_utility_function_attacker(q, thresholds, distribution, xns):
    temporary_sum = 0
    for i, threshold in enumerate(thresholds):
        temporary_sum += distribution[i] * utility_function(q, threshold, xns)
    return -temporary_sum
    
    #computes optimal PURE attacker strategy
def optimal_attacker_move(thresholds, distribution, xns):
    # this minize function doesnt always provide accurate results, but can be bounded
    # function, init_guess, method, function args, bounds, options 
    return sp.fminbound(mixed_utility_function_attacker, q1, q2, (thresholds, distribution, xns), xtol=1/(10**attacker_rounding_decimals) )

    #'a' stands for the attacker
def optimal_mixed_strategy(matrix, player = 'a'):
    if player == 'a':
        matrix = matrix.transpose()
    height, width = matrix.shape
    # [1 0 0 0 ... 0]
    function_vector = np.insert( np.zeros(width), 0, 1)
    # [-1 | A]
    boundary_matrix = np.insert(matrix, 0, values=-1, axis=1)
    # [0 1 1 ... 1]
    eq_matrix = np.array([np.insert(np.ones(width), 0, values=0, axis=0)])
    # {options} added on behalf what the functions itself demanded in stdout
    if player == 'a': #maximizing player
        ret = sp.linprog( -function_vector, -boundary_matrix, np.zeros(height), eq_matrix, np.array([1]), options={'autoscale': True, 'cholesky': False, 'sym_pos':False})
    else:             #minimizing player
        ret = sp.linprog( function_vector, boundary_matrix, np.zeros(height), eq_matrix, np.array([1]), options={'disp': False, 'cholesky': False, 'sym_pos':False})
    if ret['success'] is not True:
        print("DID NOT FIND EQUILIBRIUM!")
        exit()
    x = ret['x'][1:]
    return x
    
    #|  n  |
    #|  k  |
def n_choose_k(n, k):
    return math.factorial(n) / ( math.factorial(k)*math.factorial(n-k) )

def generate_binary_combinations(n):
    xns = []
    for i in range(n+1):
        xns.append(xn(n, i))
    return xns
    
def game(printout=True):
    #Modify numpy array printing
    np.set_printoptions(edgeitems=30, linewidth=1000, formatter=dict(float=lambda x: "%.9g" % x))
    #init the set of all x^n
    xns = generate_binary_combinations(n)
    #used in the loop for convergence check
    opt_atk = -1
    opt_def = -1
    #attack pure strategies
    qs = np.round(np.random.uniform(low = q1, high = q2, size=1), decimals=attacker_rounding_decimals)

    #defense pure strategies
    thresholds = np.round(np.random.uniform(low = 0, high = max_treshold, size=1), decimals=defender_rounding_decimals)

    #init starting game matrix
    matrix = np.array( [[utility_function(qs[0], thresholds[0], xns)]] )   
    if(printout):
        print("q's: ")
        print(qs)
        print("thresholds:")
        print(thresholds)
        print("init matrix is ")
        print(matrix)

    itr = 0
    while itr < maxitr:
        
        itr += 1
        #find best mixed strategy
        mixed_attack_distribution = optimal_mixed_strategy(matrix, player='a')
        mixed_defense_distribution = optimal_mixed_strategy(matrix, player='b')
        
        if(printout):
            print("------------------------------------------------------------------------")
            print("Mixed attack strategy: ", mixed_attack_distribution)
            print("Mixed defense strategy: ", mixed_defense_distribution)

        #save previous oteration for convergence check
        prev_opt_def = opt_def 
        prev_opt_atk = opt_atk
        #find best pure strategy answer
        opt_def = round( optimal_defender_move(qs, mixed_attack_distribution, xns), defender_rounding_decimals) #round to 1 decimal
        opt_atk = round( optimal_attacker_move(thresholds, mixed_defense_distribution, xns), attacker_rounding_decimals) #round to 6 decimals
        
        if(printout):
            print("optimal defense = ", opt_def)
            print("optimal attack = ", opt_atk)

        #check convergence
        if prev_opt_def == opt_def and prev_opt_atk == opt_atk:
            print("converge found with values q:", opt_atk, "threshold: ", opt_def)
            break
   
        #add pure stragegies to the matrix
        matrix = np.insert(matrix, 0, values=0, axis=0)
        matrix = np.insert(matrix, 0, values=0, axis=1)
        qs = np.insert(qs, 0, values=opt_atk)
        thresholds = np.insert(thresholds, 0, values=opt_def)  
        for i in range( matrix.shape[0] ):
            matrix[i][0] = utility_function(qs[i], opt_def, xns)
            matrix[0][i] = utility_function(opt_atk, thresholds[i], xns)

        if(printout):
            print(matrix)
    
    #final mixed strategy evaluation
    mixed_attack_distribution = optimal_mixed_strategy(matrix, player='a')
    mixed_defense_distribution = optimal_mixed_strategy(matrix, player='b')

    if(printout):
        print("Mixed attack strategy: ", mixed_attack_distribution)
        print("qs")
        print(qs)
        print("Mixed defense strategy: ", mixed_defense_distribution)
        print("thresholds")
        print(thresholds)
    
    return opt_def, opt_atk, n, qs, mixed_attack_distribution, thresholds, mixed_defense_distribution, itr
################ END OF THE GAME SIMULATION, UTIL AND PLOTTING FUNCTIONS ONLY #################  

def plot_attacker_util_function(q1, q2, threshold):
    utils = [] #handy container
    xns = generate_binary_combinations(n)
    for q in np.linspace(q1,q2,100): 
        utils.append(utility_function(q, threshold, xns))
    plt.plot(np.linspace(q1,q2,100), utils)
    plt.xlabel("q")
    plt.ylabel("Attacker revenue")
    plt.show()

def plot_defender_util_function(q, threshold1, threshold2):
    utils = [] #handy container
    xns = generate_binary_combinations(n)
    for threshold in np.linspace(threshold1, threshold2, 100): 
        utils.append(-utility_function(q,threshold,xns))
    #print(utils)
    plt.plot(np.linspace(threshold1, threshold2, 100), utils)
    plt.title("Defender's utility function")
    plt.xlabel("threshold")
    plt.ylabel("utility")
    plt.show()

def plot_best_responses_by_n(n1, n2):
    global n, max_treshold
    thresholds = [] 
    qs = []
    for n_iter in np.arange(n1, n2):
        n = int(n_iter)
        max_treshold = n
        game_result = game(printout=False)
        thresholds.append(game_result[0])
        qs.append(game_result[1])
    plt.plot(np.arange(n1, n2), thresholds, 'x')
    plt.title("Best responses")
    plt.xlabel("n")
    plt.ylabel("threshold")
    plt.show()

def plot_best_responses(q1, q2):
    xns = generate_binary_combinations(n)
    thresholds = [] 
    qs = []
    for q in np.linspace(q1,q2,100):
        thresholds.append(optimal_defender_move([q], [1], xns))
    t1 = min(thresholds)
    t2 = max(thresholds)
    for threshold in np.linspace(t1,t2,100):
        qs.append(optimal_attacker_move([threshold], [1], xns))
    plt.plot(np.linspace(q1,q2,100), thresholds, 'o')
    plt.plot(qs, np.linspace(t1,t2,100), 'x', color="red")
    plt.title("Best responses")
    plt.xlabel("q")
    plt.ylabel("threshold")
    plt.show()

def plot_optimal_distributions():
    _,_,n, qs, q_distribution, thresholds, threshold_distribution,_ = game(printout=False)
    qs_reduced, qs_sum_indices = np.unique(qs, return_inverse=True)
    q_distribution_reduced = []
    for i in range(len(qs_reduced)):
        temp = np.sum(q_distribution, where=(qs_sum_indices==i) )
        if temp > 0:
            q_distribution_reduced.append(temp)
    #print(qs_reduced)
    #print(q_distribution_reduced)
    plt.title("Optimal attack distribution, n="+str(n))
    plt.xlabel("q")
    plt.ylabel("%")
    plt.bar(qs_reduced, q_distribution_reduced, width=0.007, bottom=None,  align='center')
    plt.show()
    #THRESHOLDS
    thresholds = np.round(thresholds, decimals=0)
    thresholds_reduced, thresholds_sum_indices = np.unique(thresholds, return_inverse=True)
    threshold_distribution_reduced = []
    for i in range(len(thresholds_reduced)):
        temp = np.sum(threshold_distribution, where=(thresholds_sum_indices==i) )
        if temp > 0:
            threshold_distribution_reduced.append(temp)
    plt.title("Optimal defense distribution, n="+str(n))
    plt.xlabel("threshold")
    plt.ylabel("%")
    plt.bar(thresholds_reduced, threshold_distribution_reduced, width=3, bottom=None,  align='center')
    plt.show()

def plot_steps_needed(iters):
    steps = np.zeros(15)
    for i in range(iters):
        _, _, _, _, _, _, _, itr = game(printout=False)
        steps[itr] += 1
    plt.title("Number of steps needed to converge, n="+str(n))
    plt.xlabel("steps")
    plt.ylabel("occurence")
    plt.bar( range(len(steps)), steps, width=1, bottom=None,  align='edge')
    plt.show()

def plot_phi(threshold):
    xns = generate_binary_combinations(n)
    values = []
    for xn in xns:
        values.append( phi(xn, threshold) )
    plt.plot(range(n+1), values, 'x')
    plt.title("Phi")
    plt.xlabel("n")
    plt.ylabel("value")
    plt.show()

################################functions for error testing, evaluate.py################################################
def generate_optimal_thresholds(ns):
    global n, max_treshold
    thresholds = []
    for n_iter in ns:
        n = int(n_iter)
        max_treshold = n
        xns = generate_binary_combinations(n)
        thresholds.append( optimal_defender_move( [0.8], [1], xns) )
    
    print(thresholds)


if __name__ == "__main__":
    #game(printout=True)
    #plot_attacker_util_function(q1, q2, (0.65)*n)
    #plot_attacker_util_function(q1, q2, 163.3)
    #plot_attacker_util_function(q1, q2, 31.2)
    #plot_defender_util_function(qstar, 97, 105)
    #plot_defender_util_function(qstar, 2*n/3-7, 2*n/3+6)
    #plot_defender_util_function(qstar, 93, 102)
    #plot_best_responses_by_n(30, 150)
    #plot_steps_needed(20)
    plot_best_responses(0.7, 0.9)
    #plot_optimal_distributions()
    #plot_phi(97.5)
    #generate_optimal_thresholds( np.linspace(10, 300, 30, dtype = np.int) )