#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.optimize as sp
import matplotlib.pyplot as plt
import math

#WORK IN PROGRESS
##################################CONSTANTS####################################

qstar = 0.8 # arg min of cost function
gamma = 1 # scaling factor in the defender's revenue expression 
d = 2 # number of elements in the alphabet set {0;1}
p = 0.5 #H_0
q1 = 0.7 #Leftmost point of H1
q2 = 0.9 #Rightmost point of H1
n = 50  # number of observations
init_matrix_size = 1 #matrix has initial size N x N
maxitr = 10

max_treshold = n
utility_bias = 0
##################################FUNCTIONS####################################

def c(q):
    #return 0
    return np.abs(q-qstar)

def q_func(q, xn): #q(x^n)
    return (q**xn.ones) * ( (1-q) ** (xn.zeros) )

def p_func(xn): #p(x^n)
    return (2)**(-n)

def phi(xn, threshold): #the phi() used in their numerical tests
    if xn.ones >= threshold: #???
        return 1
    else:
        return 0

def utility_function(q, threshold, xns):
    attack_cost = c(q)
    temporary_sum = 0
    xn = xns[int(n/2)]
    #coef =  (gamma * p_func(xn) * xn.count) # * 1
    #print(gamma * p_func(xn) * xn.count)
    coef = (n/2) / n
    """
    coef_sum = 0
    for xn in xns:
        phi_xns = phi(xn, n/2)
        successful_attack = (1 - phi_xns) * q_func((qstar+q1)/2,xn)
        false_positive = gamma * phi_xns * p_func(xn)
        coef_sum += (xn.count) * (successful_attack + false_positive)
    coef = c( (qstar+q1)/2)
    """
    coef = n**(1/2)
    for xn in xns:
        phi_xns = phi(xn, threshold)
        successful_attack = (1 - phi_xns) * q_func(q,xn)
        false_positive = gamma * phi_xns * p_func(xn)
        temporary_sum += (xn.count) * (successful_attack + false_positive)
    return temporary_sum + utility_bias - (attack_cost/coef*(n**(1/2)))

# CUSTOM FUNCTIONS #
class xn():
    def __init__(self, n, ones):
        self.n = n
        self.ones = ones
        self.zeros = n-ones
        self.count = n_choose_k(n, ones)

def mixed_utility_function_defender(threshold, qs, distribution, xns):
    temporary_sum = 0
    for i, q in enumerate(qs):
        temporary_sum += distribution[i] * utility_function(q, threshold, xns)
    return temporary_sum
    
def optimal_defender_move(qs, distribution, xns):
    # function, init_guess, method, function args, bounds, options 
    ret = sp.minimize(mixed_utility_function_defender, n , method='Nelder-Mead', args=(qs, distribution, xns))
    return ret['x'][0]

def mixed_utility_function_attacker(q, thresholds, distribution, xns):
    temporary_sum = 0
    for i, threshold in enumerate(thresholds):
        temporary_sum += distribution[i] * utility_function(q, threshold, xns)
    return -temporary_sum
    
def optimal_attacker_move(thresholds, distribution, xns):
    # function, init_guess, method, function args, bounds, options 
    return sp.fminbound(mixed_utility_function_attacker, q1, q2, (thresholds, distribution, xns), xtol=1e-18)

    #tCORRECTED LP!
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
    if player == 'a': #maximizing player
        print("Attacker")
        ret = sp.linprog( -function_vector, -boundary_matrix, np.zeros(height), eq_matrix, np.array([1]), options={'autoscale': True, 'cholesky': False, 'sym_pos':False})
    else:             #minimizing player
        print("Deffender")
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
    
def game():
    #Modify printing of arrays
    np.set_printoptions(edgeitems=30, linewidth=1000, formatter=dict(float=lambda x: "%.9g" % x))
    xns = generate_binary_combinations(n)
    #attack pure strategies
    qs = np.random.uniform(low = q1, high = q2, size=init_matrix_size)
    print("q's: ")
    print(qs)
    #defense pure strategies
    thresholds = np.random.uniform(low = 0, high = max_treshold, size=init_matrix_size)
    print("thresholds:")
    print(thresholds)
    #init starting game matrix
    matrix = np.zeros((init_matrix_size, init_matrix_size))    
    for i1, threshold in enumerate(thresholds):
        for i2, q in enumerate(qs):
            matrix[i1][i2] = utility_function(q, threshold, xns)

    print("init matrix is ")
    print(matrix)

    itr = 0
    while itr < maxitr:
        print("------------------------------------------------------------------------")
        itr += 1
        #find best mixed strategy
        mixed_attack_distribution = optimal_mixed_strategy(matrix, player='a')
        print("Mixed attack strategy: ", mixed_attack_distribution)
        mixed_defense_distribution = optimal_mixed_strategy(matrix, player='b')
        print("Mixed defense strategy: ", mixed_defense_distribution)

        #find best pure strategy answer
        opt_def = optimal_defender_move(qs, mixed_attack_distribution, xns)
        opt_atk = optimal_attacker_move(thresholds, mixed_defense_distribution, xns)
        print("optimal defense =", opt_def)
        print("optimal attack =", opt_atk)

        #add pure stragegies to the matrix
        matrix = np.insert(matrix, 0, values=0, axis=1)
        matrix = np.insert(matrix, 0, values=0, axis=0)
        qs = np.insert(qs, 0, values=opt_atk)
        thresholds = np.insert(thresholds, 0, values=opt_def)
        for i in range(init_matrix_size+itr):
            matrix[0][i] = utility_function(opt_atk, thresholds[i], xns)
            matrix[i][0] = utility_function(qs[i], opt_def, xns)
        print(matrix)

        #convergence 
        if itr > 1 and (abs(qs[0]-qs[1]) < 1e-5) and (abs(thresholds[0]-thresholds[1]) < 1e-1):
            break
    
    #final mixed strategy evaluation
    mixed_attack_distribution = optimal_mixed_strategy(matrix, player='a')
    print("Mixed attack strategy: ", mixed_attack_distribution)
    mixed_defense_distribution = optimal_mixed_strategy(matrix, player='b')
    print("Mixed defense strategy: ", mixed_defense_distribution)
    print("qs")
    print(qs)
    print("thresholds")
    print(thresholds)
    return opt_def, opt_atk, n, qs, mixed_attack_distribution, thresholds, mixed_defense_distribution
################ END OF THE GAME SIMULATION, UTIL AND PLOTTING FUNCTIONS ONLY #################  

def plot_attacker_util_function(q1, q2, threshold):
    utils = [] #handy container
    xns = generate_binary_combinations(n)
    for q in np.linspace(q1,q2,100): 
        utils.append(utility_function(q, threshold, xns))
    plt.plot(np.linspace(q1,q2,100), utils)
    plt.show()

def plot_defender_util_function(q, threshold1, threshold2):
    utils = [] #handy container
    xns = generate_binary_combinations(n)
    for threshold in np.linspace(threshold1, threshold2, 100): 
        utils.append(-utility_function(q,threshold,xns))
    #print(utils)
    plt.plot(np.linspace(threshold1, threshold2, 100), utils)
    plt.show()

def plot_best_responses_by_n(n1, n2):
    global n, max_treshold
    thresholds = [] 
    qs = []
    for n_iter in np.arange(n1, n2):
        n = int(n_iter)
        max_treshold = n
        game_result = game()
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
    _,_,n, qs, q_distribution, thresholds, threshold_distribution = game()
    qs = np.round(qs, decimals=6)
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
    print(thresholds_reduced)
    print(threshold_distribution_reduced)
    plt.title("Optimal defense distribution, n="+str(n))
    plt.xlabel("threshold")
    plt.ylabel("%")
    plt.bar(thresholds_reduced, threshold_distribution_reduced, width=5, bottom=None,  align='center')
    plt.show()


if __name__ == "__main__":
    #game()
    #plot_attacker_util_function(q1, q2, 2*n/3)
    #plot_defender_util_function(qstar, 10, 50)
    #plot_best_responses_by_n(15, 150)
    #plot_best_responses(0.7, 0.9)
    plot_optimal_distributions()