#!/usr/bin/python
# -*- coding: utf-8 -*-

#developed for Python 3.8
#author - Tomas Kasl, FEL CVUT

import numpy as np
import scipy.optimize as sp
import matplotlib.pyplot as plt
import math
from util import *

# should work as presented in the paper
# CHANGES - compared the reference_bayes_initial file:
#   utility_function dropped the p(x) member
#   utility_function has value increased by +1 000 000 to account for illegal threshold for phi but let remain the descendence direction
#   (illegal values are those, for which the Neyman_Pearson condition for error < € does not hold)
#   (that is sum_xns(phi(x) * p(x) )) > € )

##################################CONSTANTS####################################

qstar = 0.8 # arg min of cost function
gamma = 1 # scaling factor in the defender's revenue expression 
d = 2 # number of elements in the alphabet set {0;1}
p = 0.5 #H_0
q1 = 0.7 #Left bound of Q
q2 = 0.9 #Right bound of Q
n = 150  # number of observations
maxitr = 6
epsilon = 0.1
utility_bias = 100 #converges more steadily with all values > 0
attacker_rounding_decimals = 6
defender_rounding_decimals = 1

##################################FUNCTIONS####################################

    #(float) q -> (float) cost
def c(q):
    return np.abs(q-qstar)

    #the phi used in the numerical test in the initial paper
    #(bin. vector) xn, (float) threshold -> (bool) decision
def phi(xn, threshold):
    if xn.ones >= threshold:
        return 1
    else:
        return 0

    #the zero-sum game equivalent utility function
    #(float) q, (float) threshold, (list of bin. vectors) xns -> (float) utility gain
def utility_function(q, threshold, xns):
    attack_cost = c(q)
    #check the € condition holds
    temporary_sum = 0
    for xn in xns:
        temporary_sum += phi(xn, threshold)*p_func(xn)
    np_bias = 1e6 if  (temporary_sum > epsilon) else 0 
    #compute the utility itself
    temporary_sum = 0
    for xn in xns:
        phi_xns = phi(xn, threshold)
        successful_attack = (1 - phi_xns) * q_func(q,xn)
        temporary_sum += (xn.count) * successful_attack
    return (temporary_sum + utility_bias - attack_cost) + np_bias

# CUSTOM FUNCTIONS #

    #computes the utility for chosen MIXED attacker strategy and PURE defender strategy
    #(float) threshold, (float list) qs, (float list) q-distribution, (list of bin. vectors) xns -> (float) utility gain
def mixed_utility_function_defender(threshold, qs, distribution, xns):
    temporary_sum = 0
    for i, q in enumerate(qs):
        temporary_sum += distribution[i] * utility_function(q, threshold, xns)
    return temporary_sum
    
    #computes optimal PURE defender strategy
    #(float list) qs, (float list) q-distribution, (list of bin. vectors) xns -> (float) threshold
def optimal_defender_move(qs, distribution, xns):
    ret = sp.minimize(mixed_utility_function_defender, n/3 , method='Nelder-Mead', args=(qs, distribution, xns), tol=1/(10**defender_rounding_decimals) )
    return ret['x'][0]

    #computes the utility for chosen PURE attacker strategy and MIXED defender strategy
    #(float) q, (float list) thresholds, (float list) threshold-distribution, (list of bin. vectors) xns -> (float) utility gain
def mixed_utility_function_attacker(q, thresholds, distribution, xns):
    temporary_sum = 0
    for i, threshold in enumerate(thresholds):
        temporary_sum += distribution[i] * utility_function(q, threshold, xns)
    return -temporary_sum
    
    #computes optimal PURE attacker strategy
    #(float list) thresholds, (float list) threshold-distribution, (list of bin. vectors) xns -> (float) q
def optimal_attacker_move(thresholds, distribution, xns):
    return sp.fminbound(mixed_utility_function_attacker, q1, q2, (thresholds, distribution, xns), xtol=1/(10**attacker_rounding_decimals) )
    
    #the game is solved using the Double Oracle
    #(bool) printout -> (float) last_threshold, (float) last_q, (int) n, (float list) qs, (float list) q-distribution, 
    #                   (float list) thresholds, (float list) threshold-distribution, (int) iterations
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
    thresholds = np.round(np.random.uniform(low = n/2, high = n, size=1), decimals=defender_rounding_decimals)

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
        new_matrix = np.zeros( (itr+1, itr+1) )
        qs = np.insert(qs, 0, values=opt_atk)
        thresholds = np.insert(thresholds, 0, values=opt_def) 
        for x in range(itr):
            for y in range(itr):
                new_matrix[1+x][1+y] = matrix[x][y]
        for i in range( itr+1 ):
            new_matrix[i][0] = utility_function(qs[i], opt_def, xns)
        for i in range( itr+1):
            new_matrix[0][i] = utility_function(opt_atk, thresholds[i], xns)
        matrix = new_matrix
            
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
    plt.show()

def plot_defender_util_function(q, threshold1, threshold2):
    utils = [] #handy container
    xns = generate_binary_combinations(n)
    for threshold in np.linspace(threshold1, threshold2, 100): 
        utils.append(-utility_function(q,threshold,xns))
    plt.plot(np.linspace(threshold1, threshold2, 100), utils)
    plt.show()

def plot_best_responses_by_n(n1, n2):
    global n
    thresholds = [] 
    qs = []
    for n_iter in np.arange(n1, n2):
        n = int(n_iter)
        game_result = game(printout=False)
        thresholds.append(game_result[0])
        qs.append(game_result[1])
    plt.plot(np.arange(n1, n2), thresholds, 'x')
    plt.title("Best defender moves")
    plt.xlabel("n")
    plt.ylabel("threshold")
    plt.show()
    plt.title("Best attacker moves")
    plt.xlabel("n")
    plt.ylabel("q")
    plt.plot(np.arange(n1, n2), qs)
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
    #QS
    qs = np.round(qs, decimals=6)
    qs_reduced, qs_sum_indices = np.unique(qs, return_inverse=True)
    q_distribution_reduced = []
    for i in range(len(qs_reduced)):
        temp = np.sum(q_distribution, where=(qs_sum_indices==i) )
        if temp > 0:
            q_distribution_reduced.append(temp)
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
    plt.bar(thresholds_reduced, threshold_distribution_reduced, width=1, bottom=None,  align='center')
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

################################functions for error rate testing, evaluate.py################################################

    #for generating a list of optimal defender thresholds for each n
    #the list is used for real-world error rate measurement
    #(int list) n's -> (float list) thresholds 
def generate_optimal_thresholds(ns):
    global n
    thresholds = []
    for n_iter in ns:
        n = int(n_iter)
        xns = generate_binary_combinations(n)
        thresholds.append( optimal_defender_move( [0.8], [1], xns) )
    
    return thresholds

##########################################################################################################################š

if __name__ == "__main__":
    #game()
    plot_attacker_util_function(q1, q2, n/3)
    #plot_defender_util_function(qstar, 0, n)
    #plot_defender_util_function(qstar, n/4, n/3)
    #plot_best_responses_by_n(4, 250)
    #plot_steps_needed(100)
    #plot_best_responses(0.7, 0.9)
    #plot_optimal_distributions()
    #print( generate_optimal_thresholds( np.linspace(10, 300, 30, dtype = np.int) ) )