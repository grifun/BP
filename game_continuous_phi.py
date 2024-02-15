#!/usr/bin/python
# -*- coding: utf-8 -*-

#developed for Python 3.8
#author - Tomas Kasl, FEL CVUT

import numpy as np
import scipy.optimize as sp
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
from util import *

# phi changed into continuous here: 
#   sigmoid ->          alpha*( 1/(1+exp(-alpha * q(x)/p(x) ) + beta ) 
#   linear function ->  alpha*( sum(x^n)/n ) + beta 
#
# CHANGES - compared the reference_bayes file:
#   optimal_defender_move functions now minimizes the utility function over 2 variables, alpha and beta
#   -> different minimalization 'method' used - can optimize multiple variables
# 
# the list structure:
# alphas_betas = [
#                [alpha_0,beta_0],
#             ...[alpha_n,beta_n] ]

##################################CONSTANTS####################################

#PHI = "linear"
PHI = "sigmoid"

qstar = 0.8 # arg min of cost function
gamma = 1 # scaling factor in the defender's revenue expression 
d = 2 # number of elements in the alphabet set {0;1}
p = 0.5 #H_0
q1 = 0.7 #Left bound of Q
q2 = 0.9 #Right bound of Q
n = 150  # number of observations
maxitr = 50
utility_bias = 1 #converges more steadily with all values > 0
attacker_rounding_decimals = 6
defender_rounding_decimals = 1
##################################FUNCTIONS####################################

    #(float) q -> (float) cost
def c(q):
    return np.abs(q-qstar)

    #the phi proposed in the thesis
    #(bin. vector) xn, (float) alpha, (float) beta -> (bool) decision
def phi(xn, alpha, beta): 
    if PHI == "sigmoid":
        exponent = ( -alpha * q_func(qstar,xn)/p_func(xn) +beta)
        return 1/(1+np.exp(exponent))
    elif PHI == "linear":
        value = xn.ones/n
        if alpha * value + beta > 1:
            return 1
        if alpha * value + beta < 0:
            return 0
        else:
            return alpha * value + beta
    else:
        print("UNKNOWN PHI, linear AND sigmoid are allowed only")
        exit()

    #the zero-sum game equivalent utility function
    #(float) q, (float) alpha, (float) beta, (list of bin. vectors) xns -> (float) utility gain
def utility_function(q, alpha, beta, xns):
    attack_cost = c(q)
    temporary_sum = 0
    for xn in xns:
        phi_xns = phi(xn, alpha, beta )
        successful_attack = (1 - phi_xns) * q_func(q,xn)
        false_positive = gamma * phi_xns * p_func(xn)
        temporary_sum += (xn.count) * (successful_attack + false_positive)
    return temporary_sum - attack_cost + utility_bias

# CUSTOM FUNCTIONS #

    #computes the utility for chosen MIXED attacker strategy and PURE defender strategy
    #(float list) {alpha, beta}, (float list) qs, (float list) q-distribution, (list of bin. vectors) xns -> (float) utility gain
def mixed_utility_function_defender( alpha_beta , qs, distribution, xns):
    alpha = alpha_beta[0]
    beta = alpha_beta[1]
    temporary_sum = 0
    for i, q in enumerate(qs):
        temporary_sum += distribution[i] * utility_function(q, alpha, beta , xns)
    return temporary_sum

    #computes optimal PURE defender strategy
    #(float list) qs, (float list) q-distribution, (list of bin. vectors) xns -> (float) alpha, (float) beta
def optimal_defender_move(qs, distribution, xns):
    ret = sp.minimize(mixed_utility_function_defender, (1,0), method='SLSQP',  args=(qs, distribution, xns))
    return ret['x']

    #computes the utility for chosen PURE attacker strategy and MIXED defender strategy
    #(float) q, (structure) alphas_betas, (float list) threshold-distribution, (list of bin. vectors) xns -> (float) utility gain
def mixed_utility_function_attacker(q, alphas_betas, distribution, xns):
    temporary_sum = 0
    for i, (alpha, beta) in enumerate(alphas_betas):
        temporary_sum += distribution[i] * utility_function(q, alpha, beta, xns)
    return -temporary_sum
    
    #computes optimal PURE attacker strategy
    #(structure) alphas_betas, (float list) alpha_beta-distribution, (list of bin. vectors) xns -> (float) q
def optimal_attacker_move(alphas_betas, distribution, xns):
    return sp.fminbound(mixed_utility_function_attacker, q1, q2, (alphas_betas, distribution, xns), xtol=1/(10**attacker_rounding_decimals))

    #the game is solved using the Double Oracle
    #(bool) printout -> (float) last_alpha, (float) last_beta, (float) last_q, (int) n, (float list) qs, (float list) q-distribution, 
    #                   (float list) thresholds, (float list) threshold-distribution, (int) iterations
def game(printout=True):
    #Modify printing of arrays
    np.set_printoptions(edgeitems=30, linewidth=1000, formatter=dict(float=lambda x: "%.9g" % x))
    xns = generate_binary_combinations(n)
    #used in the loop for convergence check
    opt_atk = -1
    opt_alpha = -1
    opt_beta = -1    
    #attack pure strategies
    qs = np.random.uniform(low=q1, high=q2, size=1)
    #defense pure strategies
    alphas_betas = np.round([ [np.random.rand(1)[0], np.random.rand(1)[0]] ], decimals = defender_rounding_decimals)
    #init starting game matrix
    matrix = np.array( [[utility_function(qs[0], alphas_betas[0][0], alphas_betas[0][1], xns)]] )   

    if(printout):
        print("q's: ")
        print(qs)
        print("alpha, beta:")
        print(alphas_betas)
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
        
        prev_opt_alpha = opt_alpha
        prev_opt_beta = opt_beta
        prev_opt_atk = opt_atk

        #find best pure strategy answer
        opt_alpha, opt_beta = np.round(optimal_defender_move(qs, mixed_attack_distribution, xns), decimals=defender_rounding_decimals)
        opt_atk = round(optimal_attacker_move(alphas_betas, mixed_defense_distribution, xns), attacker_rounding_decimals) 
        
        if(printout):
            print("optimal alpha, beta = ", opt_alpha, opt_beta)
            print("optimal attack =", opt_atk)
        
        #check convergence
        if prev_opt_alpha == opt_alpha and prev_opt_beta == opt_beta and prev_opt_atk == opt_atk:
            print("converge found with values q:", opt_atk, "alpha, beta = ", opt_alpha, opt_beta)
            break

        #add pure stragegies to the matrix
        new_matrix = np.zeros( (itr+1, itr+1) )
        qs = np.insert(qs, 0, values=opt_atk)
        alphas_betas = np.insert(alphas_betas, [0], [opt_alpha, opt_beta], axis = 0)
        for x in range(itr):
            for y in range(itr):
                new_matrix[1+x][1+y] = matrix[x][y]
        for i in range( itr+1 ):
            new_matrix[0][i] = utility_function(opt_atk, alphas_betas[i][0], alphas_betas[i][1], xns)
            new_matrix[i][0] = utility_function(qs[i], opt_alpha, opt_beta, xns)
        matrix = new_matrix

        if(printout):
            print(matrix)
    
    #final mixed strategy evaluation
    mixed_attack_distribution = optimal_mixed_strategy(matrix, player='a')
    mixed_defense_distribution = optimal_mixed_strategy(matrix, player='b')
    
    if(printout):
        print("Mixed attack strategy: ", mixed_attack_distribution)
        print("Mixed defense strategy: ", mixed_defense_distribution)
        print("qs")
        print(qs)
        print("alphas, betas:")
        print(alphas_betas)
    
    return opt_alpha, opt_beta, opt_atk, n, qs, mixed_attack_distribution, alphas_betas, mixed_defense_distribution, itr

################ END OF THE GAME SIMULATION, UTIL AND PLOTTING FUNCTIONS ONLY #################  

def plot_attacker_util_function(q1, q2, alpha, beta):
    utils = [] #handy container
    xns = generate_binary_combinations(n)
    for q in np.linspace(q1,q2,100): 
        utils.append(utility_function(q, alpha, beta, xns))
    plt.plot(np.linspace(q1,q2,100), utils)
    plt.show()

def plot_defender_util_function(q, alpha1, alpha2, beta):
    utils = [] #handy container
    xns = generate_binary_combinations(n)
    for alpha in np.linspace(alpha1, alpha2, 100): 
        utils.append(-utility_function(q, alpha, beta, xns))
    plt.plot(np.linspace(alpha1, alpha2, 100), utils)
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
    plt.title("Best responses")
    plt.xlabel("n")
    plt.ylabel("threshold")
    plt.show()

def plot_optimal_distributions():
    _,_,_, n, qs, q_distribution, ab, ab_distribution,_ = game(printout=False)
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
    #alpha and betas
    ab = np.round(ab, decimals=2)
    ab_reduced, ab_sum_indices = np.unique(ab, return_inverse=True, axis=0)
    ab_distribution_reduced = []
    #parse only strategies with reasonable probability to be played, that is >1e-4
    for i in range(len(ab_reduced)):
        temp = np.sum(ab_distribution, where=(ab_sum_indices==i) )
        if temp > 0:
            ab_distribution_reduced.append(temp)
    x_axis = []
    alphas = []
    betas = []
    for i,ab in enumerate(ab_reduced):
        if ab_distribution_reduced[i] > 1e-4:
            x_axis.append(str(ab[0])+"|"+str(ab[1]))
            alphas.append(ab[0])
            betas.append(ab[1])
    ab_distribution_reduced = np.array(ab_distribution_reduced)[ np.array(ab_distribution_reduced) > 1e-4 ]

    print(ab_distribution_reduced)
    #2D defender plot
    plt.title("Optimal defense distribution, n="+str(n))
    plt.xlabel("alpha, beta values")
    plt.ylabel("%")
    plt.bar(x_axis, ab_distribution_reduced, width=0.5, bottom=None,  align='center')
    plt.show()
    #3D defender plot
    fig = plt.figure()
    ax = fig.gca(projection = '3d')
    dx = 0.01 * np.ones( len(alphas) )
    dy = 0.1 * np.ones( len(alphas) )
    ax.set_xlabel('Alpha')
    ax.set_ylabel('Beta')
    ax.bar3d(alphas, betas, np.zeros( len(alphas) ), dx, dy, ab_distribution_reduced, color = 'b')
    plt.show()

def plot_phi(alpha, beta):
    xns = generate_binary_combinations(n)
    values = []
    for xn in xns:
        decision = phi(xn, alpha, beta)
        values.append( decision )
        print(xn.ones, ": ", decision)
    plt.plot(range(n+1), values, 'x')
    plt.title("Phi")
    plt.xlabel("n")
    plt.ylabel("value")
    plt.show()

def plot_steps_needed(iters):
    steps = np.zeros(maxitr+1)
    count = 0
    while count  < iters:
        try:
            _, _, _, _, _, _, _,_, itr = game(printout=False)
            steps[itr] += 1
            count += 1
        except:
            print("rerun")
    plt.title("Number of steps needed to converge, n="+str(n))
    plt.xlabel("steps")
    plt.ylabel("occurence")
    plt.bar( range(len(steps)), steps, width=1, bottom=None,  align='edge')
    plt.show()

################################functions for error rate testing, evaluate.py################################################
    #for generating a list of optimal defender thresholds for each n
    #the list is used for real-world error rate measurement
    #(int list) n's -> (float list) alphas, (float list) betas
def generate_optimal_thresholds(ns):
    global n
    alphas = []
    betas = []
    for n_iter in ns:
        n = int(n_iter)
        xns = generate_binary_combinations(n)
        alpha_beta = optimal_defender_move( [0.8], [1], xns)
        alphas.append( alpha_beta[0])
        betas.append( alpha_beta[1])

    return alphas, betas

    #for generating a list of optimal mixed strategies of both players
    #the list is used for real-world error rate measurement
    #(int list) n's -> (float list) thresholds 
def generate_optimal_mixed_strategies(ns):
    global n
    alphas_betas = []
    alpha_beta_distributions = []
    qs = []
    q_distributions = []
    for n_iter in ns:
        n = int(n_iter)
        _, _, _, _, q_s, mixed_attack_distribution, ab_s, mixed_defense_distribution, _ = game(printout=False)
        alphas_betas.append(  ab_s )
        alpha_beta_distributions.append( np.cumsum(mixed_defense_distribution) )
        qs.append(q_s)
        q_distributions.append( np.cumsum(mixed_defense_distribution) )

    return qs, q_distributions, alphas_betas, alpha_beta_distributions
    
    #loop of multiple runs for time-measurement
    #(int) count of DO runs -> None
def time_bench(count):
    for i in range(count):
        game(printout=False)

##########################################################################################################################š

if __name__ == "__main__":
    #game()
    #plot_attacker_util_function(q1, q2, 100, 100)
    #plot_defender_util_function(qstar, 0, 1, 13)
    #plot_best_responses_by_n(50, 70)
    plot_optimal_distributions()
    #plot_phi(103.9, -69.5)
    #plot_steps_needed(100)
    #print( generate_optimal_thresholds( np.linspace(10, 300, 30, dtype = np.int) ) )