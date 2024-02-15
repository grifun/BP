#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.optimize as sp
import matplotlib.pyplot as plt

##################################CONSTANTS####################################

qstar = 0.8 # arg min of cost function
gamma = 1 # scaling factor in the defender's revenue expression 
d = 2 # number of elements in the alphabet set {0;1}
p = 0.5 #H_0
q1 = 0.6 #Leftmost point of H1
q2 = 0.9 #Rightmost point of H1
n = 1 # number of observations
init_matrix_size = 2 #matrix has initial size N x N
maxitr = 3

##################################FUNCTIONS####################################

def c(q):
    return np.abs(q-qstar)

def q_func(q, X):
    Xs = np.sum(X)
    return (q**Xs) * ( (1-q) ** (n - Xs) )

def p_func(X):
    return (2)**(-n)


def phi(X, threshold):
    #print("q_func(qstar, X)/p_func(X) = ",q_func(qstar, X)/p_func(X))
    if (q_func(qstar, X)/p_func(X)) >= threshold: #???
        return 1
    else:
        return 0

def utility_function(q, threshold, X):
    attack_cost = c(q)
    temporary_sum = 0
    for xn in xns:
        phi_X = phi(xn, threshold)
        successful_attack = (1 - phi_X) * q_func(q,xn)
        false_positive = gamma * phi_X * p_func(xn)
        temporary_sum += (successful_attack + false_positive - attack_cost)
    return temporary_sum +1000

# CUSTOM FUNCTIONS #

def mixed_utility_function_deffender(threshold, qs, distribution, X):
    temporary_sum = 0
    for i, q in enumerate(qs):
        temporary_sum += distribution[i] * utility_function(q, threshold, X)
    return temporary_sum
    
def optimal_deffender_move(qs, distribution, X):
    # function, >, <, (args), tolerance, maxiter 
    ret = sp.fminbound(mixed_utility_function_deffender, 0, 2, (qs, distribution, X))
    return ret

def mixed_utility_function_attacker(q, thresholds, distribution, X):
    temporary_sum = 0
    for i, threshold in enumerate(thresholds):
        temporary_sum += distribution[i] * utility_function(q, threshold, X)
    return -temporary_sum
    
def optimal_attacker_move(qs, distribution, X):
    # function, >, <, (args), tolerance, maxiter 
    ret = sp.fminbound(mixed_utility_function_attacker, q1, q2, (thresholds, distribution, X))
    return ret

def optimal_mixed_strategy(matrix, player = 'a'):
    #make all elements of matrix > 0
    #matrix = matrix + np.abs(np.amin(matrix))+1
    height, width = matrix.shape
    if player == 'a':
        #print("Alice")
        ret = sp.linprog( np.array(np.ones(height)), A_ub = -matrix.transpose(), b_ub = -np.ones(width), options = {'maxiter':10000000} )
    else:
        #print("Bob")
        ret = sp.linprog( -np.array(np.ones(width)), A_ub = matrix, b_ub = np.ones(height), options = {'maxiter':10000000} )
    if ret['success'] != True:
        print("DID NOT FIND EQUILIBRIUM!")
        exit()
    val = np.abs( 1/ret['fun'] )
    print("1 / util_gain = ", val)
    x = ret['x'] * val
    #print("optimal distribution = ", x)
    return x

def generateBinaryCombinations(n, arr, i):  
    if i == n:
        xns.append(np.array(arr))
        return

    arr[i] = 0
    generateBinaryCombinations(n, arr, i + 1)  
  
    arr[i] = 1
    generateBinaryCombinations(n, arr, i + 1)
    
####################################MAIN#######################################

xns = []

if __name__ == "__main__":
    generateBinaryCombinations(n, [None] * n , 0)
    xns = np.array(xns)
    #print(xns)
    
    #attack pure strategies
    qs = np.random.uniform(low = q1, high = q2, size=init_matrix_size)
    #pure defense strategies
    thresholds = np.random.uniform(low = 0, high = 2, size=init_matrix_size)
    print("thresholds:")
    print(thresholds)
    #init starting game matrix
    matrix = np.zeros((init_matrix_size, init_matrix_size))    
    for i1, q in enumerate(qs):
        for i2, threshold in enumerate(thresholds):
            matrix[i1][i2] = utility_function(q, threshold, xns)

    print("init matrix is ", matrix)

    itr = 0
    while itr < maxitr:
        itr += 1
        #find best mixed strategy
        mixed_attack_distribution = optimal_mixed_strategy(matrix, player='a')
        print(mixed_attack_distribution)
        mixed_defense_distribution = optimal_mixed_strategy(matrix, player='b')
        print(mixed_defense_distribution)

        #find best pure strategy
        opt_def = optimal_deffender_move(qs, mixed_attack_distribution, xns)
        opt_atk = optimal_attacker_move(thresholds, mixed_defense_distribution, xns)
        print("optimal defense =", opt_def)
        print("optimal attack =", opt_atk)

        #add to the matrix
        matrix = np.insert(matrix, 0, values=0, axis=1)
        matrix = np.insert(matrix, 0, values=0, axis=0)
        qs = np.insert(qs, 0, values=opt_atk)
        thresholds = np.insert(thresholds, 0, values=opt_def)
        for i in range(init_matrix_size+itr):
            matrix[0][i] = utility_function(qs[i], opt_def, xns)
            matrix[i][0] = utility_function(opt_atk, thresholds[i], xns)
        print(matrix)
    