#!/usr/bin/python
# -*- coding: utf-8 -*-

#developed for Python 3.8
#author - Tomas Kasl, FEL CVUT

import numpy as np
import scipy.optimize as sp
import matplotlib.pyplot as plt
import math

      #the class representing a vector of observations X^n
class xn():
    def __init__(self, n, ones):
        self.n = n
        self.ones = ones
        self.zeros = n-ones
        self.count = n_choose_k(n, ones)
    
    #|  n  |
    #|  k  |
    #(int) n, (int) k -> (int) binomial number
def n_choose_k(n, k):
    return math.factorial(n) / ( math.factorial(k)*math.factorial(n-k) )
     
    #q(x^n)
    #(float) q, (bin. vector) xn -> (float) q-likelihood
def q_func(q, xn):
    return (q**xn.ones) * ( (1-q) ** (xn.zeros) )

    #p(x^n)
    #(bin. vector) xn -> (float) p-likelihood
def p_func(xn): 
    return (2)**(-xn.n)

    #LP for solving mutual optimal mixed strategies, using the inner-point method
    # 'a' stands for the attacker
    #(float matrix) payoff_matrix -> (float list) strategy_distribution
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
        ret = sp.linprog( -function_vector, -boundary_matrix, np.zeros(height), eq_matrix, np.array([1]), options={'autoscale': True, 'cholesky': False, 'sym_pos':False, 'maxiter':1e10})
    else:             #minimizing player
        ret = sp.linprog( function_vector, boundary_matrix, np.zeros(height), eq_matrix, np.array([1]), options={'disp': False, 'cholesky': False, 'sym_pos':False, 'maxiter':1e10})
    if ret['success'] is not True:
        print("DID NOT FIND EQUILIBRIUM!")
        exit()
    x = ret['x'][1:]
    return x

    #generates an array of all possible vectors of length n
    #represented by the class xns above
def generate_binary_combinations(n):
    xns = []
    for i in range(n+1):
        xns.append(xn(n, i))
    return xns