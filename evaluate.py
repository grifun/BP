#!/usr/bin/python
# -*- coding: utf-8 -*-

#developed for Python 3.8
#author - Tomas Kasl, FEL CVUT

import numpy as np
import scipy.optimize as sp
import matplotlib.pyplot as plt
import math
from util import *
import game_bayes_initial
import game_neyman_pearson
import game_continuous_phi

##################################CONSTANTS####################################

framework = "bayes"
#framework = "bayes-sigmoid"
#framework = "bayes-linear"
#framework = "neyman"

q = 0.8
gamma = 1
vector_count = 250000
printout = True

ns = np.linspace(10, 300, 30, dtype = np.int64)

#LOAD VALUES FROM SOLVED GAMES#
if framework == "bayes":
    thresholds = game_bayes_initial.generate_optimal_thresholds(ns)

elif framework == "bayes-sigmoid":
    sig_alphas, sig_betas = game_continuous_phi.generate_optimal_thresholds(ns)

elif framework == "bayes-linear":
    lin_alphas, lin_betas = game_continuous_phi.generate_optimal_thresholds(ns)

elif framework == "neyman":
    thresholds = game_neyman_pearson.generate_optimal_thresholds(ns)

else:
    print("UNKNOWN TYPE OF framework, ONLY bayes, bayes-sigmoid, and neyman ARE ALLOWED")
    exit()

##################################FUNCTIONS####################################

    #q(x^n)
    #(float) q, (bin. vector) xn, (int) n -> (float) q-likelihood
def q_func(q, xn, n): #q(x^n)
    return (q**xn) * ( (1-q) ** (n-xn) )

    #p(x^n)
    #(bin. vector) xn, (int) n -> (float) p-likelihood
def p_func(xn, n): #p(x^n)
    return (0.5)**(n)

    #the phi used in the numerical test in the initial paper
    #(bin. vector) xn, (int) n, (int) iteration -> (bool) decision
def phi_thr(xn, n, i): 
    if xn >= thresholds[i]:
        return 1
    else:
        return 0

    #the linear phi proposed in the thesis
    #(bin. vector) xn, (int) n, (int) iteration -> (bool) decision
def phi_sigmoid(xn, n, i): 
    alpha = sig_alphas[i]
    beta = sig_betas[i]
    exponent = ( -alpha * q_func(0.8, xn, n)/p_func(xn, n) +beta)
    prob =  1/(1+np.exp(exponent))
    return np.random.binomial(1, prob, 1) 

    #the sigmoid phi proposed in the thesis
    #(bin. vector) xn, (int) n, (int) iteration -> (bool) decision
def phi_linear(xn, n, i):
    alpha = lin_alphas[i]
    beta = lin_betas[i]
    value = xn/n
    if alpha * value + beta > 1:
        return 1
    if alpha * value + beta < 0:
        return 0
    else:
        return np.random.binomial(1, alpha * value + beta, 1) 


    #generates representations of vectors, both benign (~p) and adversarial (~q)
    #the binary vector is represented by integer equal to amount of ones in it = k
    #(int) n, (int) count -> (int list) k's of benign set, (int list) k's of adversarial set
def generate(n, count):
    benign = []
    adversarial = []
    for i in range(count):
        benign.append( sum (np.random.binomial(1, 0.5, n) ) )
        adversarial.append( sum (np.random.binomial(1, q, n) ) )
    return benign, adversarial

########## the same function, only with code for mixed optimal strategies######
    #(int) n, (int) iteration, (int) count -> (int list) k's of benign set, (int list) k's of adversarial set
def generate_mixed(n, i, count):
    q_distribution = q_distributions[i]
    #randomly choose a pure strategy given the distribution
    rand_value = np.random.random_sample()
    index = 0
    while(q_distribution[index] < rand_value):
        index+=1
    
    q = qs[i][index]
    
    benign = []
    adversarial = []
    
    for i in range(count):
        benign.append( sum (np.random.binomial(1, 0.5, n) ) )
        adversarial.append( sum (np.random.binomial(1, q, n)) )
    return benign, adversarial

   #evaluates the error rates of the vector classification, provided with phi and its threshold/arguments
   #(int list) k's of benign set, (int list) k's of adversarial set, (int) n, (int) iteration, (function) phi
   # ->
   #(float) false_positive_error_rate, (float) false_negative_error_rate
def evaluate(benign, adversarial, n, i, phi):
    false_positive = 0
    false_negative = 0
    for vec in adversarial:
        if phi(vec, n, i) == 0:
            false_negative += 1
    for vec in benign:
        if phi(vec, n, i) == 1:
            false_positive += 1
    return false_positive/len(benign), false_negative/len(adversarial)

   #evaluates the error rates of the vector classification under Neyman_Pearson, provided with phi and its threshold/arguments
   #(int list) k's of benign set, (int list) k's of adversarial set, (int) n, (int) iteration, (function) phi
   # ->
   #(float) false_negative_error_rate
def evaluate_neyman(benign, adversarial, n, i, phi):
    false_negative = 0
    for vec in adversarial:
        if phi(vec, n, i) == 0:
            false_negative += 1
    return false_negative/len(adversarial)

##########################################################################################################################š

if __name__ == "__main__":
    error_rates = []
    error_exponents = []
    
    for i, n in enumerate(ns):
        if framework == "bayes":
            benign, adversarial = generate(n, vector_count)
            fp_percentage, fn_percentage = evaluate(benign, adversarial, n, i, phi_thr)
            if(printout):   
                print("for n = ",n, "FP rate =", fp_percentage, ", FN rate = ", fn_percentage )
            error_rate = (fp_percentage+gamma*fn_percentage)/(1+gamma)
        
        elif framework == "bayes-sigmoid":
            benign, adversarial = generate(n, vector_count)
            fp_percentage, fn_percentage = evaluate(benign, adversarial, n, i, phi_sigmoid)
            if(printout):   
                print("for n = ",n, "FP rate =", fp_percentage, ", FN rate = ", fn_percentage )
            error_rate = (fp_percentage+gamma*fn_percentage)/(1+gamma)

        elif framework == "bayes-linear":
            benign, adversarial = generate(n, vector_count)
            fp_percentage, fn_percentage = evaluate(benign, adversarial, n, i, phi_linear)
            if(printout):   
                print("for n = ",n, "FP rate =", fp_percentage, ", FN rate = ", fn_percentage )
            error_rate = (fp_percentage+gamma*fn_percentage)/(1+gamma)
        
        elif framework == "neyman":
            benign, adversarial = generate(n, vector_count)
            error_rate = evaluate_neyman(benign, adversarial, n, i, phi_thr)
            if(printout):
                print("for n = ",n,"error rate = ", error_rate)

        else:
            print("UNKNOWN TYPE OF framework, ONLY bayes, bayes-sigmoid, and neyman ARE ALLOWED")
            exit()

        error_rates.append( error_rate )
        error_exponents.append( -1/n * np.log(error_rate) )

    if(printout):    
        print( error_rates )
        print( error_exponents )
        
    plt.plot( ns, error_rates )
    plt.xlabel("n")
    plt.ylabel("error rate")
    plt.show()

    plt.plot( ns, error_exponents )
    plt.xlabel("n")
    plt.ylabel("error exponent")
    plt.show()
    