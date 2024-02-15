****Strategic Games in Adversarial Classification Problems****

**bachelor thesis**

Code for solving the game by the Double oracle algorithm 

Files:

 - game_bayes_initial.py - DO implementation of the model (under Bayesian framework), as proposed in the initial paper (1) + functions for plots present in the thesis
 - game_neyman_pearson.py - DO implementation of the model (under Neyman-Pearson framework), + functions for plots present in the thesis
 - game_continuous_phi.py - DO implementation, Bayesian framework, but with continuous (both sigmoid or linear) decision function phi(). + functions for plots
 - util.py - file for utility functions: q(x^n), p(x^n), the x^n class definition and LP code for solving the zero-sum mutual optimal mixed strategies
 - evaluate.py - code for evaluating the classifiers error rate/exponent. Includes code for both generating the samples and classifying them
 

(1) - Nonzero-sum Adversarial Hypothesis Testing Games
Sarath Yasodharan and Patrick Loiseau