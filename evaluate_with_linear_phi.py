import numpy as np
import scipy.optimize as sp
import matplotlib.pyplot as plt
import math

#######################################################

ns = [10,  20,  30,  40,  50,  60,  70,  80,  90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300]
qs  = [0.7662616349258525, 0.7631450537999663, 0.7656514589163643, 0.7694801941001614, 0.7682678517821693, 0.7439093080089877, 0.7486501147401425, 0.7505501253082975, 0.7505821889355281, 0.7507813374856765, 0.7455856741356164, 0.7479425826140708, 0.7399698299274963, 0.7420616791176252, 0.7419748444498815, 0.7420126898108178, 0.7417433095755449, 0.741558144607578, 0.7414169515938258, 0.7413433238760762, 0.7412759832289677, 0.7409011545313147, 0.7404385846574717, 0.740114297228859, 0.7400739009715794, 0.7400487282100557, 0.7370973621273673, 0.7355494267116437, 0.7353527452920333, 0.735137153129312]
alphas_betas = [([[1.11, 0],       [1.25, 0],       [1.11, 0],       [1.11, 0],       [0.259, 0.07]]), ([[1.21, 0],       [1.25, 0],       [1.33, 0],       [1.18, 0],       [1.11, 0],       [0.07, 0.845]]), ([[1.22, 0],       [1.25, 0],       [1.3, 0],       [1.2, 0],       [1.2, 0],       [0.953, 0.642]]), ([[1.21, 0],       [1.25, 0],       [1.27, 0],       [1.33, 0],       [1.21, 0],       [1.21, 0],       [0.023, 0.31]]), ([[1.23, 0],       [1.22, 0],       [1.22, 0],       [1.25, 0],       [0.259, 0.631]]), ([[1.28, 0],       [1.25, 0],       [1.3, 0],       [1.36, 0],       [1.22, 0],       [1.18, 0],       [0.653, 0.67]]), ([[1.26, 0],       [1.25, 0],       [1.27, 0],       [1.3, 0],       [1.36, 0],       [1.23, 0],       [1.11, 0],       [0.786, 0.085]]), ([[1.25, 0],       [1.27, 0],       [1.28, 0],       [1.31, 0],       [1.36, 0],       [1.23, 0],       [1.25, 0],       [0.85, 0.361]]), ([[1.27, 0],       [1.25, 0],       [1.22, 0],       [1.29, 0],       [0.302, 0.038]]), ([[1.27, 0],       [1.25, 0],       [1.22, 0],       [1.28, 0],       [0.339, 0.445]]), ([[1.26, 0],       [1.28, 0],       [1.26, 0],       [1.25, 0],       [1.29, 0],       [1.37, 0],       [1.22, 0],       [1.21, 0],       [0.34, 0.581]]), ([[1.26, 0],       [1.27, 0],       [1.28, 0],       [1.26, 0],       [1.3, 0],       [1.38, 0],       [1.22, 0],       [1.19, 0],       [0.262, 0.945]]), ([[1.27, 0],       [1.29, 0],       [1.27, 0],       [1.31, 0],       [1.38, 0],       [1.23, 0],       [1.11, 0],       [0.642, 0.398]]), ([[1.27, 0],       [1.28, 0],       [1.29, 0],       [1.27, 0],       [1.31, 0],       [1.39, 0],       [1.23, 0],       [1.16, 0],       [0.085, 0.311]]), ([[1.27, 0],       [1.28, 0],       [1.26, 0],       [1.3, 0],       [1.38, 0],       [1.23, 0],       [1.17, 0],       [0.172, 0.387]]), ([[1.28, 0],       [1.27, 0],       [1.28, 0],       [1.26, 0],       [1.3, 0],       [1.38, 0],       [1.23, 0],       [1.13, 0],       [0.669, 0.717]]), ([[1.28, 0],       [1.27, 0],       [1.26, 0],       [1.28, 0],       [1.23, 0],       [1.32, 0],       [0.317, 0.083]]), ([[1.28, 0],       [1.27, 0],       [1.28, 0],       [1.26, 0],       [1.3, 0],       [1.23, 0],       [1.37, 0],       [0.956, 0.442]]), ([[1.28, 0],       [1.27, 0],       [1.26, 0],       [1.28, 0],       [1.23, 0],       [1.34, 0],       [0.445, 0.132]]), ([[1.27, 0],       [1.28, 0],       [1.29, 0],       [1.27, 0],       [1.31, 0],       [1.39, 0],       [1.23, 0],       [1.18, 0],       [0.158, 0.569]]), ([[1.27, 0],       [1.28, 0],       [1.27, 0],       [1.29, 0],       [1.33, 0],       [1.39, 0],       [1.24, 0],       [1.26, 0],       [0.094, 0.002]]), ([[1.28, 0],       [1.27, 0],       [1.29, 0],       [1.33, 0],       [1.39, 0],       [1.24, 0],       [1.25, 0],       [0.998, 0.914]]), ([[1.28, 0],       [1.29, 0],       [1.27, 0],       [1.31, 0],       [1.39, 0],       [1.24, 0],       [1.21, 0],       [0.033, 0.023]]), ([[1.28, 0],       [1.27, 0],       [1.26, 0],       [1.28, 0],       [1.33, 0],       [1.4, 0],       [1.24, 0],       [1.25, 0],       [0.287, 0.706]]), ([[1.28, 0],       [1.27, 0],       [1.26, 0],       [1.28, 0],       [1.24, 0],       [1.32, 0],       [0.122, 0.207]]), ([[1.27, 0],       [1.28, 0],       [1.29, 0],       [1.27, 0],       [1.31, 0],       [1.39, 0],       [1.24, 0],       [1.21, 0],       [0.523, 0.806]]), ([[1.28, 0],       [1.27, 0],       [1.3, 0],       [1.27, 0],       [1.26, 0],       [1.24, 0],       [1.24, 0],       [1.28, 0],       [0.666, 0.134]]), ([[1.28, 0],       [1.29, 0],       [1.27, 0],       [1.31, 0],       [1.39, 0],       [1.24, 0],       [1.18, 0],       [0.272, 0.622]]), ([[1.28, 0],       [1.29, 0],       [1.27, 0],       [1.31, 0],       [1.39, 0],       [1.24, 0],       [1.22, 0],       [0.617, 0.718]]), ([[1.28, 0],       [1.29, 0],       [1.3, 0],       [1.28, 0],       [1.32, 0],       [1.4, 0],       [1.24, 0],       [1.24, 0],       [0.802, 0.566]])]
alpha_beta_distributions = [([0.283954036, 0.432091927, 0.716045963, 0.999999999, 1]), ([0.0879588709, 0.280832937, 0.280833222, 0.999999683, 0.999999992, 1]), ([0.000118274114, 0.420726, 0.420767562, 0.71038328, 0.999998998, 1.00000032]), ([0.125391418, 0.74921716, 0.749217164, 0.749217164, 0.874608582, 1, 1]), ([0.0593367107, 0.123725172, 0.188113633, 0.999998452, 1.00000016]), ([0.051441602, 0.999999966, 0.999999969, 0.999999974, 0.999999994, 1, 1]), ([0.0596770258, 0.735308276, 0.999780879, 0.99984006, 0.999852022, 0.999991358, 0.999998632, 1.0000002]), ([0.262883126, 0.737116498, 0.73711679, 0.73711679, 0.737116792, 0.737116873, 0.999999999, 0.999999999]), ([0.614223878, 0.999278889, 0.999279615, 0.999996241, 1.00000035]), ([0.736200179, 0.999998378, 0.999998693, 0.999999984, 0.999999998]), ([0.332774894, 0.667166425, 0.999941319, 0.999978226, 0.999998021, 0.999998549, 0.999999319, 0.999999994, 1.00000002]), ([0.0618810765, 0.938078715, 0.938083132, 0.999964209, 0.999980739, 0.999982883, 0.999995866, 0.99999993, 1.00000009]), ([0.492419654, 0.507579813, 0.999999467, 0.999999503, 0.999999584, 0.999999974, 0.999999996, 1]), ([0.420416444, 0.57957482, 0.579575316, 0.999991761, 0.999994297, 0.999994515, 0.999999366, 0.999999985, 1.00000002]), ([0.726082298, 0.999605626, 0.999856155, 0.999991452, 0.999995931, 0.99999855, 1, 1.00000003]), ([0.188377983, 0.811621468, 0.999999451, 0.999999843, 0.999999987, 0.999999992, 0.999999996, 0.999999998, 0.999999998]), ([0.228539055, 0.771460766, 0.77146078, 0.999999835, 0.999999967, 0.999999999, 1]), ([0.265755437, 0.733978361, 0.999733798, 0.999909933, 0.999992832, 0.999996234, 0.999999978, 1.00000003]), ([0.300578218, 0.699189612, 0.699293279, 0.999871497, 0.999942473, 0.999998992, 1.00000021]), ([0.166217236, 0.833773419, 0.833775025, 0.999992261, 0.999992368, 0.999992692, 0.999998853, 0.999999983, 1.00000003]), ([0.13506222, 0.864893236, 0.999955456, 0.999958256, 0.99995972, 0.999960029, 0.999962189, 0.999999996, 1.00000001]), ([0.782964187, 0.999999148, 0.999999222, 0.999999332, 0.999999354, 0.99999963, 0.999999999, 1]), ([0.829705684, 0.829978931, 0.999830917, 0.99983117, 0.999837749, 0.999955714, 0.999999527, 1.00000033]), ([0.437358058, 0.562641375, 0.562641473, 0.999999531, 0.999999553, 0.999999559, 0.999999635, 0.999999999, 1]), ([0.45838447, 0.541425098, 0.541582685, 0.999967155, 0.99999231, 0.999999932, 1.00000005]), ([0.0215118124, 0.978281036, 0.978386112, 0.999897924, 0.999962957, 0.999965385, 0.999993019, 0.999999949, 1.00000005]), ([0.49697155, 0.499873979, 0.499874354, 0.502776783, 0.503018535, 0.503023243, 0.503027951, 0.999999501, 1.00000019]), ([0.977802815, 0.999948828, 0.999993673, 0.999997189, 0.999997901, 0.999999688, 0.999999986, 1.00000001]), ([0.953096676, 0.998219865, 0.998629924, 0.999854366, 0.999876636, 0.999952382, 0.999999578, 1.00000028]), ([0.463277333, 0.535735953, 0.536682909, 0.999960242, 0.999991376, 0.999995907, 0.999997907, 0.999999907, 1.00000015])]
vector_count = 250000
gamma = 1

######################################################


def q_func(q, xn, n): #q(x^n)
    return (q**xn) * ( (1-q) ** (n-xn) )

def p_func(xn, n): #p(x^n)
    return (2)**(-n)

def phi_linear(xn, n, i): #the phi() used in their numerical tests
    ab_distribution = alpha_beta_distributions[i]
    rand_value = np.random.random_sample()
    index = 0
    while(ab_distribution[index] < rand_value):
        index+=1
    
    alpha = alphas_betas[i][index][0]
    beta = alphas_betas[i][index][1]

    #print("chosen alpha,beta = ", str(alpha) + ", " + str(beta))
    value = xn/n
    if alpha * value + beta > 1:
        return 1
    if alpha * value + beta < 0:
        return 0
    else:
        return alpha * value + beta

def generate(n, outer_i, count):
    benign = []
    adversarial = []
    for i in range(count):
        benign.append( sum (np.random.binomial(1, 0.5, n) ) )
        adversarial.append( sum (np.random.binomial(1, qs[outer_i], n)) )
    return benign, adversarial

def evaluate(benign, adversarial, n, i, phi):
    false_positive = 0
    false_negative = 0
    for vec in benign:
        if phi(vec, n, i) == 1:
            false_positive += 1
    for vec in adversarial:
        if phi(vec, n, i) == 0:
            false_negative += 1
    
    return false_positive/len(benign), false_negative/len(adversarial)

        
if __name__ == "__main__":
    error_rates = []
    error_exponents = []
    
    for i, n in enumerate(ns):
        benign, adversarial = generate(n, i, vector_count)
        fp_percentage, fn_percentage = evaluate(benign, adversarial, n, i, phi_linear)
        print( fp_percentage, fn_percentage )
        error_rate = (fp_percentage+gamma*fn_percentage)/(1+gamma)
        error_rates.append( error_rate )
        error_exponents.append( -1/n * np.log(error_rate) )
        
    #print( error_rates )
    plt.plot( ns, error_rates )
    plt.xlabel("n")
    plt.ylabel("error rate")
    plt.show()

    #print( error_exponents )
    plt.plot( ns, error_exponents )
    plt.xlabel("n")
    plt.ylabel("error exponent")
    plt.show()
    