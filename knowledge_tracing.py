

# Bayesian Knowledge Tracing
#
#########################################################################################################
# BKT is used to model the acquisition of a skill or knowledge component (KC).
# We assume that skill acquisiton is binary - the skill is either learned (1) or unlearned (0).
# Once a skill is learned, we assume that it remains learned forever.
# We further assume that learning a skill is governed by the following parameters:
#
#    L0: Initial probability that the skill has been learned
#    T:  Probability that the student will learn the skill at each opportunity to use it.
#    S:  Probability that the student who has learned the skill will give an incorrect answer (slip).
#    G:  Probability that the student who has not learned the skill will give a correct answer (guess).
#
# Given time-series data on whether students correctly answer questions involving this skill,
# we seek to estimate these four parameters. We fit the parameters by minimizing a cost function,
# defined as the mean squared residual between the predicted probability of success and the 
# actual result (0 or 1).
#########################################################################################################

import math
import random
import numpy
import scipy.optimize
import time

start = time.time()

# Configuration:
random.seed(42)
S_max = 0.1                 # Maximum probability of slipping
G_max = 0.3                 # Maximum probability of guessing correctly
STUDENTCOL = 2              # Column containing Student name (counting from zero)
RIGHTCOL = 5                # Column indicating if the student was correct (1 or 0)
KCCOL = 3                   # Column containing the Knowledge Component (KC)
HEADER = True               # Does file contain a header row?
SEP = ","                   # Separator (use '\t' for tab-separated files or ',' for comma-separated files
                            #    Use '\t' for the test data in Ryan Baker's BKT solver
fname = "./data/bigdata-edu-Data-Sets-Asgn4-dataset.csv"    # Name and location of input file
# fname = "./BKT-BruteForce/TestData.txt"

# Read data into arrays
#    Student: Name or ID of student
#    Right:   Integer value (0 or 1) indicating if student response was correct.
# We assume the input file is sorted by KC, then by student.

with open(fname, "rt") as f:
    if HEADER:
        f.readline()
    Student = []
    Right = []
    rows = [line.split(SEP) for line in f.readlines()]
    KCs = sorted(list(set([row[KCCOL] for row in rows])))
f.close()


# Cost function for the BKT model
# Returns the mean squared residual between the predicted probability and the actual result (0 or 1).
def cost(x):
    L0, T, S, G = x
    SS = 0.0
    L_old = L_new = L_cond = 0.0
    for i in xrange(N):
        if i==0 or Student[i] != Student[i-1]:
            L_old = L0
        else:
            L_old = L_new
        PCorr = bound(L_old*(1-S) + (1-L_old)*G, 1e-6, 1-1e-6)
        SS += (Right[i] - PCorr)**2
        L_cond = L_old
        if Right[i] == 1:
            L_cond = L_old*(1-S) / (PCorr)                
        else:
            L_cond = L_old*S / (1-PCorr)
        L_new = L_cond + (1 - L_cond) * T
    return SS/N

# Minimize the cost function using the BFGS algorithm. Requires the numpy and scipy packages.
def mincost():
    L0 = random.random()
    T = random.random()
    S = random.random() * S_max
    G = random.random() * G_max
    x0 = numpy.array((L0, T, S, G))
    result = scipy.optimize.fmin_l_bfgs_b(cost,(L0, T, S, G), approx_grad=1, bounds=((0,1), (0,1), (0,S_max), (0,G_max)))
    return result[0], result[1]

# Force x to a value between a and b. Helper function for the minimize() function.
def bound(x, a, b):
    return max(a, min(x, b))

# Return a random number between -step and +step. Helper function for the minimize() function.
def rnd(step):
    return step*(2*random.random()-1)

# Minimize the cost function using a random walk algorithm. Use mincost instead because it is faster and more accurate.
def minimize():
  
    for m in xrange(40):
        L0 = random.random()
        T = random.random()
        S = random.random() * S_max
        G = random.random() * G_max
    
        value = previous_value = best_value = cost(L0, T, S, G)
        best_params = (L0, T, S, G)

        if m == 0:
            all_time_best_value = value
            all_time_best_params = best_params

        attempts = 0
        for n in xrange(2000):
            step = random.choice((0.05, 0.005, 0.0005))
            l = bound(L0 + rnd(step), 0, 1)
            t = bound(T + rnd(step), 0, 1)
            s = bound(S + rnd(step), 0.0001, S_max)
            g = bound(G + rnd(step), 0.0001, G_max)
            value = cost((l,t,s,g))
            if value < previous_value:
                attempts = 0
                previous_value = value
                L0, T, S, G = l, t, s, g
                if value < best_value:
                    best_value = value
                    best_params = (L0, T, S, G)
                    if value < all_time_best_value:
                        all_time_best_value = value
                        all_time_best_params = best_params
            else:
                attempts += 1
                if attempts > 100:
                    break
    return all_time_best_params, all_time_best_value

# Fit the parameters of the BKT model for each knowledge component in the dataset.
print '\t'.join(['Skill', 'L0', 'T', 'S', 'G', 'MSE'])
for KC in KCs:
    Student = []
    Right = []
    for row in rows:
        if row[KCCOL] == KC:
            Student.append(row[STUDENTCOL])
            Right.append(int(row[RIGHTCOL]))
    N = len(Student)
    (L0, T, S, G), SS = mincost()
#   (L0, T, S, G), SS = minimize()
    print "%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.4f" % (KC, L0, T, S, G, SS)

