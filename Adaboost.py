"""## Adaboost Algorithm"""
import math
import numpy as np
from Point import Point
from Rule import Rule

maximum_weight = 30


# input : points with labels (points list), rules list
# output : weights of rules (list)
#
# Algorithm:
#   1.  Initialize point weights 𝐷_𝑡 (𝑥_𝑖) = 1/𝑛
#   2.  number of iterations k = 8
#   3.  for each rule classify each of the points and save the errors of the rule.
#         if the rule classifies a point correctly - don't add to the error.
#         else - add the point weight to the rule error.
#   4.  at the end of each iteration - save the rule with the minimal error (best rule of the current iteration)
#       set the weight of the rule to 𝛼_𝑡=1/2  ln⁡〖(1−𝜖_𝑡 (ℎ_𝑡))/(𝜖_𝑡 (ℎ_𝑡))〗
#   5.  Update point weights -  𝐷_(𝑡+1) (𝑥_𝑖 )=1/𝑍_𝑡  𝐷_𝑡 (𝑥_𝑖 ) 𝑒_^(−𝛼_𝑡 ℎ_𝑡 (𝑥_𝑖 ) 𝑦_𝑖 )
#   (we need to check what Z_t should be)

#######################################
# looks like there is not Enough rules! there is 50 points so there are
# 50choose2 combinations of 2 points that make a line.
# 50C2 = 1225
# each line makes 2 rules
# so we expect 2500 rules but we only get 1900
# TODO: we need to make sure that a line wont be considered twice (two sides of same line is ok)
#######################################

def run(points, adaboost_rules):
    # Initialize point weights 𝐷_𝑡 (𝑥_𝑖 )=1/𝑛
    initial_point_weight = 1 / len(points)

    # list of lists [point, weight]
    weighted_points = [[p, initial_point_weight] for p in points]

    # list of lists [rule, error, weight]
    w_e_rules = [[rule, 0, 0] for rule in adaboost_rules]

    best_rules = []

    k = 8  # number of iterations
    z = 0  # sum the points weights
    for i in range(k):
        minimal_error_rule = w_e_rules[0]
        for rule in w_e_rules:
            for p in weighted_points:
                # return 1 if wrong, else 0
                if not rule[0].classify_is_correct(p[0]):
                    rule[1] += p[1]

            if rule[1] < minimal_error_rule[1]:
                minimal_error_rule = rule

        # update the weight of the minimal error rule and save it to best rules
        if (minimal_error_rule[1]) != 0:
            minimal_error_rule[2] = (1 / 2) * np.log((1 - minimal_error_rule[1]) / (minimal_error_rule[1]))
        else:
            minimal_error_rule[2] = maximum_weight  # TODO: risky
        best_rules.append(minimal_error_rule)

        # update all points weights
        for p in weighted_points:
            z += p[1]
        for p in weighted_points:
            p[1] = (1 / z) * p[1] * math.pow(math.e, (-minimal_error_rule[2] *
                                                      minimal_error_rule[0].classify(p[0]) * p[0].type))
    return best_rules
