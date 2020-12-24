"""## Adaboost Algorithm"""
import math
import numpy as np
from sklearn.model_selection import train_test_split

MAX_WEIGHT = 30


def df_to_points(features, labels):
    np_features = features.to_numpy()
    np_labels = labels.to_numpy()
    points = []
    for f, l in zip(np_features, np_labels):
        p = Point(f[0], f[1], l)
        points.append(p)
    return points


def rules_from_points(points_list):
    new_rules = []
    points = points_list
    for p1 in points:
        for p2 in points:
            if p1.x != p2.x or p1.y != p2.y:
                rule = Rule(p1, p2, "up")
                new_rules.append(rule)
                rule = Rule(p1, p2, "down")
                new_rules.append(rule)
        points.remove(p1)
    return new_rules

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


def voting(best_rules, k, weighted_point):
    aihix_sum = 0
    for i in range(k):
        aihix_sum += best_rules[i]["weight"] * best_rules[i]["rule"].classify(weighted_point["point"])
    return aihix_sum


def get_empirical_error_on_test(best_rules, k, test_points):
    pass


def get_true_error_on_training(best_rules, k, train_points):
    pass


def run(features_df, labels_df):
    # split the data set randomly
    train_x, test_x, train_y, test_y = train_test_split(features_df, labels_df, test_size=0.5)
    train_points = df_to_points(train_x, train_y)
    test_points = df_to_points(test_x, test_y)
    all_possible_rules = rules_from_points(train_points)

    # Initialize point weights 𝐷_𝑡 (𝑥_𝑖 )=1/𝑛
    initial_point_weight = 1 / len(train_points)

    # list of dictionaries {point, weight}
    weighted_points = [{"point": p, "weight": initial_point_weight} for p in train_points]

    # list of dictionaries {rule, error, weight}
    w_e_rules = [{"rule": rule, "error": 0, "weight": 0} for rule in all_possible_rules]

    best_rules = []

    k = 8  # number of iterations
    z = 0  # sum the points weights
    for i in range(k):
        minimal_error_rule = w_e_rules[0]
        for rule in w_e_rules:
            for p in weighted_points:
                # return 1 if wrong, else 0
                if not rule["rule"].classify_is_correct(p["point"]):
                    rule["error"] += p["weight"]

            if rule["error"] < minimal_error_rule["weight"]:
                minimal_error_rule = rule
        # TODO: is the second rule necessary? if the mana is 0 its ok I think
        # update the weight of the minimal error rule and save it to best rules
        if (minimal_error_rule["error"]) != 0 and ((1 - minimal_error_rule["error"]) /
                                                   minimal_error_rule["error"] > 0):

            minimal_error_rule["weight"] = (1 / 2) * np.log((1 - minimal_error_rule["error"]) /
                                                            minimal_error_rule["error"])
        else:
            minimal_error_rule["weight"] = MAX_WEIGHT  # TODO: risky
        best_rules.append(minimal_error_rule)

        # update all points weights
        for p in weighted_points:
            z += p["weight"]
        for p in weighted_points:
            p["weight"] = (1 / z) * p["weight"] * math.pow(math.e, (-minimal_error_rule["weight"] *
                                                                    minimal_error_rule["rule"].classify(p["point"]) *
                                                                    p["point"].type))

    # at this point we have the list of 8 best rules after one adaboost run
    # TODO: this function should return list of 8 errors:
    #  first error - first rule error on test set
    #   second error - the error of two first rules on test set
    #   third error - the error of 3 first rules on test set
    #   etc..
    #   do the computing here
    #
    hkx_stats = [{"empirical_error_on_test": 0, "true_error_on_training": 0} for k in range(8)]
    for k in range(8):
        hkx_stats[k]["empirical_error_on_test"] = get_empirical_error_on_test(best_rules, k, test_points)
        hkx_stats[k]["true_error_on_training"] = get_true_error_on_training(best_rules, k, train_points)

    # returning stats of one adaboost run
    # TODO: at the main - run 100 adaboost runs. sum all stats of each rules combination.
    # after 100 runs, divide all stats by 100. and wev got our results!
    # do it for both data sets
    ##
    return hkx_stats


# def run(points, adaboost_rules):
#     # Initialize point weights 𝐷_𝑡 (𝑥_𝑖 )=1/𝑛
#     initial_point_weight = 1 / len(points)
#
#     # list of lists [point, weight]
#     weighted_points = [{"point": p, "weight": initial_point_weight} for p in points]
#
#     # list of lists [rule, error, weight]
#     w_e_rules = [{"rule": rule, "error": 0, "weight": 0} for rule in adaboost_rules]
#
#     best_rules = []
#
#     k = 8  # number of iterations
#     z = 0  # sum the points weights
#     for i in range(k):
#         minimal_error_rule = w_e_rules[0]
#         for rule in w_e_rules:
#             for p in weighted_points:
#                 # return 1 if wrong, else 0
#                 if not rule["rule"].classify_is_correct(p["point"]):
#                     rule["error"] += p["weight"]
#
#             if rule["error"] < minimal_error_rule["weight"]:
#                 minimal_error_rule = rule
#
#         # update the weight of the minimal error rule and save it to best rules
#         if (minimal_error_rule["error"]) != 0 and ((1 - minimal_error_rule["error"]) /
#                                                    minimal_error_rule["error"] > 0):
#
#             minimal_error_rule["weight"] = (1 / 2) * np.log((1 - minimal_error_rule["error"]) /
#                                                             minimal_error_rule["error"])
#         else:
#             minimal_error_rule["weight"] = MAX_WEIGHT  # TODO: risky
#         best_rules.append(minimal_error_rule)
#
#         # update all points weights
#         for p in weighted_points:
#             z += p["weight"]
#         for p in weighted_points:
#             p["weight"] = (1 / z) * p["weight"] * math.pow(math.e, (-minimal_error_rule["weight"] *
#                                                                     minimal_error_rule["rule"].classify(p["point"]) *
#                                                                     p["point"].type))
#     # TODO: this function should return list of 8 errors:
#     #  first error - first rule error
#     #   second error - the error of two first rules
#     #   third error - the error of 3 first rules
#     #   etc..
#     ##
#     return best_rules
