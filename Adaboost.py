# using python 3.8
# 311148902
# 311300784

# Load libraries
import math

import numpy as np
# Import scikit-learn metrics module for accuracy calculation
import pandas as pd
# Import train_test_split function
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', None)

"""## Data preparation"""

# importing body_temperature dataset
body_temperature = pd.read_table("HC_Body_Temperature.txt", sep='    ', names=["temp", "gender", "heart_rate"],
                                 engine='python', encoding="ISO-8859-1")

# reformat gender column: female = -1, male = 1
body_temperature.gender = body_temperature.gender.apply(lambda x: -1 if x == 1 else 1)

# splitting body_temperature to train and test

# splitting each of the sets to x (features) and y (label)

# importing iris dataset
iris = pd.read_csv("iris.data", names=["sepal_length", "sepal_width", "petal_length", "petal_width", "iris_type"],
                   encoding="ISO-8859-1", low_memory=False)

# dropping the undesired data from table. we only interested in 2nd and 3rd columns as features
iris = iris.drop(labels=["sepal_length", "petal_width"], axis=1)

# ignore Iris-setosa rows
iris = iris.loc[iris.iris_type != "Iris-setosa"]

# reformat iris_type column: Iris-versicolor = -1, Iris-virginica = 1
iris.iris_type = iris.iris_type.apply(lambda x: -1 if x == "Iris-versicolor" else 1)
iris.head()

# all datasets are currently pandas.dataframe

# splitting iris to train and test
iris_Y = iris.iris_type
# print(iris_Y.shape)
iris_X = iris.drop(labels="iris_type", axis=1)

# 50% training and 50% test
iris_X_train, iris_X_test, iris_Y_train, iris_Y_test = train_test_split(iris_X, iris_Y, test_size=0.5)
print(iris_X_train.shape)
print(iris_X_test.shape)
print(iris_Y_train.shape)
print(iris_Y_test.shape)
# splitting each of the sets to x (features) and y (label)


"""## Points and rules"""


class Point:
    def __init__(self, x, y, point_type):
        self.x = x
        self.y = y
        self.type = point_type

    def print_p(self):
        print("x:", self.x, ", y:", self.y, ", type:", self.type)

    def to_string(self):
        s = "x:"
        s += str(self.x)
        s += ", y:"
        s += str(self.y)
        s += ", type:"
        s += str(self.type)
        s += ")"
        return s


class Rule:
    def __init__(self, p1, p2, side):
        self.p1 = p1
        self.p2 = p2
        if side == "up" or side == "down":
            self.side = side
        if self.p1.x - self.p2.x == 0:
            self.parallel_to_y = True
            self.m = 0
            self.n = 0
        else:
            self.parallel_to_y = False
            self.m = (p1.y - p2.y) / (p1.x - p2.x)
            self.n = p1.y - (self.m * p1.x)  # n = y - mx

    def classify(self, p):
        if self.side == "up":
            return self.classify_first_direction(p)
        elif self.side == "down":
            return self.classify_second_direction(p)

    def classify_is_correct(self, p):
        if self.side == "up":
            return self.first_classify_is_correct(p)
        elif self.side == "down":
            return self.second_classify_is_correct(p)

    def classify_first_direction(self, p):
        # if the rule is an ordinary line
        # 
        if not self.parallel_to_y:
            y = p.y
            mxn = (self.m * p.x) + self.n
            if y >= mxn:
                return -1
            else:
                return 1

        # if the rule is a line parallel to the Y axis
        else:
            if p.x >= self.p1.x:
                return 1
            else:
                return -1

    def classify_second_direction(self, p):
        return self.classify_first_direction(p) * -1

    def first_classify_is_correct(self, p):
        if self.classify_first_direction(p) == p.type:
            return True
        else:
            return False

    def second_classify_is_correct(self, p):
        if self.classify_second_direction(p) == p.type:
            return True
        else:
            return False

    def print_r(self):
        if self.p1.x - self.p2.x == 0:
            print("x =", self.p1.x, "side:", self.side, "(p1:", self.p1.to_string(), ", p2:", self.p2.to_string())
        else:
            print("y =", self.m, "x +", self.n, "side:", self.side, "(p1:", self.p1.to_string(), ", p2:",
                  self.p2.to_string())


def ds_to_points(features, labels):
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


"""## Adaboost Algorithm"""


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
            minimal_error_rule[2] = 0
        best_rules.append(minimal_error_rule)

        # update all points weights
        for p in weighted_points:
            z += p[1]
        for p in weighted_points:
            p[1] = (1 / z) * p[1] * math.pow(math.e, (-minimal_error_rule[2] *
                                                      minimal_error_rule[0].classify(p[0]) * p[0].type))
    return best_rules


"""## Iris"""

# convert iris dataset to points
iris_train_points = ds_to_points(iris_X_train, iris_Y_train)
print("number of iris_train_points:", len(iris_train_points))
iris_test_points = ds_to_points(iris_X_test, iris_Y_test)
rules = rules_from_points(iris_train_points)
print("number of rules:", len(rules))

########
adaboost_result = run(iris_train_points, rules)

for r in adaboost_result:
    r[0].print_r()

########
#######################################
# looks like there is not Enough rules! there is 50 points so there are 
# 50choose2 combinations of 2 points that make a line.
# 50C2 = 1225
# each line makes 2 rules
# so we expect 2500 rules but we only get 1900
# * are we creating rules from a point to it self?? (solved)

# TODO: check algorithm
#######################################


"""# Test"""
