# using python 3.8
# 311148902
# 311300784


# Load libraries

import pandas as pd
from sklearn.model_selection import train_test_split

import Adaboost
from Point import Point
from Rule import Rule

pd.set_option('display.max_columns', None)

"""## Data preparation"""
# importing body_temperature dataset
body_temperature = pd.read_table("HC_Body_Temperature.txt", sep='    ', names=["temp", "gender", "heart_rate"],
                                 engine='python', encoding="ISO-8859-1")

# reformat gender column: female = -1, male = 1
body_temperature.gender = body_temperature.gender.apply(lambda x: -1 if x == 1 else 1)

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


# splitting each of the sets to x (features) and y (label)


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


"""## Iris"""

# convert iris dataset to points

print("iris_X_train.shape:")
print(iris_X_train.shape)
print("iris_Y_train.shape:")
print(iris_Y_train.shape)
print("iris x train head:")
print(iris_X_train.head)
print("iris y train head:")
print(iris_Y_train.head)

# extract data from files
# for each dataset:
#   run adaboost 100 times
#       split dataset to train and test
#       run adaboost on the test set and get 8 best rules and their weights
#       do some calculations on the 8 best rules:
#           create a list
#           for k in range(8):
#               compute the empirical error of the function Hk on the test set
#               compute the true error of Hk on the training set
#
#  #

"""# Test"""
iris_train_points = ds_to_points(iris_X_train, iris_Y_train)
print("number of iris_train_points:", len(iris_train_points))
print("points list:")
for p in iris_train_points:
    p.print_p()
rules = rules_from_points(iris_train_points)
print("number of rules:", len(rules))
for r in rules:
    r.print_r()

adaboost_result = Adaboost.run(iris_train_points, rules)

print("\n Results:")

for r in adaboost_result:
    r[0].print_r()
    print(r[2])
