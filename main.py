# using python 3.8
# 311148902
# 311300784

import pandas as pd
import Adaboost



# TODO: extract data from files
# TODO: for each dataset:
#   run adaboost 100 times
#       split dataset to train and test
#       run adaboost on the test set and get 8 best rules and their weights
#       do some calculations on the 8 best rules:
#           create a list
#           for k in range(8):
#               compute the empirical error of the function Hk on the test set
#               compute the true error of Hk on the training set
#

def main():
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

    # splitting iris to features and labels
    iris_y = iris.iris_type
    # print(iris_y.shape)
    iris_x = iris.drop(labels="iris_type", axis=1)

    # TODO: separate body_temp to features and labels

    ################## at this point iris and body_temp should be clean and separated to features and labels ##################

    #single adaboost run.
    # input: features panda dataframe, labels panda dataframe
    # output: list of 8 best rules with errors and weights
    sum_of_8_best_complicated_rules_stats = \
        [{"empirical_error_on_test": 0.0, "true_error_on_training": 0.0} for i in range(8)]
    for i in range(100):
        curr_run = Adaboost.run(iris_x, iris_y)
        for index, rule in enumerate(sum_of_8_best_complicated_rules_stats):
            rule["empirical_error_on_test"] += curr_run[index]["empirical_error_on_test"]
            rule["true_error_on_training"] += curr_run[index]["true_error_on_training"]
        ################ debugging ################
        # print("run #", i, "stats:", curr_run)

    avg_errors_of_8_best = [{"empirical_error_on_test": 0.0, "true_error_on_training": 0.0} for i in range(8)]
    for index, rule in enumerate(avg_errors_of_8_best):
        rule["empirical_error_on_test"] = sum_of_8_best_complicated_rules_stats[index]["empirical_error_on_test"] / 100
        rule["true_error_on_training"] = sum_of_8_best_complicated_rules_stats[index]["true_error_on_training"] / 100
        print("Rule #", index, "info:\n"
              , "\tempirical_error_on_test -", rule["empirical_error_on_test"]
              , "\ttrue_error_on_training -", rule["true_error_on_training"])





    ######### Tests ########
    # # 50% training and 50% test
    # iris_X_train, iris_X_test, iris_Y_train, iris_Y_test = train_test_split(iris_x, iris_y, test_size=0.5)
    #
    # # splitting each of the sets to x (features) and y (label)
    # """## Iris"""
    #
    # # convert iris dataset to points
    #
    # print("iris_X_train.shape:")
    # print(iris_X_train.shape)
    # print("iris_Y_train.shape:")
    # print(iris_Y_train.shape)
    # print("iris x train head:")
    # print(iris_X_train.head)
    # print("iris y train head:")
    # print(iris_Y_train.head)
    #
    # iris_train_points = ds_to_points(iris_X_train, iris_Y_train)
    # print("number of iris_train_points:", len(iris_train_points))
    # print("points list:")
    # for p in iris_train_points:
    #     p.print_p()
    # rules = rules_from_points(iris_train_points)
    # print("number of rules:", len(rules))
    # for r in rules:
    #     r.print_r()
    #
    # adaboost_result = Adaboost.run(iris_train_points, rules)
    #
    # print("\n Results:")
    #
    # for r in adaboost_result:
    #     r["rule"].print_r()
    #     print(r["weight"])
    #


if __name__ == "__main__":
    main()


