import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import random

# save the state of a random function
random.seed(1)

array_gini_list = []
gini_score_list = []


# accuracy and predict class
class accuracy_point:
    def __init__(self):
        self.accuracy = None
        self.correct_predict = 0
        self.wrong_predict = 0
        self.predict_list = []
        self.true_list = None

    def predict(self, sample):

        self.true_list = sample[model.target].values
        print(self.true_list)
        for i in sample.iloc:

            # self.true_list.append(i[root.target])
            q = model
            while not q.leaf:

                columns = q.minimum_gini[1]
                base_label = q.minimum_gini[0]

                if int(i[columns]) <= int(base_label):

                    q = q.left_link
                else:

                    q = q.right_link
            total_label_posetive = len([j for j in q.data[q.target] if j == 1])
            total_label_negative = len([j for j in q.data[q.target] if j == 0])
            if total_label_posetive > total_label_negative:
                self.predict_list.append("1")
            else:
                self.predict_list.append("0")
        return self.accuracy_sample()

    def accuracy_sample(self):

        result = zip(self.predict_list, self.true_list)
        for i in result:
            if int(i[0]) == int(i[1]):
                self.correct_predict += 1
            else:
                self.wrong_predict += 1
        self.accuracy = self.correct_predict / (self.correct_predict + self.wrong_predict)


# Decision_Tree class
class Decision_Tree:
    def __init__(self, info, target, min_sample=1, max_depht=-1, depth=0):
        self.data = info  # data
        self.target = target  # target feature
        self.columns = info.columns  # columns name
        self.depth = depth  # current depth
        self.minimum_gini = []  # minimum gini in Node
        self.leaf = False  # leaf check

        self.left_link = None
        self.right_link = None

        self.max_depht = max_depht  # maximum tree depth
        self.min_sample = min_sample  # minimum sample in Node

    # process feature to find minimum gini
    def process_feature(self):
        # check maximum depth and leaf
        if (self.depth > self.max_depht-1 and self.max_depht != -1) or self.is_leaf() == 0:
            self.leaf = True
            return

        global gini_score_list, array_gini_list
        array_gini_list = []
        gini_score_list.clear()
        columns = self.columns.drop(self.target)

        for i in columns:
            self.subset_size(i)
        # data not split by any features
        if len(array_gini_list) == 0:
            self.leaf = True
            return

        # minimum gini in Node
        minimum_gini_index = np.argmin(array_gini_list[:, 2], axis=0)
        self.minimum_gini = array_gini_list[minimum_gini_index]
        self.split_data()

    # Get the size of two Nodes
    def subset_size(self, col):
        global array_gini_list

        for i in range(min(self.data[col]), max(self.data[col]) + 1):
            first = []
            second = []
            for j in range(len(self.data)):
                if self.data[col][j] <= i:
                    first.append(self.data[self.target][j])
                else:
                    second.append(self.data[self.target][j])
            # check minimum sample
            if len(second) > int(self.min_sample) and len(first) > int(self.min_sample):
                self.gini_score(first, second, i, col)

            array_gini_list = np.array(gini_score_list).reshape(-1, 3)

    #Get the lowest gini number
    def gini_score(self, lst_pos, lst_neg, base, col):
        global gini_score_list
        total = 0
        gini_score_list.append(base)
        gini_score_list.append(col)
        for i in lst_pos, lst_neg:
            label_posetive = len([j for j in i if j == 1])
            label_negative = len([j for j in i if j == 0])
            total_label = label_negative + label_posetive
            gini_score_i = (total_label / len(self.data)) * (
                    1 - ((label_posetive / total_label) ** 2 + (label_negative / total_label) ** 2))
            total += gini_score_i
        gini_score_list.append(total)

    def split_data(self):
        first = []
        second = []
        for i in self.data.iloc():

            if int(i[self.minimum_gini[1]]) <= int(self.minimum_gini[0]):
                first.append(i.values)
            else:
                second.append(i.values)
        left_node = Decision_Tree(pd.DataFrame(first, columns=self.columns), target=self.target,
                                  min_sample=self.min_sample,max_depht=self.max_depht, depth=self.depth + 1)
        right_node = Decision_Tree(pd.DataFrame(second, columns=self.columns), self.target, min_sample=self.min_sample,
                                   max_depht=self.max_depht, depth=self.depth + 1)
        self.left_link = left_node
        self.right_link = right_node
        left_node.process_feature()
        right_node.process_feature()

    def is_leaf(self):

        label_posetive = len([j for j in self.data[self.target] if j == 1])
        label_negative = len([j for j in self.data[self.target] if j == 0])
        total_label = label_negative + label_posetive
        gini_score_i = (total_label / len(self.data)) * (
                1 - ((label_posetive / total_label) ** 2 + (label_negative / total_label) ** 2))

        return gini_score_i


# read data
data = pd.read_csv("PCOS.csv")

X = data.drop("PCOS (Y/N)", axis=1)
y = data[["PCOS (Y/N)"]]

# specify train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.025, random_state=1)

# create data train
data_train = pd.concat([X_train, y_train], axis=1)
data_train = [i for i in data_train.values]
data_train = pd.DataFrame(data_train, columns=data.columns)

# create data test
data_test = pd.concat([X_test, y_test], axis=1)
data_test = [i for i in data_test.values]
data_test = pd.DataFrame(data_test, columns=data.columns)

# create model
model = Decision_Tree(data_train, 'PCOS (Y/N)', max_depht=10)
model.process_feature()

# create accuracy class
a = accuracy_point()
a.predict(data_test)
print(a.accuracy)


