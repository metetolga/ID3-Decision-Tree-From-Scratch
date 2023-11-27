## BFS
from collections import Counter, deque
from enum import Enum
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd


class Species(Enum):
    ASTILBE = 0
    BELLFLOWER = 1
    CARNATION = 2
    DAISY = 3
    DANDELION = 4
    IRIS = 5
    ROSE = 6
    SUNFLOWER = 7
    TULIP = 8
    WATER_LILY = 9


class TreeNode:
    def __init__(self, data, feature, value=None, label=None, parent=None, gain=None):
        self.data = data
        self.feature = feature
        self.value = value
        self.label = label
        self.parent = parent
        self.gain = gain
        self.target = "engine_fuel"
        self.is_leaf = False
        self.dist = Counter()
        self.neighbors = list()
        self.prep()

    def prep(self):
        if len(self.data) <= 0:
            return

        for value in self.data[self.target]:
            self.dist[value] += 1

    def __eq__(self, other):
        if isinstance(other, TreeNode):
            return (self.parent.feature == other.parent.feature and
                    self.feature == other.feature and
                    self.value == other.value)
        return False


class DecisionTree:
    def __init__(self, target) -> None:
        self.target = target
        self.tree = None
        self.original_data = None
        self.features = None
        self.__twigs = None

    def fit_transform(self, data):
        self.features = set(data.columns) - {self.target}
        self.original_data = data
        self.tree = self.__traverse_tree(data=data)

    @staticmethod
    def __entropy(target):
        elements, counts = np.unique(target, return_counts=True)
        etr = -np.sum(
            [(counts[i] / np.sum(counts)) * np.log2(counts[i] / np.sum(counts)) for i in range(len(elements))])
        return etr

    @staticmethod
    def __information_gain(data, split, target):
        total_entropy = DecisionTree.__entropy(data[target])
        vals, counts = np.unique(data[split], return_counts=True)
        weighted_entropy = np.sum([(counts[i] / np.sum(counts)) * DecisionTree.__entropy(
            data.where(data[split] == vals[i]).dropna()[target]) for i in range(len(vals))])
        gain = total_entropy - weighted_entropy
        return gain

    def __traverse_tree(self, data) -> TreeNode:
        gains = Counter()
        self.features = set(self.features) - {self.target}
        for ftr in self.features:
            gains[ftr] = DecisionTree.__information_gain(data, ftr, self.target)

        best_feature = gains.most_common(1)[0][0]
        tree_node = TreeNode(data=data, feature=best_feature)
        self.features = set(self.features) - {best_feature}

        que = deque()
        que.append(tree_node)
        while len(que) > 0:
            for value in np.unique(data[que[0].feature]):
                if que[0].is_leaf:
                    continue
                sub_data = que[0].data.where(que[0].data[que[0].feature] == value).dropna()
                sub_tree = self.__make_neighbor(sub_data, que[0].feature, que[0])
                if sub_tree.value is None:
                    sub_tree.value = value
                que[0].neighbors.append(sub_tree)
                que.append(sub_tree)
            que.popleft()

        return tree_node

    def __make_neighbor(self, data, feature, parent) -> TreeNode:
        if len(data) == 0:
            temp = TreeNode(data=data, feature=feature, parent=parent)
            temp.is_leaf = True
            temp.label = np.unique(self.original_data[self.target])[np.argmax(np.unique(self.original_data[self.target], return_counts=True)[1])]
            return temp

        if len(self.features) == 0:
            temp = TreeNode(data=data, feature=feature, parent=parent)
            temp.is_leaf = True
            temp.label = temp.dist.most_common(1)[0][0]
            return temp

        if len(np.unique(data[self.target])) == 1:
            temp = TreeNode(data, feature, np.unique(data[feature])[0], np.unique(data[self.target])[0], parent)
            temp.is_leaf = True
            return temp

        gains = Counter()
        self.features = set(self.features) - {self.target}
        for ftr in self.features:
            gains[ftr] = DecisionTree.__information_gain(data, ftr, self.target)

        best_feature, gain = gains.most_common(1)[0]
        self.features = set(self.features) - {best_feature}
        tree_node = TreeNode(data=data, feature=best_feature, parent=parent)
        tree_node.gain = gain
        return tree_node

    def predict(self, test_data):
        return test_data.apply(self.__single_predict, axis=1)

    def __single_predict(self, test_row):
        temp_node = self.tree
        while not temp_node.is_leaf:
            attr = test_row[temp_node.feature]
            for neighbor in temp_node.neighbors:
                if neighbor.value == attr:
                    temp_node = neighbor
                    break

        if temp_node.is_leaf:
            return temp_node.label

    def prune(self, X_valid, y_valid):
        y_pred_last = self.predict(X_valid)
        last_acc = accuracy_score(y_valid, y_pred_last)
        if not self.__twigs:
            self.__twigs = self.twigs()
        least_gain_node = self.__twigs[0]
        least_parent = least_gain_node.parent
        for i in range(len(least_parent.neighbors)):
            if least_gain_node == least_parent.neighbors[i]:
                least_gain_node.is_leaf = True
                least_gain_node.label = least_gain_node.dist.most_common(1)[0][0]
                break
        y_pred_curr = self.predict(X_valid)
        curr_acc = accuracy_score(y_valid, y_pred_curr)

        print(f"Before prune the {least_gain_node.feature} node, acc: {last_acc}")
        print(f"After prune the {least_gain_node.feature} node, acc: {curr_acc}\n")

        if curr_acc >= last_acc:
            self.__twigs.pop(0)
            least_gain_node.neighbors.clear()
            if len(self.__twigs) != 0:
                self.prune(X_valid, y_valid)
        else:
            least_gain_node.is_leaf = False
            least_gain_node.label = None

    def rules(self):
        if not self.tree:
            print("Call fit_transform() before printing rules")
            return
        temp = self.tree
        self.__get_rules(temp, list())

    def __get_rules(self, tree_node, rules):
        if tree_node.is_leaf:
            for node in rules:
                # node.value is None or
                if node.parent is None:
                    continue
                print(f"{node.parent.feature}: {node.value}")
            print(f"{tree_node.parent.feature}: {tree_node.value}")
            print(f"label: {tree_node.label}\n")
            return
        rules.append(tree_node)
        for neighbor in tree_node.neighbors:
            self.__get_rules(neighbor, rules)
        rules.pop(len(rules) - 1)

    def twigs(self):
        tree_node = self.tree
        temp_twigs = list()
        self.__get_twigs(tree_node, temp_twigs)
        self.__twigs = sorted(temp_twigs, key=lambda twig: twig.gain)
        return self.__twigs

    def __get_twigs(self, tree_node, result):
        if DecisionTree.__is_twig(tree_node):
            result.append(tree_node)
        else:
            for neighbor in tree_node.neighbors:
                self.__get_twigs(neighbor, result)

    @staticmethod
    def __is_twig(tree_node) -> bool:
        if len(tree_node.neighbors) == 0:
            return False

        for neighbor in tree_node.neighbors:
            if not neighbor.is_leaf:
                return False

        return True


# df = pd.DataFrame({
#     'Weather': ['Sunny', 'Sunny', 'Overcast', 'Rainy', 'Rainy', 'Rainy', 'Overcast', 'Sunny', 'Sunny', 'Rainy', 'Sunny',
#                 'Overcast', 'Overcast', 'Rainy'],
#     'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot',
#                     'Mild'],
#     'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal',
#                  'High', 'Normal', 'High'],
#     'Windy': [False, True, False, False, False, True, True, False, False, False, True, True, False, True,],
#     'Play Tennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
# })
# df = pd.DataFrame({
#     'age': ['middle', 'middle', 'young', 'middle', 'senior', 'senior', 'senior', 'middle', 'young', 'young', 'senior',
#                 'young', 'young', 'senior'],
#     'sex': ['f', 'm', 'f', 'f', 'f', 'm', 'm', 'm', 'f', 'm', 'm', 'm', 'f', 'f'],
#     'bp': ['normal', 'high', 'high', 'high', 'normal', 'low', 'low', 'low', 'normal', 'low', 'normal', 'normal', 'high',
#            'normal'],
#     'cholesterol': ['high', 'normal', 'normal', 'normal', 'normal', 'normal', 'high', 'high', 'normal', 'normal',
#                     'normal', 'high', 'high', 'high'],
#     'drug': ['A', 'A', 'B', 'A', 'A', 'A', 'B', 'A', 'B', 'A', 'A', 'A', 'B', 'B']
# })
#
# decision_tree = DecisionTree("drug")
# features = df.columns
# decision_tree.fit_transform(df)
# X_valid = pd.DataFrame({
#     "age": ["middle"],
#     "sex": ["f"],
#     "bp": ["low"],
#     "cholesterol": ["normal"]
# })
# print('-----BEFORE-----')
# decision_tree.rules()
# y_valid = pd.Series(['A'])
# decision_tree.prune(X_valid, y_valid)
# print('-----AFTER-----')
# decision_tree.rules()


# print(decision_tree.predict(test_row))
# decision_tree.rules()
# twigs = decision_tree.twigs()
# for twig in twigs:
#     print(f"{twig.parent.feature}: {twig.value} with {twig.feature}, {len(twig.neighbors)} leaves. Gain {twig.gain}")


