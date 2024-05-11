
import numpy as np
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt


class Node:
    def __init__(self, attribute=None, threshold=None, left=None, right=None,*,value=None):
        self.attribute = attribute
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        
    def is_leaf_node(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_attributes=None):
        self.min_samples_split=min_samples_split
        self.max_depth=max_depth
        self.n_attributes=n_attributes
        self.root=None

    def fit(self, Examples, Target):
        Examples = Examples.astype(float)
        Target = Target.astype(int)
        self.n_attributes = Examples.shape[1] if not self.n_attributes else min(Examples.shape[1], self.n_attributes)
        self.root = self._grow_tree(Examples, Target)

    def _best_split(self, Examples, Target, attribute_indices):
        best_gain = -1
        split_attribute, split_threshold = None, None

        for attribute in attribute_indices:
            Examples_column = Examples[:, attribute]
            thresholds = np.unique(Examples_column)

            for threshold in thresholds:
                gain = self._information_gain(Target, Examples_column, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_attribute = attribute
                    split_threshold = threshold

        return split_attribute, split_threshold


    def _grow_tree(self, Examples, Target, depth=0):
        n_samples, n_features = Examples.shape
        n_labels = len(np.unique(Target))

        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split:
            leaf_value = self._most_common_label(Target)
            return Node(value=leaf_value)

        attribute_indices = np.random.choice(n_features, self.n_attributes, replace=False)

        best_attribute, best_threshold = self._best_split(Examples, Target, attribute_indices)

        left_indices, right_indices = self._split(Examples[:, best_attribute], best_threshold)
        left = self._grow_tree(Examples[left_indices, :], Target[left_indices], depth + 1)
        right = self._grow_tree(Examples[right_indices, :], Target[right_indices], depth + 1)
        return Node(attribute=best_attribute, threshold=best_threshold, left=left, right=right)

    def _split(self, X_column, split_thresh):
        left_index = np.argwhere(X_column <= split_thresh).flatten()
        right_index = np.argwhere(X_column > split_thresh).flatten()
        return left_index, right_index


    def _information_gain(self, Target, Examples_column, threshold):
        parent_entropy = self._entropy(Target)

        left_indices, right_indices = self._split(Examples_column, threshold)

        n = len(Target)
        n_left, n_right = len(left_indices), len(right_indices)
        if n_left == 0 or n_right == 0:
            return 0

        e_left, e_right = self._entropy(Target[left_indices]), self._entropy(Target[right_indices])
        child_entropy = (n_left / n) * e_left + (n_right / n) * e_right

        information_gain = parent_entropy - child_entropy
        return information_gain
    

    def _entropy(self, Target):
        Target = Target.astype(int)
        counter = Counter(Target)  
        histogram = np.array([counter[i] for i in range(max(Target) + 1)])  
        probs = histogram/ len(Target)
        return -np.sum([p * np.log(p) for p in probs if p > 0])

    def predict(self, Examples):
        return np.array([self._traverse_tree(example, self.root) for example in Examples])

    def _traverse_tree(self, example, node):
        if node.is_leaf_node():
            return node.value

        if example[node.attribute] <= node.threshold:
            return self._traverse_tree(example, node.left)
        return self._traverse_tree(example, node.right)
    
    def _most_common_label(self, Target):
        counter = Counter(Target)
        value = counter.most_common(1)[0][0]
        return value

# Custom accuracy calculation function
def calculate_accuracy(y_true, y_pred):
    correct_predictions = np.sum(y_true == y_pred)
    accuracy = correct_predictions / len(y_true)
    return accuracy

def evaluate_decision_tree(X_train, y_train, X_test, y_test, max_depths):
    train_accuracies = []
    test_accuracies = []

    for max_depth in max_depths:
        tree = DecisionTree(max_depth=max_depth)
        tree.fit(X_train.to_numpy(), y_train.to_numpy())

        y_train_pred = tree.predict(X_train.to_numpy())
        y_test_pred = tree.predict(X_test.to_numpy())

        train_accuracy = calculate_accuracy(y_train, y_train_pred)
        test_accuracy = calculate_accuracy(y_test, y_test_pred)

        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

    return train_accuracies, test_accuracies


data_path = "data_moods.csv"

column_names = ["name", "album", "artist","id","release_date","popularity","length","danceability","acousticness","energy","instrumentalness","liveness","valence","loudness","speechiness","tempo","key","time_signature","mood"]

data = pd.read_csv(data_path, header=None, names=column_names)

# Drop the first row
data = data.drop(data.index[0]).reset_index(drop=True)


# Convert necessary columns to numeric
numeric_columns = ["danceability", "energy", "key", "loudness", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo"]

for col in numeric_columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')
    
data['mood'], unique_labels = pd.factorize(data['mood'])

# Shuffle the dataset
data = data.sample(frac=1).reset_index(drop=True)

# Select the required columns
sel_data = data[["danceability", "energy", "key", "loudness", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo", "mood"]]

# Splitting the data into training (70%), testing (20%), and evaluation (10%)
n = len(sel_data)
train_end = int(0.7 * n)
test_end = int(0.9 * n)

train_data = sel_data[:train_end]
test_data = sel_data[train_end:test_end]
eval_data = sel_data[test_end:]

# Extract features and target
X_train = train_data.iloc[:, :-1]  # All columns except the last one
y_train = train_data.iloc[:, -1]   # Last column as the target
X_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1]
X_eval = eval_data.iloc[:, :-1]
y_eval = eval_data.iloc[:, -1]

# Evaluate decision tree with different depths
max_depths = list(range(1, 21))
# accuracies = []

train_accuracies, test_accuracies = evaluate_decision_tree(X_train, y_train, X_test, y_test, max_depths)

#for depth in max_depths:
#    clf = DecisionTree(max_depth=depth)
#    clf.fit(X_train, y_train)
#    predictions = clf.predict(X_eval)
#    acc = np.sum(predictions == y_eval) / len(y_eval)
#    accuracies.append(acc)
#

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(max_depths, train_accuracies, label='Train Accuracy', marker='o')
plt.plot(max_depths, test_accuracies, label='Test Accuracy', marker='o')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.title('Decision Tree Accuracy vs Max Depth')
plt.legend()
plt.grid(True)
plt.show()
