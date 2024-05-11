from decision_tree import DecisionTree
import numpy as np
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class random_forest:
    def __init__(self, number_trees, max_depth, min_sample_split, n_attributes=None):
        self.number_trees = number_trees
        self.min_samples = min_sample_split
        self.n_attributes = n_attributes
        self.max_depth = max_depth
        self.trees = []
         
    def fit(self, x, y):
        self.trees = []
        for _ in range(self.number_trees):
            decision_tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples, n_attributes=self.n_attributes)
            x_sample, y_sample = self.bootstrap_samples(x, y)
            decision_tree.fit(x_sample, y_sample)
            self.trees.append(decision_tree)
    
    def predict(self, x):
        predictions = np.array([tree.predict(x) for tree in self.trees])
        tree_predictions = np.swapaxes(predictions, 0, 1)
        predictions = np.array([self.most_common_label(pred) for pred in tree_predictions])
        return predictions


    def most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common     
    

    def bootstrap_samples(self, x, y):
        number_samples = x.shape[0]
        index = np.random.choice(number_samples, number_samples, replace=True)
        return x[index], y[index]
    
    
    

# manual calculations of accuracy
def calculate_accuracy(y_true, y_pred):
    correct_count = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    accuracy = correct_count / len(y_true)
    return accuracy

def create_confusion_matrix(y_true, y_pred, class_labels):
    # Create a matrix of zeros with dimensions (n_classes, n_classes)
    n_classes = len(class_labels)
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    
    # Populate the confusion matrix
    for actual, predicted in zip(y_true, y_pred):
        matrix[actual][predicted] += 1
    
    return matrix

def compute_learning_curve(X_train, y_train, X_eval, y_eval, increments=5):
    train_sizes = np.linspace(0.1, 1.0, increments)
    train_scores = []

    for size in train_sizes:
        size = int(len(X_train) * size)
        indices = np.random.permutation(len(X_train))[:size]
        X_train_sample = X_train.iloc[indices]
        y_train_sample = y_train.iloc[indices]

        forest = random_forest(number_trees=10, max_depth=10, min_sample_split=2)
        forest.fit(X_train_sample.values, y_train_sample.values)

        y_pred_eval = forest.predict(X_eval.values)
        accuracy = calculate_accuracy(y_eval.values, y_pred_eval)
        train_scores.append(accuracy)

    return train_sizes, train_scores

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

# Fit the random forest model
forest = random_forest(number_trees=10, max_depth=10, min_sample_split=2)
forest.fit(X_train.values, y_train.values)

# Predict on the evaluation set
y_pred_eval = forest.predict(X_eval.values)
eval_accuracy = calculate_accuracy(y_eval.values, y_pred_eval)
print("Evaluation Accuracy:", eval_accuracy)

confusion_matrix_eval = create_confusion_matrix(y_eval.values, y_pred_eval, unique_labels)


plt.figure(figsize=(10, 7))
sns.heatmap(confusion_matrix_eval, annot=True, fmt="d", cmap="Blues", xticklabels=unique_labels, yticklabels=unique_labels)
plt.title('Confusion Matrix for Evaluation Dataset')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig("confusion_matrix.png")
plt.show()

# Call the function to get the training sizes and corresponding scores
train_sizes, train_scores = compute_learning_curve(X_train, y_train, X_eval, y_eval, increments=10)

# Plotting the learning curve
plt.figure(figsize=(10, 6))
plt.plot(train_sizes * len(X_train), train_scores, marker='o', linestyle='-', color='b', label='Evaluation Accuracy')
plt.title('Learning Curve')
plt.xlabel('Training Size')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.grid(True)
plt.savefig("learning_curve.png")
plt.show()

# now, predict on data set that has no mood column in it

data_path_pred = "spotify_dataset.csv"

column_names = ["track_id", "artists", "album_name", "track_name", "popularity", "duration_ms",	"explicit", "danceability", "energy", "key", "loudness", "mode", "speechiness", "acousticness", "instrumentalness", "liveness",	"valence", "tempo", "time_signature", "track_genre"]

data_pred = pd.read_csv(data_path_pred, header=None, names=column_names)

# Drop the first row
data_pred = data_pred.drop(data.index[0]).reset_index(drop=True)

# Convert necessary columns to numeric
numeric_columns = ["danceability", "energy", "key", "loudness", "speechiness", 
                   "acousticness", "instrumentalness", "liveness", "valence", "tempo"]
for col in numeric_columns:
    data_pred[col] = pd.to_numeric(data_pred[col], errors='coerce')
    

# Shuffle the dataset
# data = data.sample(frac=1).reset_index(drop=True)

# Select the required columns
sel_pred_data = data_pred[["danceability", "energy", "key", "loudness", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo"]]

new_moods = forest.predict(sel_pred_data.values)
decoded_moods = unique_labels[new_moods]
print(decoded_moods)

data_pred['predicted_mood'] = new_moods

# print(data["predicted_mood"])



