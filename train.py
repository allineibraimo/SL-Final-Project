from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import decision_tree
import matplotlib.pyplot as plt

def classify_mood(row):
    if row["valence"] > 0.5 and row["tempo"] > 120:
        if row["energy"] > 0.5:
            if row["danceability"] > 0.5:
                return "Excited"
            return "Happy"
        if row["acousticness"] > 0.3:
            return "Love"
    if row["energy"] < 0.4 and row["tempo"] < 100:
        return "Calm"
    if row["valence"] < 0.5 and row["energy"] < 0.4:
        return "Sad"
    if row["loudness"] > -5:
        if row["speechiness"] > 0.5:
            return "Anger"
    return "Unknown"


data_path = "spotify_dataset.csv"

column_names = ["track_id", "artists", "album_name", "track_name", "popularity", "duration_ms",	"explicit", "danceability", "energy", "key", "loudness", "mode", "speechiness", "acousticness", "instrumentalness", "liveness",	"valence", "tempo", "time_signature", "track_genre"]

data = pd.read_csv(data_path, header=None, names=column_names)

# Drop the first row
data = data.drop(data.index[0]).reset_index(drop=True)

#change to floats
data["danceability"] = pd.to_numeric(data["danceability"], errors='coerce')
data["energy"] = pd.to_numeric(data["energy"], errors='coerce')
data["key"] = pd.to_numeric(data["key"], errors='coerce')
data["loudness"] = pd.to_numeric(data["loudness"], errors='coerce')
data["mode"] = pd.to_numeric(data["mode"], errors='coerce')
data["speechiness"] = pd.to_numeric(data["speechiness"], errors='coerce')
data["acousticness"] = pd.to_numeric(data["acousticness"], errors='coerce')
data["instrumentalness"] = pd.to_numeric(data["instrumentalness"], errors='coerce')
data["liveness"] = pd.to_numeric(data["liveness"], errors='coerce')
data["valence"] = pd.to_numeric(data["valence"], errors='coerce')
data["tempo"] = pd.to_numeric(data["tempo"], errors='coerce')


# Apply the mood classification function
data["mood"] = data.apply(classify_mood, axis=1)

# Remove tracks with "Unknown" mood
data = data[data["mood"] != "Unknown"]

# Select the required columns
X = data[["danceability", "energy", "key", "loudness", "mode", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo"]]

# Encode the mood labels to integers
mood_labels, uniques = pd.factorize(data["mood"])
data["mood_label"] = mood_labels

# Prepare features and target
y = data["mood_label"].values
X = X.values


# Correct train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

max_depths = range(1, 20) 
accuracies = []

for depth in max_depths:
    clf = decision_tree.DecisionTree(max_depth=depth)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    acc = np.sum(predictions == y_test) / len(y_test)
    accuracies.append(acc)



# # Initialize the DecisionTree
# clf = decision_tree.DecisionTree(max_depth=10)

# # Fit the model
# clf.fit(X_train, y_train)

# # Predictions
# predictions = clf.predict(X_test)

# # Define accuracy function
# def accuracy(y_true, y_pred):
#     accuracy = np.sum(y_true == y_pred) / len(y_true)
#     return accuracy

# # Calculate accuracy
# acc = accuracy(y_test, predictions)
# print(f"Model Accuracy: {acc}")

plt.figure(figsize=(10, 5))
plt.plot(max_depths, accuracies, marker='o')
plt.title('Model Accuracy vs. Tree Depth')
plt.xlabel('Max Depth of Decision Tree')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()
plt.savefig("dt_depth.png")
