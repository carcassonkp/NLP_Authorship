import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics


# Load Dataset
dataset = pd.read_csv("data/text_stop_stem.csv")

# Drop disputed chapters

dataset = dataset[dataset['author'] != 'madison with hamilton']


# Extract labels and features

text_data = dataset['body'].tolist()
labels = dataset['author'].tolist()

vectorizer = CountVectorizer()
bow_matrix = vectorizer.fit_transform(text_data)

features = pd.DataFrame(bow_matrix.toarray(), columns=vectorizer.get_feature_names_out())
features_with_labels = pd.concat([features, pd.Series(labels, name='label')], axis=1)

# Split the features and labels
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=0)

classifier = SVC()
classifier.fit(X_train, y_train)

# Predict the labels for the test data
y_pred = classifier.predict(X_test)

# Calculate accuracy
accuracy = metrics.accuracy_score(y_test, y_pred)

# Calculate precision, recall, and F1 score
precision = metrics.precision_score(y_test, y_pred, average='weighted')
recall = metrics.recall_score(y_test, y_pred, average='weighted')
f1_score = metrics.f1_score(y_test, y_pred, average='weighted')

# Print the metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)