import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
from gensim.models import Word2Vec
import chardet

# Load Dataset
dataset = pd.read_csv("data/text.csv")

# Drop disputed chapters

dataset = dataset[dataset['author'] != 'madison with hamilton']


# Extract labels and features

text_data = dataset['body'].tolist()
labels = dataset['author'].tolist()


# Split the text data into sentences
sentences = [text.split() for text in text_data]

# Train the Word2Vec model
model = Word2Vec(sentences, size=100, window=5, min_count=1, workers=4)

# Initialize an empty feature matrix
features = pd.DataFrame()

# Iterate through each sentence and compute the average word vector as features
for sentence in sentences:
    feature_vector = pd.Series([model.wv[word] for word in sentence]).mean()
    features = features.append(feature_vector, ignore_index=True)

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