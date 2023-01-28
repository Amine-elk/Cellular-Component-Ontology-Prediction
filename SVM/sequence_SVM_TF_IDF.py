import csv
import numpy as np
from sklearn.metrics import accuracy_score, log_loss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import os 
os.chdir('C:/Users/elfak/OneDrive/Bureau/altegrad_challenge_2022')
# Read sequences
sequences = list()
with open('data/sequences.txt', 'r') as f:
    for line in f:
        sequences.append(line[:-1])

# Split data into training and test sets
sequences_train = list()
sequences_test = list()
proteins_test = list()
y_train = list()
with open('data/graph_labels.txt', 'r') as f:
    for i,line in enumerate(f):
        t = line.split(',')
        if len(t[1][:-1]) == 0:
            proteins_test.append(t[0])
            sequences_test.append(sequences[i])
        else:
            sequences_train.append(sequences[i])
            y_train.append(int(t[1][:-1]))

# Map sequences to 
vec = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))
X_train = vec.fit_transform(sequences_train)
X_test = vec.transform(sequences_test)

df = X_train
y_ = y_train

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    df, y_train, test_size=0.33, random_state=42)
# Train a logistic regression classifier and use the classifier to
# make predictions
clf = SVC(kernel='rbf', probability= True)
clf.fit(X_train, y_train) 
y_pred_proba = clf.predict_proba(X_test)

from sklearn.metrics import classification_report

y_pred = clf.predict(X_train)
accuracy_score(y_train, y_pred)
print(classification_report(y_train, y_pred))


y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)
print(classification_report(y_train, y_pred))
# Write predictions to a file
# with open('sample_submission.csv', 'w') as csvfile:
#     writer = csv.writer(csvfile, delimiter=',')
#     lst = list()
#     for i in range(18):
#         lst.append('class'+str(i))
#     lst.insert(0, "name")
#     writer.writerow(lst)
#     for i, protein in enumerate(proteins_test):
#         lst = y_pred_proba[i,:].tolist()
#         lst.insert(0, protein)
#         writer.writerow(lst)
