# imports
import pandas as pd
import numpy as np

# load in file into a pandas dataframe
df = pd.read_csv("Titanic.csv")

# first, let's clean up our data set
df = df.dropna()

# drop the labeled 'male/female' column - sex code(female = 1, male = 0)
SolutionCol = df["Survived"]
df = df.drop(["Survived", "Unnamed: 0"], axis=1)

# fix Class from 1st, 2nd, etc to just 1, 2, 3 - numerical values
df["Class"] = df["Class"].map(lambda x: str(x)[0])

# split train-test data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    df, SolutionCol, test_size=0.25, random_state=1
)

# setup the model
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

# score the model - accuracy
from sklearn.metrics import accuracy_score

acc = accuracy_score(y_test, predictions)

# precision
from sklearn.metrics import precision_score

precision = precision_score(y_test, predictions)

# true positive, false positive, false negative, true negative rate
from sklearn.metrics import confusion_matrix

confusionMatrix = confusion_matrix(y_test, predictions)

# print the values
print("Accuracy: " + str(int(acc * 100)) + "%")
print("Precision: " + str(precision))
print("Confusion Matrix: " + str(confusion_matrix(y_test, predictions)))

# important features
list(zip(df.columns.values, clf.feature_importances_))
