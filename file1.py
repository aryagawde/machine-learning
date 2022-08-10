import pandas as pd
import numpy as np
import matplotlib.pyplot as mlt
from sklearn.impute import SimpleImputer
import seaborn as sns

testDataset = pd.read_csv('train.csv')

# print(testDataset.head())

# print(testDataset.isnull())

# sns.heatmap(testDataset.isnull(), yticklabels=False, cbar=False, cmap='viridis')

# sns.set_style('whitegrid')
# sns.countplot(x = 'Survived', hue='Pclass', data=testDataset, palette='rainbow')

# sns.distplot(testDataset['Age'].dropna(), kde=False, color='darkred', bins=40)

# sns.countplot(x = 'SibSp', data=testDataset)

# testDataset['Fare'].hist(color='green', bins=40, figsize=(8, 4))

# mlt.figure(figsize=(12, 7))
# sns.boxplot(x='Pclass', y='Age', data=testDataset, palette='winter')
# mlt.show()

def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]

    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age

testDataset['Age'] = testDataset[['Age', 'Pclass']].apply(impute_age, axis=1)
# sns.heatmap(testDataset.isnull(), yticklabels=False, cbar=False, cmap='viridis')
# mlt.show()

testDataset.drop('Cabin', axis=1, inplace=True)

# print(testDataset.head())

# sns.heatmap(testDataset.isnull(), yticklabels=False, cbar=False, cmap='viridis')
# mlt.show()

testDataset.dropna(inplace=True)

# testDataset.info()
# pd.get_dummies(testDataset['Embarked'], drop_first=True).head()

sex = pd.get_dummies(testDataset['Sex'], drop_first=True)
embark = pd.get_dummies(testDataset['Embarked'], drop_first=True)

testDataset.drop(['Sex', 'Embarked', 'Name', 'Ticket'], axis=1, inplace=True)

testDataset = pd.concat([testDataset,sex,embark], axis=1)
# print(testDataset.head())

# testDataset.drop('Cabin', axis=1, inplace=True)

testDataset.drop('Survived', axis=1).head()

print(testDataset['Survived'].head())

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(testDataset.drop('Survived', axis=1), testDataset['Survived'], test_size=0.30, random_state=101)
from sklearn.linear_model import LogisticRegression


logmodel = LogisticRegression(max_iter=889)
logmodel.fit(X_train, y_train)

predictions = logmodel.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix

accuracy = confusion_matrix(y_test, predictions)

accuracy = accuracy_score(y_test, predictions)
print(accuracy)

print(predictions)