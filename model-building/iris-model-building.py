import pandas as pd
Iris = pd.read_csv('Iris.csv')

# Ordinal feature encoding
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
df = Iris.copy()
target = 'Species'
# encode = ['sex','island']

# for col in encode:
#     dummy = pd.get_dummies(df[col], prefix=col)
#     df = pd.concat([df,dummy], axis=1)
#     del df[col]

df = df.drop(columns=['Id'])

target_mapper = {'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2}
def target_encode(val):
    return target_mapper[val]

df['Species'] = df['Species'].apply(target_encode)

# Separating X and y
X = df.drop('Species', axis=1)
Y = df['Species']

# Build random forest model
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X, Y)

# Saving the model
import pickle
pickle.dump(clf, open('Iris_clf.pkl', 'wb'))
