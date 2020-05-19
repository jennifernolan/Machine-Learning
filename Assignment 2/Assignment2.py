#Reading file from pandas
#libraries to import
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import tree
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

#column headings of the dataset
columnHeadings = ['id', 'age', 'job', 'martial', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'campaign', 'pdays', 'previous', 'poutcome', 'deposit']

#read the dataset from the specified file
phoneData = pd.read_csv('trainingset.csv', header = None, names = columnHeadings)

print(phoneData.head())

#extract the target feature (furthest right column)
target = phoneData['deposit']

#Extract continuous/numeric features
numeric_features = ['age', 'balance', 'day', 'campaign', 'pdays', 'previous']
numeric_dfs = phoneData[numeric_features]

#Extract categorical features
cat_dfs = phoneData.drop(numeric_features + ['deposit'], axis=1)
#transform to array of dictionaries of feature level pairs
cat_dfs = cat_dfs.T.to_dict().values()
#convert to numeric encoding
vectorizer = DictVectorizer(sparse = False)
vec_cat_dfs = vectorizer.fit_transform(cat_dfs)
#merge categorical and numeric features into training dataframe
train_dfs = np.hstack((numeric_dfs.as_matrix(), vec_cat_dfs))

#--------------------------------------------
# Compare accuracy of entropy and Gini criterion
#--------------------------------------------
decTreeModel2 = tree.DecisionTreeClassifier(criterion='entropy')
instances_train, instances_test, target_train, target_test = train_test_split(train_dfs, target, test_size=0.4, random_state=0)
decTreeModel2.fit(instances_train, target_train)
predictions = decTreeModel2.predict(instances_test)
print("Entropy Accuracy= " + str(accuracy_score(target_test, predictions, normalize=True)))

decTreeModel3 = tree.DecisionTreeClassifier(criterion='gini')
decTreeModel3.fit(instances_train, target_train)
predictions = decTreeModel3.predict(instances_test)
print("Gini Accuracy= " + str(accuracy_score(target_test, predictions, normalize=True)))

#read the queries files
phoneQueries = pd.read_csv('queries.csv', header = None, names = columnHeadings)

print(phoneQueries.head())

#prepare the queries set to make predictions
query_num = phoneQueries[numeric_features].as_matrix()
query_cat = phoneQueries.drop(numeric_features,axis=1)
query_cat_dfs = query_cat.T.to_dict().values()
query_vect_dfs = vectorizer.transform(query_cat_dfs)
query = np.hstack((query_num, query_vect_dfs))

prediction = []
ids = []

#for loop to go through each query row and make a prediction using Gini criterion as it is more accurate
for i in range(0, len(query)):
    ids.append(phoneQueries['id'][i])
    predictions = decTreeModel3.predict([query[i]])
    prediction.append(predictions[0])
    
#convert the predictions to the correct specified format
predict = pd.DataFrame(prediction, index = ids)

#place the predictions in a .csv file
predict.to_csv("C16517636Predict.csv", header=None)





