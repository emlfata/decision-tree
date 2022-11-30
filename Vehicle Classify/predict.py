# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 20:44:56 2022

@author: muham
"""

#data processing
import pandas as pd
df1 = pd.read_csv("decisiontree1.csv")
encoding = {"mesin": {"bensin": 0, "diesel": 1},
            "penggerak": {"depan": 0, "belakang": 1}}
df1.replace(encoding, inplace=True)

X = df1.drop(['ID','label'], axis=1) #traning feature
y = df1['label'] #label

#select data
import sklearn.model_selection as ms
X_train, X_test, y_train, y_test = ms.train_test_split(X,y, test_size = 0.2)

#MODEL 1 (splitting entropy)
import sklearn.tree as tree
model1 = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5) #entropy
model1.fit(X_train, y_train)

y_prediksi = model1.predict(X_test)
print("Splitting Entropy")
print(y_prediksi)
#model accuracy
import sklearn.metrics as met
print(met.accuracy_score(y_test, y_prediksi))

#visualize
import pydotplus as pp
labels = ['mesin','bangku','penggerak']
dot_data = tree.export_graphviz(model1, out_file=None, feature_names=labels, 
                                filled=True, rounded=True)
graph = pp.graph_from_dot_data(dot_data)
graph.write_png('decisionentropy.png')


#MODEL 2 (splitting gini index)
import sklearn.tree as tree
model2 = tree.DecisionTreeClassifier(criterion='gini', max_depth=5) #gini index
model2.fit(X_train, y_train)

y_prediksi = model2.predict(X_test)
print("\nSplitting Gini Index")
print(y_prediksi)
#model accuracy
import sklearn.metrics as met
print(met.accuracy_score(y_test, y_prediksi))

#visualize
import pydotplus as pp
labels = ['mesin','bangku','penggerak']
dot_data = tree.export_graphviz(model2, out_file=None, feature_names=labels, 
                                filled=True, rounded=True)
graph = pp.graph_from_dot_data(dot_data)
graph.write_png('decisiongini.png')