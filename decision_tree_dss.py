# -*- coding: utf-8 -*-
"""
Created on Wed May  1 23:13:55 2019

@author: hp
"""
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

# Load libraries
#import mglearn 
import matplotlib.pyplot as plt
import numpy as np
#mglearn.plots.plot_tree_not_monotone()
from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics , tree
#import graphviz
from sklearn.tree import export_graphviz




digit = load_digits()

y = ['0', '1', '2' , '3' ,' 4' , '5' , '6' , '7' , '8'  , '9' ]
X_train, X_test, y_train, y_test = train_test_split(digit.data, digit.target, stratify=digit.target, random_state=42)

clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train, y_train)
print()

print('Accuracy on the training subset: {:.3f}'.format(clf.score(X_train, y_train)*100))
print('Accuracy on the test subset: {:.3f}'.format(clf.score(X_test, y_test)*100))
#to fixed the overfit 
#try to run in max depth =14 

clf = DecisionTreeClassifier( random_state=0,max_depth=13)
clf.fit(X_train, y_train)

print('\n\n\nAccuracy on the new  training subset: {:.3f}'.format(clf.score(X_train, y_train)*100))
print('Accuracy on the test new subset: {:.3f}'.format(clf.score(X_test, y_test)*100))

#import pandas as pd
#pixels = pd.DataFrame(digit.data)
#labels = pd.DataFrame(digit.target)
#pixels.describe
export_graphviz(clf, out_file='tree.dot', class_names=y, feature_names=range(0,64), impurity=False, filled=True)
#y_pred = clf.predict(X_test)
#clf.score(X_test, y_test)
#print(metrics.accuracy_score(y_pred, y_test))
#prdicte
y_predicted = clf.predict(X_test[30].reshape(1,-1))
label = y_predicted
pixel = X_test[30]
pixel = np.array(pixel, dtype='uint8')
pixel = pixel.reshape((8,8))
plt.title('Label is {label}'.format(label=label))
plt.imshow(pixel, cmap='gray')
plt.show()
print('i guess it is ',y_predicted)




#y_predicted = clf.predict(X_test[5].reshape(1,-1))
#label = y_predicted
#pixel = X_test[5]
#pixel = np.array(pixel, dtype='uint8')
#pixel = pixel.reshape((8,8))
#plt.title('Label is {label}'.format(label=label))
#plt.imshow(pixel, cmap='gray')
#plt.show()
#print('i guess it is ',y_predicted)




