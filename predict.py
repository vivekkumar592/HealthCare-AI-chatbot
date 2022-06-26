# importing necessary libraries 
from sklearn import datasets 
import pandas as pd
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
#from sklearn.cross_validation import tain_test_split
# loading the iris dataset 
training = pd.read_csv('Training.csv')
#testing  = pd.read_csv('Testing.csv')
cols     = training.columns
cols     = cols[:-1]
X        = training[cols]
y     = training['prognosis']






X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print(X_train.shape)
# training a KNN classifier 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn import metrics
k_range=range(1,26)
scores={}
scores_list=[]
for k in k_range:
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    y_pred=knn.predict(X_test)
    scores[k]=metrics.accuracy_score(y_test,y_pred)
    scores_list.append(metrics.accuracy_score(y_test,y_pred))

import matplotlib.pyplot as plt
plt.plot(k_range,scores_list)
plt.xlabel('Value of K for knn')
plt.ylabel('Testing Accuracy')
knn=KNeighborsClassifier(n_neighbors=7)
knn.fit(X,y)
import pickle
predicting=open('predicting_model','wb')
pickle.dump(knn,predicting)

