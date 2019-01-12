#!/usr/bin/python3
import pandas as pd 
import seaborn as sb
# loading  tree factor for decisiontree Classifier 
from sklearn import tree
from sklearn.model_selection import  train_test_split

# loading  dataset 
df=pd.read_csv('diabetes.csv')

#  creating  training and testing datasets 
labels=df['Outcome']

features=df[df.columns[:-1]]

# split datasets into training and testing 
#10%
train_data,test_data,train_target,test_target=(train_test_split(features,labels,test_size=0.1))
#20%
train_data1,test_data1,train_target1,test_target1=(train_test_split(features,labels,test_size=0.2))

#  calling  decision tree
clf=tree.DecisionTreeClassifier()
#  training  data
trained_algo10=clf.fit(train_data,train_target)
trained_algo20=clf.fit(train_data1,train_target1)
# time for prediction 
output10=trained_algo10.predict(test_data)
output20=trained_algo20.predict(test_data1)
print("predicted output",output10)
print("predicted output",output20)

# loading KNN 
from  sklearn.neighbors  import  KNeighborsClassifier
clf=KNeighborsClassifier(n_neighbors=5)
trained10=clf.fit(train_data,train_target)
trained20=clf.fit(train_data1,train_target1)
outputk10=trained10.predict(test_data)
outputk20=trained20.predict(test_data1)



# calculating  accuracy for  decisiontree 
from  sklearn.metrics  import  accuracy_score
acc10=accuracy_score(test_target,output10)
acc20=accuracy_score(test_target1,output20)
print("accuracy of DSC Tree with 10 % ",acc10)
print("accuracy of DSC Tree with 20%  ",acc20)

# calculating  accuracy for  KNN
from  sklearn.metrics  import  accuracy_score
acck10=accuracy_score(test_target,outputk10)
acck20=accuracy_score(test_target1,outputk20)
print("accuracy of  KNN with 10 % ",acck10)
print("accuracy of KNN Tree with 20%  ",acck20)

# plotting graph
import matplotlib.pyplot  as plt
plt.plot(acc10,acck10)
plt.plot(acc20,acck20)
plt.bar(acc10,acck10)
plt.bar(acc20,acck20)
plt.show()
