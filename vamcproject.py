#!/usr/bin/env python
# coding: utf-8

# In[198]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
dt=pd.read_csv("student_data.csv")
dt.head()


# In[199]:


#dt=dt
#plt.scatter(dt['sex'],dt['age'])
#plt.xlabel("sex")
#plt.ylabel("age")
#plt.show()


# In[200]:


dt['sex'][dt['sex'] == 0] = 'F'
dt['sex'][dt['sex'] == 1] = 'M'
dt['school'][dt['school'] == 0] = 'GP'
dt['school'][dt['school'] == 1] = 'MS'

dt['famsize'][dt['famsize'] == 0] = 'GT3'
dt['famsize'][dt['famsize'] == 1] = 'LE3'

dt['address'][dt['address'] == 0] = 'U'
dt['address'][dt['address'] == 1] = 'R'

dt['Pstatus'][dt['Pstatus'] == 0] = 'A'
dt['Pstatus'][dt['Pstatus'] == 1] = 'T'


dt['Mjob'][dt['Mjob'] == 0] = 'at_home'
dt['Mjob'][dt['Mjob'] == 1] = 'health'
dt['Mjob'][dt['Mjob'] == 2] = 'other'
dt['Mjob'][dt['Mjob'] == 3] = 'teacher'
dt['Mjob'][dt['Mjob'] == 4] = 'services'

dt['Fjob'][dt['Fjob'] == 0] = 'at_home'
dt['Fjob'][dt['Fjob'] == 1] = 'health'
dt['Fjob'][dt['Fjob'] == 2] = 'other'
dt['Fjob'][dt['Fjob'] == 3] = 'teacher'
dt['Fjob'][dt['Fjob'] == 4] = 'services'
dt['reason'][dt['reason'] == 0] = 'home'
dt['reason'][dt['reason'] == 1] = 'course'
dt['reason'][dt['reason'] == 2] = 'other'
dt['reason'][dt['reason'] == 3] = 'reputation'
dt['guardian'][dt['guardian'] == 0] = 'mother'
dt['guardian'][dt['guardian'] == 1] = 'father'
dt['guardian'][dt['guardian'] == 2] = 'other'
dt['schoolsup'][dt['schoolsup'] == 0] = 'yes'
dt['schoolsup'][dt['schoolsup'] == 1] = 'no'
dt['famsup'][dt['famsup'] == 0] = 'yes'
dt['famsup'][dt['famsup'] == 1] = 'no'
dt['paid'][dt['paid'] == 0] = 'yes'
dt['paid'][dt['paid'] == 1] = 'no'
dt['activities'][dt['activities'] == 0] = 'yes'
dt['activities'][dt['activities'] == 1] = 'no'
dt['nursery'][dt['nursery'] == 0] = 'yes'
dt['nursery'][dt['nursery'] == 1] = 'no'
dt['internet'][dt['internet'] == 0] = 'yes'
dt['internet'][dt['internet'] == 1] = 'no'
dt['higher'][dt['higher'] == 0] = 'yes'
dt['higher'][dt['higher'] == 1] = 'no'
dt['romantic'][dt['romantic'] == 0] = 'yes'
dt['romantic'][dt['romantic'] == 1] = 'no'


# In[201]:


dt.head()
d1=pd.get_dummies(dt.sex)
d2=pd.get_dummies(dt.school)
d3=pd.get_dummies(dt.famsize)
d4=pd.get_dummies(dt.Pstatus)
d5=pd.get_dummies(dt.Mjob)
d6=pd.get_dummies(dt.Fjob)
d7=pd.get_dummies(dt.reason)
d8=pd.get_dummies(dt.guardian)
d9=pd.get_dummies(dt.schoolsup)
d10=pd.get_dummies(dt.famsup)
d11=pd.get_dummies(dt.paid)
d12=pd.get_dummies(dt.activities)
d13=pd.get_dummies(dt.nursery)
d14=pd.get_dummies(dt.internet)
d15=pd.get_dummies(dt.higher)
d16=pd.get_dummies(dt.romantic)
d17=pd.get_dummies(dt.address)
dt.drop(['sex','school','famsize', 'Pstatus','Mjob','Fjob','reason','guardian','address','schoolsup','famsup', 'paid','activities','nursery','internet','higher','romantic'],axis=1,inplace=True )
dt= pd.concat([dt,d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15,d16,d17],axis=1)



# In[ ]:





# In[239]:


dt.head()
dt.columns


# In[240]:


y=dt.passed
x=dt.drop(["passed",],axis=1)


# In[241]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)


# In[242]:


from sklearn.neighbors import KNeighborsClassifier


# In[243]:


knn = KNeighborsClassifier( n_neighbors=10)
knn


# In[244]:


#knn.fit(x_train,y_train)


# # Logistic Regression

# In[245]:


y=dt.passed
x=dt.drop(["passed",],axis=1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=5)


# In[246]:


from sklearn.linear_model import LogisticRegression


# In[247]:


model=LogisticRegression()
model.fit(X_train,y_train)


# In[248]:


predictions=model.predict(X_test)


# In[249]:


from sklearn.metrics import classification_report
classification_report(y_test,predictions)


# In[214]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)


# In[215]:


from sklearn.metrics import accuracy_score


# In[217]:


accuracy_score(y_test,predictions)


# # Applying Decision Tree

# In[218]:


X=dt.drop(["passed"],axis=1)
y=dt.passed


# In[219]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)


# In[220]:


from sklearn import tree


# In[221]:


md=tree.DecisionTreeClassifier()


# In[222]:


md.fit(X_train,y_train)


# In[223]:


md.score(X_test,y_test)


# # Applying Random Forest

# In[224]:


X=dt.drop(["passed"],axis=1)
y=dt.passed


# In[225]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# In[226]:


from sklearn.ensemble import RandomForestClassifier


# In[227]:


rf = RandomForestClassifier (n_estimators=100)


# In[228]:


rf.fit(X_train, y_train)


# In[229]:


accuracy = rf.score(X_test, y_test)
print("Accuracy = {}%".format(accuracy * 100))
 


# # Classification Report

# In[233]:


predictions=rf.predict(X_test)


# In[234]:


from sklearn.metrics import classification_report


# In[235]:


classification_report(y_test,predictions)


# In[236]:


from sklearn.metrics import accuracy_score


# In[237]:


accuracy_score(y_test,predictions)


# In[ ]:




