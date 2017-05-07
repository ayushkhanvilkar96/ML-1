
# coding: utf-8

# In[10]:

from sklearn import tree


# In[11]:

features=[[140,1],[130,1],[150,0],[170,0]] #first value is the weight and second value is the texture 1-Smooth 0-Bumpy


# In[14]:

labels=[0,0,1,1] #0 for apples and 1 for oranges
clf=tree.DecisionTreeClassifier()


# In[15]:

clf=clf.fit(features,labels)


# In[24]:

print(clf.predict([[160,0]]))


# In[ ]:



