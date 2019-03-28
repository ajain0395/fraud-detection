#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[1]:


from PIL import Image
import numpy as np
import copy
import cv2 as cv
import os
from scipy.sparse import csr_matrix, hstack
import matplotlib.pyplot as plt
import seaborn as seab
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn import metrics
np.set_printoptions(suppress=True) 


# In[149]:


def heatmap(confusionmat,title="Confusion Matrix",title2="",index=0):
    plt.figure()
    ax = plt.axes()
    seab.heatmap(confusionmat,linewidths=0.4,linecolor='white',annot=True,fmt='g') 
    ax.set_title(title + title2)
def getConfusionMat(predicted, actual,classcount):
    confusionmatrix = []
    for i in range (classcount):
        confusionmatrix.append([])
        for j in range (classcount):
            confusionmatrix[i].append(0)
    for i in range(0, len(predicted)):
        confusionmatrix[actual[i]][predicted[i]]+=1
    return confusionmatrix


# In[2]:


all_data = pd.read_csv("../Dataset/PS_20174392719_1491204439457_log.csv")


# In[3]:


print len(all_data)


# In[4]:


print all_data.keys()
all_data.shape


# In[99]:


print all_data.iloc[0,:]
all_data.head()


# # remove feature isFlagged fraud

# In[6]:


new_all_data = pd.DataFrame()
new_all_data = new_all_data.append(all_data.loc[:,['step','type','amount','nameOrig','oldbalanceOrg','newbalanceOrig','nameDest','oldbalanceDest','newbalanceDest','isFraud']])


# In[7]:


print len(new_all_data)


# In[ ]:





# In[8]:


set(new_all_data.type)


# In[9]:


new_filterdata = pd.DataFrame()


# In[10]:


new_filterdata = pd.DataFrame()
new_filterdata= new_filterdata.append(new_all_data.loc[new_all_data.loc[:,'type'] == 'TRANSFER',:],ignore_index=True)
len(new_filterdata)
new_filterdata= new_filterdata.append(new_all_data.loc[new_all_data.loc[:,'type'] == 'CASH_OUT',:],ignore_index=True)
len(new_filterdata)
print new_filterdata.keys()


# In[11]:


print set(new_filterdata.type)


# In[12]:


new_filterdata.head()


# In[13]:


copydata = copy.deepcopy(new_filterdata)


# In[14]:


copydata.tail()


# In[15]:


copydata.head()


# In[16]:


# copydata.reset_index()


# In[17]:


copydata.loc[copydata.type == 'TRANSFER', 'type'] = 1
copydata.loc[copydata.type == 'CASH_OUT', 'type'] = 2


# In[18]:


copydata.tail()


# In[19]:


print len(set(copydata.nameOrig))
print len(set(copydata.nameDest))


# In[20]:


copydata.head()


# In[ ]:





# In[21]:


# copy2 = copy.deepcopy(copydata)
# copy2.to_csv("../Dataset/filtereddata_withtypeconverted",index=False)


# In[22]:


# copydata = copy.deepcopy(copy2)
unique_nameOrig = set(copydata.nameOrig)
unique_nameDest = set(copydata.nameDest)


# In[ ]:





# In[23]:


print len(set(copydata.nameOrig))
print len(set(copydata.nameDest))


# In[24]:


print len(copydata.nameDest)


# In[ ]:





# In[25]:


listdata = np.array(copydata)


# In[26]:


print listdata[0]


# In[27]:


tmpunique = unique_nameDest.union(unique_nameOrig)
index = 1
tmpdict = {}
for i in range(len(listdata)):
    if(listdata[i][3] in tmpdict):
        listdata[i][3] = tmpdict[listdata[i][3]]
    else:
        tmpdict[listdata[i][3]] = index
        listdata[i][3] = index
        index += 1
    if(listdata[i][6] in tmpdict):
        listdata[i][6] = tmpdict[listdata[i][6]]
    else:
        tmpdict[listdata[i][6]] = index
        listdata[i][6] = index
        index += 1


# In[28]:


print listdata[0]


# In[ ]:





# In[29]:


newdataframe = pd.DataFrame(listdata,columns=copydata.keys())


# In[30]:


# newdataframe


# In[54]:


# newdataframe.to_csv("../Dataset/completepreprocesseddata.csv",index=False)


# In[141]:


train_data = newdataframe.loc[:,['step','type','amount','nameOrig','oldbalanceOrg','newbalanceOrig','nameDest','oldbalanceDest','newbalanceDest']]


# In[142]:


train_data.head()


# In[143]:


train_labels = newdataframe.loc[:,['isFraud']]


# In[144]:


train_labels.head()


# In[145]:


# train_data.to_csv("../Dataset/train_data.csv",index=False)
# train_labels.to_csv("../Dataset/train_labels.csv",index=False)


# In[ ]:





# In[146]:


from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


# In[147]:


# clf = SVC(gamma='auto')
clf = GaussianNB()
clf.fit(np.array(train_data),np.array(train_labels).astype(int))


# In[156]:


y = clf.predict(np.array(train_data))
print "Accuracy on Train Data: " +str( accuracy_score(y,np.array(train_labels).astype(int)))
heatmap(getConfusionMat(predicted=y,actual=np.array(train_labels.isFraud).astype(int),classcount=2),title2=" with GaussianNB")


# In[65]:


print "Count of Non Fraud Transactions: " +str(list(np.array(train_labels).astype(int)).count(0))


# In[61]:


print "Count of Fraud Transactions: " +str(list(np.array(train_labels).astype(int)).count(1))


# In[67]:


print "Percentage of Non Fraud Transactions: " +str(list(np.array(train_labels).astype(int)).count(0)/float(len(train_labels)) * 100)
print "Percentage of Fraud Transactions: " +str(list(np.array(train_labels).astype(int)).count(1)/float(len(train_labels)) * 100)


# In[58]:


#Transfer
dfFraudTransfer = newdataframe.loc[(newdataframe.isFraud == 1) & (newdataframe.type == 1)]
#Cashout
dfFraudCashout = newdataframe.loc[(newdataframe.isFraud == 1) & (newdataframe.type == 2)]

print ('\n Fraudulent TRANSFER count = '+str(len(dfFraudTransfer))) # 4097

print ('\n Fraudulent CASH_OUT count= '+str(len(dfFraudCashout))) 


# In[97]:


corr = train_data.astype(float).corr()
ax = plt.axes()
ax.set_title("Features Correlation")
seab.heatmap(corr,linewidths=0.4,linecolor='white',annot=False,cmap ="YlGnBu",)


# In[ ]:





# In[ ]:





# In[108]:


from sklearn import metrics
probabiloties = clf.predict_proba(np.array(train_labels).astype(int))
mett=metrics.classification_report(np.array(train_labels.isFraud).astype(int),y)
print mett
p,r,t = metrics.precision_recall_curve(probas_pred=probabiloties[:,0],y_true=np.array(train_labels.isFraud).astype(int))


# In[109]:


plt.plot(r, p, marker='.')


# In[122]:


train_data = np.array(train_data)
train_labels = np.array(train_labels).astype(int)


# In[123]:


from sklearn.model_selection import train_test_split, learning_curve
from sklearn.utils import shuffle
train_data,train_labels = shuffle(train_data,train_labels)
trainX, testX, trainY, testY = train_test_split(train_data, train_labels, test_size = 0.2)


# In[157]:


# clf = SVC(gamma='auto')
clf = GaussianNB()
clf.fit(trainX,trainY)


# In[160]:


y = clf.predict(testX)
print "Accuracy on Train Data: " +str( accuracy_score(y_true=testY,y_pred=y))
heatmap(getConfusionMat(predicted=y,actual=testY.flatten(),classcount=2),title2=" with GaussianNB")


# In[134]:


print testY.shape


# In[135]:


print y.shape


# In[ ]:





# In[ ]:




