#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[23]:


from PIL import Image
import numpy as np
import copy
import cv2 as cv
import os
from scipy.sparse import csr_matrix, hstack
import matplotlib.pyplot as plt
import seaborn as seab
import warnings

# import tensorflow as tf
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.patches as mpatches

# from imblearn.datasets import fetch_datasets
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
from collections import Counter

warnings.filterwarnings("ignore")
import pandas as pd
from sklearn import metrics
np.set_printoptions(suppress=True) 


# In[2]:


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


# In[3]:


all_data = pd.read_csv("../Dataset/PS_20174392719_1491204439457_log.csv")


# In[4]:


print len(all_data)


# In[5]:


print all_data.keys()
all_data.shape


# In[6]:


print all_data.iloc[0,:]
all_data.head()


# # remove feature isFlagged fraud

# In[7]:


new_all_data = pd.DataFrame()
new_all_data = new_all_data.append(all_data.loc[:,['step','type','amount','nameOrig','oldbalanceOrg','newbalanceOrig','nameDest','oldbalanceDest','newbalanceDest','isFraud']])


# In[8]:


print len(new_all_data)


# In[ ]:





# # Data Visualisations

# In[9]:


print(new_all_data.type.value_counts())

f, ax = plt.subplots(1, 1, figsize=(8, 8))
new_all_data.type.value_counts().plot(kind='bar', title="Transaction type", ax=ax, figsize=(8,8))
plt.show()


# In[10]:


ax = new_all_data.groupby(['type', 'isFraud']).size().plot(kind='bar')
ax.set_title("# of transaction which are the actual fraud per transaction type")
ax.set_xlabel("(Type, isFraud)")
ax.set_ylabel("Count of transaction")
for p in ax.patches:
    ax.annotate(str(format(int(p.get_height()), ',d')), (p.get_x(), p.get_height()*1.01))


# In[11]:


ax = all_data.groupby(['type', 'isFlaggedFraud']).size().plot(kind='bar')
ax.set_title("# of transaction which is flagged as fraud per transaction type")
ax.set_xlabel("(Type, isFlaggedFraud)")
ax.set_ylabel("Count of transaction")
for p in ax.patches:
    ax.annotate(str(format(int(p.get_height()), ',d')), (p.get_x(), p.get_height()*1.01))


# In[17]:


tmp = all_data.loc[(all_data['type'].isin(['TRANSFER', 'CASH_OUT'])),:]
ax = pd.value_counts(tmp['isFraud'], sort = True).sort_index().plot(kind='bar',color = "green", title="Fraud transaction count")
for p in ax.patches:
    ax.annotate(str(format(int(p.get_height()), ',d')), (p.get_x(), p.get_height()))
    
plt.show()


# # Data Visualisation End

# In[24]:





# In[9]:


set(new_all_data.type)


# In[27]:


# pop = fsga.generate(100)
new_filterdata = pd.DataFrame()


# In[28]:


new_filterdata = pd.DataFrame()
new_filterdata= new_filterdata.append(new_all_data.loc[new_all_data.loc[:,'type'] == 'TRANSFER',:],ignore_index=True)
len(new_filterdata)
new_filterdata= new_filterdata.append(new_all_data.loc[new_all_data.loc[:,'type'] == 'CASH_OUT',:],ignore_index=True)
len(new_filterdata)
print new_filterdata.keys()


# In[29]:


print set(new_filterdata.type)


# In[30]:


new_filterdata.head()


# In[37]:


copydata = copy.deepcopy(new_filterdata)


# In[38]:


copydata.tail()


# In[39]:


copydata.head()


# In[40]:


# copydata.reset_index()


# In[41]:


copydata.loc[copydata.type == 'TRANSFER', 'type'] = 0
copydata.loc[copydata.type == 'CASH_OUT', 'type'] = 1


# In[42]:


copydata.tail()


# In[43]:


print len(set(copydata.nameOrig))
print len(set(copydata.nameDest))


# In[47]:


from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')
X = [['Male', 1], ['Female', 3], ['Female', 2]]
enc.fit(X)
print  enc.categories_
print X
enc.transform(X).toarray()


# In[22]:


copydata.head()


# In[ ]:





# In[23]:


# copy2 = copy.deepcopy(copydata)
# copy2.to_csv("../Dataset/filtereddata_withtypeconverted",index=False)


# In[24]:


# copydata = copy.deepcopy(copy2)
unique_nameOrig = set(copydata.nameOrig)
unique_nameDest = set(copydata.nameDest)


# In[ ]:





# In[25]:


print len(set(copydata.nameOrig))
print len(set(copydata.nameDest))


# In[29]:


print len(copydata.nameDest)


# In[ ]:





# In[30]:


listdata = np.array(copydata)


# In[31]:


print listdata[0]


# In[32]:


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


# In[33]:


print listdata[0]


# In[ ]:





# In[26]:


newdataframe = pd.DataFrame(listdata,columns=copydata.keys())


# In[35]:


# newdataframe


# In[36]:


# newdataframe.to_csv("../Dataset/completepreprocesseddata.csv",index=False)


# In[37]:


train_data = newdataframe.loc[:,['step','type','amount','nameOrig','oldbalanceOrg','newbalanceOrig','nameDest','oldbalanceDest','newbalanceDest']]


# In[38]:


train_data.head()


# In[40]:


train_labels = newdataframe.loc[:,['isFraud']]


# In[41]:


train_labels.head()


# In[42]:


# train_data.to_csv("../Dataset/train_data.csv",index=False)
# train_labels.to_csv("../Dataset/train_labels.csv",index=False)


# In[ ]:





# In[43]:


from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


# In[44]:


# clf = SVC(gamma='auto')
clf = GaussianNB()
clf.fit(np.array(train_data),np.array(train_labels).astype(int))


# In[48]:


# y = clf.predict(np.array(train_data))
# print "Accuracy on Train Data: " +str( accuracy_score(y,np.array(train_labels).astype(int)))
# heatmap(getConfusionMat(predicted=y,actual=np.array(train_labels.isFraud).astype(int),classcount=2),title2=" with GaussianNB")


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


# In[48]:


corr = train_data.astype(float).corr()
ax = plt.axes()
ax.set_title("Features Correlation")
seab.heatmap(corr,linewidths=0.4,linecolor='white',annot=True,cmap ="YlGnBu")


# In[25]:


train_data_correlation = newdataframe.loc[:,['step','type','amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest']]


# In[57]:


train_data_correlation.tail()


# In[58]:


train_labels.tail()


# In[69]:


# from sklearn.manifold import TSNE
# X_embedded = TSNE(n_components=2).fit_transform(np.array(train_data_correlation))
print int(train_labels.iloc[0] == 1)


# In[61]:


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
scatter_X = pca.fit_transform(np.array(train_data_correlation))
for i in range(0,len(scatter_X)):
    if(int(train_labels.iloc[i] == 1) == 1):
        color = 'red'
        else:
            color = 'blue'
        plt.scatter(scatter_X[i][0],scatter_X[i][1],c=color)


# In[ ]:





# In[51]:


from sklearn import metrics
probabilities = clf.predict_proba(np.array(train_labels).astype(int))
mett=metrics.classification_report(np.array(train_labels.isFraud).astype(int),y)
print mett
p,r,t = metrics.precision_recall_curve(probas_pred=probabilities[:,0],y_true=np.array(train_labels.isFraud).astype(int))


# In[18]:


# plt.plot(r, p, marker='^')


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




