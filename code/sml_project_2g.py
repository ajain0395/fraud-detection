#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[101]:


import numpy as np
import copy
from sklearn.linear_model import LogisticRegression
import os
from scipy.sparse import csr_matrix, hstack
import matplotlib.pyplot as plt

import warnings
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve,precision_recall_curve,accuracy_score,roc_auc_score
import seaborn as seab
import matplotlib.pyplot as plt


import numpy as np
import copy

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
from imblearn.over_sampling import SMOTE


# from imblearn.datasets import fetch_datasets
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import precision_recall_curve,precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
from collections import Counter
from PIL import Image
import numpy as np
import copy
# import cv2
import os
from scipy.cluster.vq import whiten
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import warnings
from sklearn.decomposition import PCA
warnings.filterwarnings("ignore")
import pandas as pd
from scipy.sparse import csr_matrix, hstack
import matplotlib.pyplot as plt
import seaborn as seab
from numpy import histogram
import warnings
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC,LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import matthews_corrcoef as mcc
from sklearn.metrics import classification_report,confusion_matrix
# from xgboost import XGBClassifier
import scipy
from scipy.misc import imread
# from skimage.color import rgb2lab
# from skimage.color import rgb2gray
# from skimage.measure import regionprops
import pickle
import random
import seaborn as sb
# from skimage.feature import hog,local_binary_pattern
from sklearn.model_selection import train_test_split
# from skimage import data, exposure
# train_test_split_ratio = 0.7

warnings.filterwarnings("ignore")
import pandas as pd
from sklearn import metrics
np.set_printoptions(suppress=True) 


# In[328]:


knn = KNeighborsClassifier(weights='uniform',n_neighbors=48,p=2,leaf_size=30,metric='minkowski',n_jobs=1,algorithm='auto',metric_params=None)
randomforest =RandomForestClassifier(n_estimators=150,random_state=42)
logistic = LogisticRegression()
gnb = GaussianNB()


# In[378]:


def getvariance(nonedata):
    pca=PCA(n_components=len(nonedata[0]))
    pca.fit(nonedata)
    return pca.explained_variance_ratio_.sum()
import matplotlib.patches as pat
def scatterplot(data,labels):
    plt.figure()
    pca = PCA(n_components=2)
    scatter_X = pca.fit_transform(data)
    classes = ['Fraud','Not-Fraud']
    colours = ['orange','grey']
    recs = []
    plt.xlim([-5, 30])
    plt.ylim([-5, 30])
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    #plt.title("Scatter plot: Undersampling with PCA")
    for i in range(0,len(colours)):
        recs.append(pat.Rectangle((0,0),1,1,fc=colours[i]))

    for i in range(0,len(scatter_X)):
        if((labels[i] == 1)):
            plt.scatter(scatter_X[i][0],scatter_X[i][1],c=colours[0])
        else:
            plt.scatter(scatter_X[i][0],scatter_X[i][1],c=colours[1])
    plt.legend(recs,classes,loc='best')
    plt.show()


# In[322]:


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
def p_r_c(true_labels,scores):
    plt.figure()
    precision, recall, _ = precision_recall_curve(true_labels, scores)
    plt.plot(recall, precision)
    #plt.fill_between(recall, precision, step='post', alpha=0.2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
#     plt.ylim([0.0, 1.05])
#     plt.xlim([0.0, 1.0])
    plt.title('Precision Recall Curve')
    plt.show()
    
def r_o_c(true_labels,scores):
    plt.figure()
    fpr, tpr, thresholds = roc_curve(true_labels, scores)
    plt.plot(fpr, tpr)
    #plt.fill_between(recall, precision, step='post', alpha=0.2,     color='#F59B00')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
#     plt.ylim([0.0, 1.05])
#     plt.xlim([0.0, 1.0])
    plt.title('ROC Curve')
    plt.show()
def reports(classifier,train_data,train_labels,train_test_split_ratio=.2,folding=False):
    scores = []
    print classifier
    X_train, X_test, y_train, y_test = train_test_split(train_data, train_labels,shuffle=False, test_size=train_test_split_ratio)
    if(folding == True):
        kf = KFold(n_splits=5,shuffle=False)
        kf.get_n_splits(X_train)
        print(kf)
        scores = []
        for train_index, test_index in kf.split(X_train):
            #print("TRAIN:", len(train_index), "TEST:", len(test_index))
            X_traini, X_testi = X_train[train_index], X_train[test_index]
            y_traini, y_testi = y_train[train_index], y_train[test_index]
            classifier.fit(X_traini,y_traini)
            predicted = classifier.predict(X_testi)
            scores.append(accuracy_score(predicted,y_testi))
        scores = np.array(scores)
        print ("Per fold Score 5 fold",scores)
        print ("Average Accuracy K Fold: ",scores.mean())
    
    classifier.fit(X_train,y_train)
    predicted = classifier.predict(X_test)
    prob_scores = classifier.predict_proba(X_test)
    r_o_c(y_test,prob_scores[:,1])
    p_r_c(y_test,prob_scores[:,1])
    print ("Test Data Results:")
    print ("Test Accuracy: ",accuracy_score(predicted,y_test))
    print "MCC: ",mcc(y_test, predicted)
    X = classification_report(y_test,predicted)
    print (X)
    print "ROC AUC",roc_auc_score(y_true=y_test,y_score=prob_scores[:,1])
    heatmap(confusionmat=getConfusionMat(actual=y_test,predicted=predicted,classcount=2))
    
    
def reports_test(classifier,X_train,y_train,X_test,y_test):
    scores = []
    print classifier
#     X_train, X_test, y_train, y_test = train_test_split(train_data, train_labels,shuffle=False, test_size=train_test_split_ratio)
    classifier.fit(X_train,y_train)
    predicted = classifier.predict(X_test)
    prob_scores = classifier.predict_proba(X_test)
    r_o_c(y_test,prob_scores[:,1])
    p_r_c(y_test,prob_scores[:,1])
    print ("Test Data Results:")
    print ("Test Accuracy: ",accuracy_score(predicted,y_test))
    print "MCC: ",mcc(y_test, predicted)
    X = classification_report(y_test,predicted)
    print (X)
    print "ROC AUC",roc_auc_score(y_true=y_test,y_score=prob_scores[:,1])
    heatmap(confusionmat=getConfusionMat(actual=y_test,predicted=predicted,classcount=2))


# In[4]:


all_data = pd.read_csv("../Dataset/PS_20174392719_1491204439457_log.csv")


# In[5]:


print len(all_data)


# In[6]:


print all_data.keys()
all_data.shape


# In[7]:


print all_data.iloc[0,:]
all_data.head()


# # remove feature isFlagged fraud

# In[8]:


new_all_data = pd.DataFrame()
new_all_data = new_all_data.append(all_data.loc[:,['step','type','amount','nameOrig','oldbalanceOrg','newbalanceOrig','nameDest','oldbalanceDest','newbalanceDest','isFraud']])


# In[9]:


print " Total samples in original data: ",len(new_all_data)


# In[10]:


set(new_all_data.type)


# # Working on only two type- Cashout and Transfer

# In[11]:


new_filterdata = pd.DataFrame()


# In[12]:


new_filterdata = pd.DataFrame()
new_filterdata= new_filterdata.append(new_all_data.loc[new_all_data.loc[:,'type'] == 'TRANSFER',:],ignore_index=True)
len(new_filterdata)
new_filterdata= new_filterdata.append(new_all_data.loc[new_all_data.loc[:,'type'] == 'CASH_OUT',:],ignore_index=True)
len(new_filterdata)
print new_filterdata.keys()


# In[13]:


print set(new_filterdata.type)


# In[14]:


new_filterdata.head()


# In[15]:


copydata = copy.deepcopy(new_filterdata)


# In[16]:


copydata.head()


# # Class Unbalanced

# In[17]:


print "Total samples in two types: ",len(copydata)
print "Total samples of fraud: ",len(copydata.loc[copydata['isFraud']== 1]) 
print "Total samples of not fraud: ",len(copydata.loc[copydata['isFraud']== 0]) 


# # One hot encoding of Type- Cashout and Transfer

# In[18]:


copydata['CASH_OUT']=0
copydata['TRANSFER']=0
copydata.loc[copydata['type'] == 'TRANSFER', "TRANSFER"] = 1
copydata.loc[copydata['type'] == 'CASH_OUT', "CASH_OUT"] = 1


# In[19]:


copydata.head()


# In[20]:


copydata.tail()


# In[21]:


print len(copydata.loc[copydata['type']== 'CASH_OUT'])
print len(copydata.loc[copydata['CASH_OUT']== 1   ])
print len(copydata.loc[copydata['type']== 'TRANSFER'] )
print len(copydata.loc[copydata['TRANSFER']== 1  ] )


# In[22]:


copydata_new = pd.DataFrame()
copydata_new = copydata.drop('type',axis=1)


# In[23]:


copydata_new.head()


# In[24]:


# copydata = copy.deepcopy(copy2)
unique_nameOrig = set(copydata_new.nameOrig)
unique_nameDest = set(copydata_new.nameDest)


# In[25]:


print len(unique_nameOrig)
print len(unique_nameDest)


# # Predicting all in one class

# In[26]:


y_train = copydata_new['isFraud']
y_fraud = np.ones(len(copydata_new))
x = np.zeros(len(copydata_new))
new = np.concatenate((y_fraud.T,x)).T
y_nonfraud = np.zeros(len(copydata_new))


# In[27]:


print "Predicting all as fraud:  Accuracy = ",accuracy_score(y_train,y_fraud), " AUC = ",roc_auc_score(y_train, y_fraud)
#print "Predicting all as not fraud:  Accuracy = ",accuracy_score(y_train,y_nonfraud), " AUC = ",roc_auc_score(y_train, y_nonfraud)


# # Remooving nameOrig and DestOrig

# In[28]:


filter_data = pd.DataFrame()
filter_data= copydata_new.drop('isFraud',axis=1)
filter_data= filter_data.drop('nameOrig',axis=1)
filter_data= filter_data.drop('nameDest',axis=1)


# In[29]:


filter_data.head()


# # SMOTE

# In[30]:


sm = SMOTE(ratio='minority', random_state=42)
X_res, y_res = sm.fit_sample(filter_data, y_train)


# In[184]:


filter_data.head()


# In[85]:


# print X_res.shape


# In[86]:


# print np.count_nonzero(y_res)


# In[35]:


X_res,y_res = shuffle(X_res,y_res)


# In[78]:


reports(GaussianNB(),X_res,y_res,train_test_split_ratio=0.2)


# In[108]:


reports(knn,X_res,y_res,train_test_split_ratio=0.2)


# # SMOTE END

# # 5 Feature with SMOTE

# In[90]:


filter_data_5f = filter_data.drop(['step','newbalanceOrig','newbalanceDest'],axis=1)


# In[99]:


filter_data_5f.head()


# In[91]:


sm = SMOTE(ratio='minority', random_state=42)
X_res_5f, y_res_5f = sm.fit_sample(filter_data_5f, y_train)


# In[92]:


X_res_5f,y_res_5f = shuffle(X_res_5f,y_res_5f)


# In[93]:


reports(GaussianNB(),X_res_5f,y_res_5f,train_test_split_ratio=0.2)


# In[104]:





# In[106]:


reports(knn,X_res_5f,y_res_5f,train_test_split_ratio=0.2)


# # 5 Feature with SMOTE END

# # 3 Feature with SMOTE

# In[94]:


filter_data_3f = filter_data.drop(['step','newbalanceOrig','newbalanceDest','CASH_OUT','TRANSFER'],axis=1)


# In[98]:


filter_data_3f.head()


# In[95]:


sm = SMOTE(ratio='minority', random_state=42)
X_res_3f, y_res_3f = sm.fit_sample(filter_data_3f, y_train)


# In[96]:


X_res_3f,y_res_3f = shuffle(X_res_3f,y_res_3f)


# In[97]:


reports(GaussianNB(),X_res_3f,y_res_3f,train_test_split_ratio=0.2)


# # 3 Feature with SMOTE END

# In[116]:


# import scikitplot as skplt


# In[111]:


# !pip install scikitplot


# # Scaling of all attributes

# In[279]:


from sklearn import preprocessing


# In[209]:


# filter_data_array = preprocessing.scale(filter_data)


# # Training Accuracy 80-20 split

# In[210]:


# from sklearn.model_selection import train_test_split, learning_curve
# from sklearn.utils import shuffle
# train_data,train_labels = shuffle(filter_data_array,y_train)
# trainX, testX, trainY, testY = train_test_split(train_data, train_labels, test_size = 0.2)


# In[211]:


# from sklearn.naive_bayes import GaussianNB
# from sklearn.metrics import accuracy_score

# clf = GaussianNB()
# clf = clf.fit(trainX,trainY)
# y = clf.predict(testX)
# print "Accuracy on Train Data 80-20: " +str(accuracy_score(y,testY))


# In[212]:


# trainX[0:5]


# In[213]:


# mett = metrics.classification_report(testY,y)
# print mett


# # random under sampling

# In[214]:


index_nonfraud = copydata_new.index[copydata_new['isFraud'] == 0 ].tolist()

random_index_nonfraud = np.random.choice(index_nonfraud,len(copydata.loc[copydata_new['isFraud']== 1]),replace=False)

index_fraud = copydata_new.index[copydata_new['isFraud'] == 1 ].tolist()

random_index_fraud = np.random.choice(index_fraud,len(copydata.loc[copydata_new['isFraud']== 1]),replace=False)

print len((random_index_fraud)),len((random_index_nonfraud))


# In[215]:


print "Ratio of Fraud in sampled data: ",len(random_index_fraud)/float(len(random_index_fraud)+len(random_index_nonfraud))
print "Ratio of Not Fraud in sampled data: ",len(random_index_nonfraud)/float(len(random_index_fraud)+len(random_index_nonfraud))


# In[216]:


index_req = np.concatenate((np.array(random_index_fraud),np.array(random_index_nonfraud)))
sampled_data = filter_data.iloc[index_req,:]
sampled_labels = []
for i in range(0,len(index_req)):
    sampled_labels.append( y_train.iloc[index_req[i]] )


sampled_labels=np.array(sampled_labels)
print len(sampled_data),len(sampled_labels)


# In[217]:


sampled_data.shape


# In[218]:


import keras


# In[291]:


from keras.layers import Input,Dense
from keras.models import Model,load_model


# In[220]:


encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats


# In[267]:


# this is our input placeholder
input_img = Input(shape=(None,8))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(8, activation='relu')(encoded)


# In[268]:


sampled_data, sampled_labels = shuffle(sampled_data, sampled_labels)


# In[342]:


trainX, testX, trainY, testY = train_test_split(preprocessing.scale(sampled_data), sampled_labels, test_size = 0.2,shuffle=False)


# In[343]:


trainX.shape


# In[344]:


# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)


# In[345]:


autoencoder.compile(optimizer='adam', loss='MSE')


# In[346]:


trainX3d = np.array(trainX).reshape(len(trainX),1,8)
testX3d = np.array(testX).reshape(len(testX),1,8)


# In[347]:


# print a[0]
# trainX.head(1)


# In[348]:


# trainX = np.reshape(trainX,(np.shape(trainX)[0],1,8))
# testX = np.reshape(testX, (np.shape(testX)[0],1,8))


# In[349]:


auto=autoencoder.fit(np.array(trainX3d), np.array(trainX3d),
                epochs=5,
                batch_size=3,
                shuffle=False,
                validation_data=(np.array(testX3d), np.array(testX3d)))


# In[350]:


autoencoder.save("auto")


# In[351]:


decoder_layer = autoencoder.layers[-1]


# In[352]:


# decoder = Model(encoded_input, decoder_layer(encoded_input))


# In[353]:


AE = load_model('auto')


# In[354]:


# traindadata = trainX3d


# In[355]:


# data = np.ones((1,1,8))
encoder = Model(inputs = AE.input, outputs = AE.layers[1].output)
trainX3dout = encoder.predict(trainX3d)
# print(output)
# data = np.ones((1,1,8))
# encoder = Model(inputs = AE.input, outputs = AE.layers[1].output)
testX3dout = encoder.predict(testX3d)
# print(output)


# In[356]:


trainX3dout = trainX3dout.reshape(len(trainX3dout),32)
testX3dout = testX3dout.reshape(len(testX3dout),32)


# In[357]:


trainX3dout.shape
testX3dout.shape


# In[358]:


# trainX3dout.reshape(len(trainX3dout),32)


# # Auto encoder 32 features RUS

# In[359]:


reports_test(GaussianNB(),trainX3dout,trainY,testX3dout,testY)


# In[360]:


reports_test(knn,trainX3dout,trainY,testX3dout,testY)


# In[361]:


reports_test(LogisticRegression(),trainX3dout,trainY,testX3dout,testY)


# In[362]:


reports_test(randomforest,trainX3dout,trainY,testX3dout,testY)


# In[ ]:





# In[379]:


scatterplot(trainX3dout,trainY)


# # 5 features RUC

# In[366]:


trainX5d, testX5d, trainY5d, testY5d = train_test_split(preprocessing.scale(sampled_data.drop(['step','newbalanceOrig','newbalanceDest'],axis=1)), sampled_labels, test_size = 0.2,shuffle=False)


# In[367]:


reports_test(GaussianNB(),trainX5d,trainY5d,testX5d,testY5d)


# In[368]:


reports_test(knn,trainX5d,trainY5d,testX5d,testY5d)


# In[369]:


reports_test(LogisticRegression(),trainX5d,trainY5d,testX5d,testY5d)


# In[370]:


reports_test(randomforest,trainX5d,trainY5d,testX5d,testY5d)


# In[381]:


scatterplot(trainX5d,trainY5d)


# # 8 features RUC

# In[365]:


trainX8d, testX8d, trainY8d, testY8d = train_test_split(preprocessing.scale(sampled_data), sampled_labels, test_size = 0.2,shuffle=False)


# In[372]:


reports_test(GaussianNB(),trainX8d,trainY8d,testX8d,testY8d)


# In[374]:


reports_test(knn,trainX8d,trainY8d,testX8d,testY8d)


# In[375]:


reports_test(LogisticRegression(),trainX8d,trainY8d,testX8d,testY8d)


# In[376]:


reports_test(randomforest,trainX8d,trainY8d,testX8d,testY8d)


# In[380]:


scatterplot(trainX8d,trainY8d)


# In[ ]:




