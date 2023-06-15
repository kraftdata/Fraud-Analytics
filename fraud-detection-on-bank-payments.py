#!/usr/bin/env python
# coding: utf-8

# # Fraud Detection on Bank Payments
# 
# ## Fraud and detecting it
# 
# Fraudulent activity can be observed in a variety of industries, including e-commerce, healthcare, payment systems, and banking. Fraud is a multibillion dollar industry that keeps growing. According to the PwC global economic crime report from 2018 [1], of the 7,200 organisations surveyed, half (49 percent) have encountered fraud of some type.
# 
# Fraud detection is possible even if it may seem frightening to organisations, thanks to intelligent solutions like rules engines and machine learning. The majority of individuals on Kaggle are familiar with machine learning, but this is some quick background on rule engines.
# A rules engine is a piece of software that carries out one or more business rules in a real-world runtime setting. For the purpose of conveying problem-specific knowledge to the rules engine and ultimately to production, these rules are typically created by domain experts. Limiting the amount of transactions per period (velocity rules) and disallowing transactions from previously identified fraudulent IP addresses and/or domains are two rules examples for fraud detection.
#     
# Rules are excellent at spotting various types of fraud, but because they have predefined threshold values, they may produce a large number of false positives or false negatives. Consider a rule that prevents a transaction with a value greater than $10,000 from being processed for a certain user. If this user has a history of fraud, he or she may be aware that the system has a threshold and can simply conduct a transaction that is slightly below the threshold value (9999 dollars).
# 
# ML can assist with these issues and lower the chance of fraud and the possibility that a firm will lose money. Fraud detection would be more accurate and confident with the use of rules and machine learning.
# 
# ## Banksim dataset
# 
# We use the Banksim dataset to identify fraudulent transactions. This artificially created dataset consists of payments from different consumers made over various time frames and in varying quantities. Check out the [Kaggle page](https://www.kaggle.com/ntnu-testimon/banksim1) for this dataset, which also includes a link to the original paper, for more details on the dataset.
# 
# Here what we'll do in this kernel:
# 1. [Exploratory Data Analysis (EDA)](#Explaratory-Data-Analysis)
# 2. [Data Preprocessing](#Data-Preprocessing)
# 3. [Oversampling with SMOTE](#Oversampling-with-SMOTE)
# 4. [K-Neighbours Classifier](#K-Neighbours-Classifier)
# 5. [Random Forest Classifier](#Random-Forest-Classifier)
# 6. [XGBoost Classifier](#XGBoost-Classifier)
# 7. [Conclusion](#Conclusion)

# ## Explaratory Data Analysis
# 
# In this chapter, we'll run an EDA on the data to see if we can learn anything.

# In[1]:


# Necessary imports

## Data loading, processing and for more
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE

## Visualization
import seaborn as sns
import matplotlib.pyplot as plt
# set seaborn style because it prettier
sns.set()

## Metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc

## Models
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier


# **Data**
# As we can see in the first rows below the dataset has 9 feature columns and a target column. 
# The feature columms are :
# * **Step**: This feature represents the day from the start of simulation. It has 180 steps so simulation ran for virtually 6 months.
# * **Customer**: This feature represents the customer id
# * **zipCodeOrigin**: The zip code of origin/source.
# * **Merchant**: The merchant's id
# * **zipMerchant**: The merchant's zip code
# * **Age**: Categorized age 
#     * 0: <= 18, 
#     * 1: 19-25, 
#     * 2: 26-35, 
#     * 3: 36-45,
#     * 4: 46:55,
#     * 5: 56:65,
#     * 6: > 65
#     * U: Unknown
# * **Gender**: Gender for customer
#      * E : Enterprise,
#      * F: Female,
#      * M: Male,
#      * U: Unknown
# * **Category**: Category of the purchase. I won't write all categories here, we'll see them later in the analysis.
# * **Amount**: Amount of the purchase
# * **Fraud**: Target variable which shows if the transaction fraudulent(1) or benign(0)

# In[2]:


# read the data and show first 5 rows
data = pd.read_csv("../input/bs140513_032310.csv")
data.head(5)


# Let's look at column types and missing values in data.  Oh im sorry there is **no** missing values which means we don't have to perform an imputation.

# In[3]:


data.info()


# **Fraud data** will be imbalanced like you see in the plot below and from the count of instances. To balance the dataset one can perform oversample or undersample techniques. Oversampling is increasing the number of the minority class by generating instances from the minority class . Undersampling is reducing the number of instances in the majority class by selecting random points from it to where it is equal with the minority class. Both operations have some risks: Oversample will create copies or similar data points which sometimes would not be helpful for the case of fraud detection because fraudulent transactions may vary. Undersampling means that we lost data points thus information. We will perform an oversampled technique called SMOTE (Synthetic Minority Over-sampling Technique). SMOTE will create new data points from minority class using the neighbour instances so generated samples are not exact copies but they are similar to instances we have.

# In[4]:


# Create two dataframes with fraud and non-fraud data 
df_fraud = data.loc[data.fraud == 1] 
df_non_fraud = data.loc[data.fraud == 0]

sns.countplot(x="fraud",data=data)
plt.title("Count of Fraudulent Payments")
plt.show()
print("Number of normal examples: ",df_non_fraud.fraud.count())
print("Number of fradulent examples: ",df_fraud.fraud.count())
#print(data.fraud.value_counts()) # does the same thing above


# We can see the mean amount and fraud percent by category below. Looks like leisure and the travel is the most selected categories for fraudsters. Fraudsters chose the categories which people spend more on average. Let's confirm this hypothesis by checking the fraud and non-fraud amount transacted.

# In[5]:


print("Mean feature values per category",data.groupby('category')['amount','fraud'].mean())


# Our hypothesis for fraudsters choosing the categories which people spend more is only partly correct, but as we can see in the table below we can say confidently say that a fraudulent transaction will be much more (about four times or more) than average for that category.

# In[6]:


# Create two dataframes with fraud and non-fraud data 
pd.concat([df_fraud.groupby('category')['amount'].mean(),df_non_fraud.groupby('category')['amount'].mean(),           data.groupby('category')['fraud'].mean()*100],keys=["Fraudulent","Non-Fraudulent","Percent(%)"],axis=1,          sort=False).sort_values(by=['Non-Fraudulent'])


# Average amount spend it categories are similar; between 0-500 discarding the outliers, except for the travel category which goes very high. 

# In[7]:


# Plot histograms of the amounts in fraud and non-fraud data 
plt.figure(figsize=(30,10))
sns.boxplot(x=data.category,y=data.amount)
plt.title("Boxplot for the Amount spend in category")
plt.ylim(0,4000)
plt.legend()
plt.show()


# Again we can see in the histogram below the fradulent transactions are less in count but more in amount.

# In[8]:


# Plot histograms of the amounts in fraud and non-fraud data 
plt.hist(df_fraud.amount, alpha=0.5, label='fraud',bins=100)
plt.hist(df_non_fraud.amount, alpha=0.5, label='nonfraud',bins=100)
plt.title("Histogram for fraudulent and nonfraudulent payments")
plt.ylim(0,10000)
plt.xlim(0,1000)
plt.legend()
plt.show()


# Looks like fraud occurs more in ages equal and below 18(0th category). Can it be because of fraudsters thinking it would be less consequences if they show their age younger, or maybe they really are young.

# In[9]:


print((data.groupby('age')['fraud'].mean()*100).reset_index().rename(columns={'age':'Age','fraud' : 'Fraud Percent'}).sort_values(by='Fraud Percent'))


# ## Data Preprocessing
# 
# In this part we will preprocess the data and prepare for the training.
# 
# There are only one unique zipCode values so we will drop them.

# In[10]:


print("Unique zipCodeOri values: ",data.zipcodeOri.nunique())
print("Unique zipMerchant values: ",data.zipMerchant.nunique())
# dropping zipcodeori and zipMerchant since they have only one unique value
data_reduced = data.drop(['zipcodeOri','zipMerchant'],axis=1)


# Checking the data after dropping.

# In[11]:


data_reduced.columns


# Here, we'll convert categorical characteristics into numbers. Since they are unrelated in size (e.g., customer1 is not greater than customer2) and there are too many of them (above 500k customers and merchants), it is usually preferable to convert these categorical values into dummies because training them would take an eternity. If you wish to attempt it, I've included the code for converting categorical characteristics into dummies below.
# > data_reduced.loc[:,['customer','merchant','category']].astype('category')
# > data_dum = pd.get_dummies(data_reduced.loc[:,['customer','merchant','category','gender']],drop_first=True) # dummies
# > print(data_dum.info())

# In[12]:


# turning object columns type to categorical for easing the transformation process
col_categorical = data_reduced.select_dtypes(include= ['object']).columns
for col in col_categorical:
    data_reduced[col] = data_reduced[col].astype('category')
# categorical values ==> numeric values
data_reduced[col_categorical] = data_reduced[col_categorical].apply(lambda x: x.cat.codes)
data_reduced.head(5)


# Let's define our independent variable (X) and dependant/target variable y

# In[13]:


X = data_reduced.drop(['fraud'],axis=1)
y = data['fraud']
print(X.head(),"\n")
print(y.head())


# In[14]:


y[y==1].count()


# ## Oversampling with SMOTE
# 
# Using SMOTE(Synthetic Minority Oversampling Technique) [2] for balancing the dataset. Resulted counts show that now we have exact number of class instances (1 and 0).

# In[15]:


sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)
y_res = pd.DataFrame(y_res)
print(y_res[0].value_counts())


# I'll run a train test split to gauge the effectiveness. Since there are many instances and I don't want to wait too long for training, I haven't done cross validation, although it should be preferable most of the time.

# In[16]:


# I won't do cross validation since we have a lot of instances
X_train, X_test, y_train, y_test = train_test_split(X_res,y_res,test_size=0.3,random_state=42,shuffle=True,stratify=y_res)


# I will define a function for plotting the ROC_AUC curve. It is a good visual way to see the classification performance.

# In[17]:


# %% Function for plotting ROC_AUC curve

def plot_roc_auc(y_test, preds):
    '''
    Takes actual and predicted(probabilities) as input and plots the Receiver
    Operating Characteristic (ROC) curve
    '''
    fpr, tpr, threshold = roc_curve(y_test, preds)
    roc_auc = auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


# As i talked about it before fraud datasets will be imbalanced and most of the instances will be non-fraudulent. Imagine that we have the dataset here and we are always predicting non-fraudulent. Our accuracy would be almost 99 % for this dataset and mostly for others as well since fraud percentage is very low. Our accuracy is very high but we are not detecting any frauds so it is a useless classifier. So the base accuracy score should be better at least than predicting always non-fraudulent for performing a detection.

# In[18]:


# The base score should be better than predicting always non-fraduelent
print("Base accuracy score we must beat is: ", 
      df_non_fraud.fraud.count()/ np.add(df_non_fraud.fraud.count(),df_fraud.fraud.count()) * 100)


# ## **K-Neighbours Classifier**

# In[19]:


# %% K-ello Neigbors

knn = KNeighborsClassifier(n_neighbors=5,p=1)

knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)


print("Classification Report for K-Nearest Neighbours: \n", classification_report(y_test, y_pred))
print("Confusion Matrix of K-Nearest Neigbours: \n", confusion_matrix(y_test,y_pred))
plot_roc_auc(y_test, knn.predict_proba(X_test)[:,1])


# ## **Random Forest Classifier**

# In[20]:


# %% Random Forest Classifier

rf_clf = RandomForestClassifier(n_estimators=100,max_depth=8,random_state=42,
                                verbose=1,class_weight="balanced")

rf_clf.fit(X_train,y_train)
y_pred = rf_clf.predict(X_test)

print("Classification Report for Random Forest Classifier: \n", classification_report(y_test, y_pred))
print("Confusion Matrix of Random Forest Classifier: \n", confusion_matrix(y_test,y_pred))
plot_roc_auc(y_test, rf_clf.predict_proba(X_test)[:,1])


# ## XGBoost Classifier

# In[21]:


XGBoost_CLF = xgb.XGBClassifier(max_depth=6, learning_rate=0.05, n_estimators=400, 
                                objective="binary:hinge", booster='gbtree', 
                                n_jobs=-1, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0, 
                                subsample=1, colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1, 
                                scale_pos_weight=1, base_score=0.5, random_state=42, verbosity=True)

XGBoost_CLF.fit(X_train,y_train)

y_pred = XGBoost_CLF.predict(X_test)

print("Classification Report for XGBoost: \n", classification_report(y_test, y_pred))
print("Confusion Matrix of XGBoost: \n", confusion_matrix(y_test,y_pred))
plot_roc_auc(y_test, XGBoost_CLF.predict_proba(X_test)[:,1])


# ## Conclusion
# 
# We attempted to perform fraud detection on bank payment data in this kernel, and our classifiers produced outstanding results. We used the SMOTE oversampling technique to create additional minority class cases since fraud datasets have an imbalance class problem.
# 
# I appreciate you taking the time to read or simply examine the outcomes of my first kernel; I hope you found it interesting. I would appreciate any feedback, suggestions, or comments, and I hope you have a wonderful day filled with plenty of lovely information.
