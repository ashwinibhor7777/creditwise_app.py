#!/usr/bin/env python
# coding: utf-8

# # CreditWise: Loan Approval Prediction System

# In[ ]:


Built an end-to-end supervised machine learning pipeline to predict loan approval using binary classification techniques. The project includes exploratory data analysis (EDA), feature engineering, and data preprocessing (handling missing values, feature scaling, and one-hot encoding).

Implemented and compared multiple classification models including Naive Bayes, Logistic Regression, and K-Nearest Neighbors (KNN). Model performance was evaluated using precision, recall, and F1-score, with special emphasis on handling class imbalance, where accuracy alone is insufficient.

Naive Bayes achieved the best overall performance on the imbalanced dataset, while Logistic Regression showed competitive results when balancing precision and recall. KNN performed poorly due to sensitivity to class imbalance and high dimensionality caused by one-hot encoding.


# In[1]:


import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# In[2]:


loan_df=pd.read_csv("loan_approval_data.csv")


# In[3]:


loan_df


# In[4]:


loan_df.head()


# In[5]:


loan_df.info()


# In[6]:


loan_df.isnull().sum()


# In[7]:


loan_df.describe


# In[8]:


# data clearing 
##handaling missing value


# In[9]:


categorical_cols=loan_df.select_dtypes(include=["object"]).columns
numerical_cols=loan_df.select_dtypes(include=["float64"]).columns# int64,numberalso use 



# In[10]:


categorical_cols


# In[11]:


categorical_cols.size


# In[12]:


numerical_cols


# In[13]:


numerical_cols.size


# In[14]:


categorical_cols.size+numerical_cols.size


# In[15]:


from sklearn.impute import SimpleImputer
#Define categorical columns
numerical_cols=loan_df.select_dtypes(include=["float64"]).columns# int64,numberalso use 
#Create the imputer
numerical_imp=SimpleImputer(strategy="mean")
#Apply imputation
loan_df[numerical_cols]=numerical_imp.fit_transform(loan_df[numerical_cols])


# In[16]:


loan_df.head()


# In[17]:


from sklearn.impute import SimpleImputer
#Define categorical columns
categorical_cols=loan_df.select_dtypes(include=["object"]).columns
#Create the imputer
categorical_imp=SimpleImputer(strategy="most_frequent")
#Apply imputation
loan_df[categorical_cols]=categorical_imp.fit_transform(loan_df[categorical_cols])


# In[18]:


loan_df.head()
loan_df.isnull().sum()


# # EDA 

# In[19]:


## how balanced our classes are?

calsses_count=loan_df["Loan_Approved"].value_counts()
plt.pie(calsses_count,labels=["no","yes"],autopct="%1.1f%%")
plt.title("is loan approved or not?")


# In[20]:


## analyze categories with gender value 


# In[21]:


gender_count=loan_df["Gender"].value_counts()
ax=sns.barplot(gender_count)
ax.bar_label(ax.containers[0])


# In[22]:


## analyze categories with Education_Level value 


# In[23]:


Education_Level=loan_df["Education_Level"].value_counts()
ax=sns.barplot(Education_Level,color="#cf2bc9")
ax.set_title("Education Level")
ax.set_xlabel("Education Level")
ax.set_ylabel("Count")
ax.bar_label(ax.containers[0])
plt.show()


# In[24]:


## analyze categories with Property_Area     
Property_Area =loan_df["Property_Area"].value_counts()
ax=sns.barplot(Property_Area,color="green")
ax.set_title("Property_Area ")
ax.set_xlabel("Property_Area ")
ax.set_ylabel("Count")
ax.bar_label(ax.containers[0])
plt.show()


# In[25]:


## analyze categories with Loan_Purpose
Loan_Purpose =loan_df["Loan_Purpose"].value_counts()
ax=sns.barplot(Loan_Purpose, color="orange"  )
ax.set_title("Loan_Purpose ")
ax.set_xlabel("Loan_Purpose ")
ax.set_ylabel("Count")
ax.bar_label(ax.containers[0])
plt.show()


# In[26]:


## analyze categories with Loan_Amount 
Loan_Amount =loan_df["Loan_Amount"].value_counts()
ax = sns.histplot(
    loan_df["Loan_Amount"],
    bins=10,
    color="#404fa3"

  )
ax.set_title("Loan_Amount ")
ax.set_xlabel("Loan_Amount ")
ax.set_ylabel("Count")
ax.bar_label(ax.containers[0])
plt.show()


# In[27]:


## analyze categories with Loan_Term  
Loan_Term   =loan_df["Loan_Term"].value_counts()
ax=sns.barplot(Loan_Term,color="pink"  )
ax.set_title("Loan_Term")
ax.set_xlabel("Loan_Term")
ax.set_ylabel("Count")
ax.bar_label(ax.containers[0])
plt.show()


# In[28]:


# analyze income


# In[29]:


sns.histplot(
    data=loan_df,
    x="Applicant_Income",
    bins=20
            )


# In[30]:


sns.histplot(
    data=loan_df,
    x="Coapplicant_Income",
    bins=20,
    color="pink"
            )


# In[31]:


# outliner-box plots
sns.boxplot(
    data=loan_df,
    x="Loan_Approved",
    y="Applicant_Income"

)


# In[32]:


fig, axes = plt.subplots(3, 2, figsize=(12,12))

sns.boxplot(ax=axes[0,0],data=loan_df,
            x="Loan_Approved",
            y="Applicant_Income")
sns.boxplot(ax=axes[0,1],data=loan_df,
            x="Loan_Approved",
            y="Credit_Score")
sns.boxplot(ax=axes[1,0],data=loan_df,
            x="Loan_Approved",
            y="DTI_Ratio")
sns.boxplot(ax=axes[1,1],data=loan_df,
            x="Loan_Approved",
            y="Savings")
sns.boxplot(ax=axes[2,0],data=loan_df,
            x="Loan_Approved",
            y="Age")
sns.boxplot(ax=axes[2,1],data=loan_df,
            x="Loan_Approved",
            y="Loan_Amount")
plt.tight_layout()


# In[33]:


# credit score with Loan_Approved

sns.histplot(
    data=loan_df,
    x="Credit_Score",
    hue="Loan_Approved",
    bins=20,
    multiple="dodge"
)


# In[34]:


# credit score with Applicant_Income

sns.histplot(
    data=loan_df,
    x="Applicant_Income",
    hue="Loan_Approved",
    bins=20,
    multiple="dodge"
)


# In[35]:


# remove applicant id:-Applicant_ID is an identifier, not a feature, so it was excluded to avoid misleading the model.”,“Bank gives ID so it is not needed”
loan_df=loan_df.drop("Applicant_ID",axis=1)


# In[36]:


loan_df.head()

## encoding:-mean features or out put  convering the numerical value
1.binary-useig maps 
2.onehotendcoding-get_dummeies
#now this one we are using the 
1.labelEncoder-assigns an integer to each category
2.OneHotEncoder -cretes binary columns for each category
# In[37]:


loan_df.head()


# In[38]:


loan_df.info()


# In[39]:


loan_df.columns


# In[40]:


from sklearn.preprocessing import LabelEncoder,OneHotEncoder
#LabelEncoder-in one columns val replace with f,m to 0,1 (ordinal-order is there on thst time we use this one )
# OneHotEncoder-in here each category create the new columns /according to category create a new cloumns (nominal-on order here)


# In[41]:


#LabelEncoder-in one columns val replace with f,m to 0,1 (ordinal-order is there on thst time we use this one )
le=LabelEncoder()
loan_df["Education_Level"]=le.fit_transform(loan_df["Education_Level"])
loan_df["Loan_Approved"]=le.fit_transform(loan_df["Loan_Approved"])

cols=["Employment_Status","Marital_Status","Loan_Purpose","Property_Area","Gender","Employer_Category"]


# In[42]:


loan_df.head()


# In[43]:


# OneHotEncoder-in here each category create the new columns /according to category create a new cloumns (nominal-on order here)

cols=["Employment_Status","Marital_Status","Loan_Purpose","Property_Area","Gender","Employer_Category"]

ohe=OneHotEncoder(drop="first",sparse_output=False,handle_unknown="ignore")# parameteres are drop,handle_unknown,sparse_output

encoded=ohe.fit_transform(loan_df[cols])

#2d array converting in to datafrom  
encoded_df=pd.DataFrame(encoded,columns=ohe.get_feature_names_out(cols),index=loan_df.index)
# append the data with origion data 
loan_df=pd.concat([loan_df.drop(columns=cols),encoded_df],axis=1)


# In[44]:


encoded_df.head()


# In[45]:


encoded


# In[46]:


loan_df.head()
loan_df.info()


# In[47]:


# correlation heatmap

# correlation heatmap
:-it is a representation of the relationships betn numerical variables in dataset.
it show correlation coefficient (r) betn two numeric variables
.range form -1 to 1
1 is perfect positive correlation
.-1is perfect negative correlation 
.0 is no linear correlation

# In[48]:


nums_cols=loan_df.select_dtypes(include="number")
corr_matrix = nums_cols.corr()

#heatmap:-benefits using this is the 1. i sthe quick insight get 2. detect multi collinearity mean feature of correlation remove3.data exploration 4. pre processing  
plt.figure(figsize=(15,8))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt="2f",
    cmap="coolwarm"

)


# In[49]:


nums_cols.corr()["Loan_Approved"].sort_values(ascending=False)


# ## train_test_split the data 

# In[50]:


X=loan_df.drop("Loan_Approved",axis=1)
y=loan_df["Loan_Approved"]


# In[51]:


X.head()


# In[52]:


y.head()


# In[53]:


X_train,X_test ,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=42)


# In[54]:


X_test.head()


# In[55]:


X_train.head()


# In[56]:


# feature scaling:- is important for the Logistic Regression ,KNN ✅ SVM ✅ Gradient Descent–based models ✅
from sklearn.preprocessing import StandardScaler # import from sklearn 

# create a scalled
scaler=StandardScaler()

# Fit the scaler on training data and transform it
X_train_scaled =scaler.fit_transform(X_train)

# Transform the test data using the same scaler (no fitting!)
X_test_scaled=scaler.transform(X_test)


# In[57]:


X_train_scaled


# In[58]:


X_test_scaled


# In[59]:


## train and evaluate models 


# In[60]:


# logistic regression


# In[61]:


# import LogisticRegression form sklearn website

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score,recall_score,confusion_matrix,accuracy_score,f1_score
# create the model
log_model=LogisticRegression()
# fit LogisticRegression
log_model.fit(X_train_scaled,y_train)
# # now prediction 
# y_pred=log_model.predict(X_test_scaled)

# evaluation:-
# Model Evaluation:
# For this problem, precision and recall are more important than accuracy.
# In a loan approval system, accuracy alone can be misleading because the
# dataset may be imbalanced.

# We focus on:
# - Precision: How many predicted approved loans are actually approved
# - Recall: How many truly eligible applicants are correctly approved

# This aligns better with the problem statement than using accuracy alone.
# F1-score is also important because it provides a balance between
# precision and recall, making it a better overall evaluation metric
# for this model than accuracy.
# print("logistic regression model")
# print("precision:",precision_score(y_test,y_pred))
# print("recall:",recall_score(y_test,y_pred))
# print("confusion:",confusion_matrix(y_test,y_pred))
# print("f1:",f1_score(y_test,y_pred))
# print("accuracy:",accuracy_score(y_test,y_pred))




# In[62]:


# now prediction 
y_pred=log_model.predict(X_test_scaled)


# In[63]:


# evaluation:-
from sklearn.metrics import precision_score,recall_score,confusion_matrix,accuracy_score,f1_score
print("logistic regression model")
print("precision:",precision_score(y_test,y_pred))
print("recall:",recall_score(y_test,y_pred))
print("confusion:",confusion_matrix(y_test,y_pred))
print("f1:",f1_score(y_test,y_pred))
print("accuracy:",accuracy_score(y_test,y_pred))



# In[64]:


# import KNeighborsClassifier form sklearn website
from sklearn.neighbors import KNeighborsClassifier
# create the model
knn_model=KNeighborsClassifier(n_neighbors=5)
# fit LogisticRegression
knn_model.fit(X_train_scaled,y_train)
# # now prediction 
# y_pred=knn_model.predict(X_test_scaled)




# In[65]:


# now prediction 
y_pred=knn_model.predict(X_test_scaled)


# In[66]:


# evaluation:-
from sklearn.metrics import precision_score,recall_score,confusion_matrix,accuracy_score,f1_score

print("knn model")
print("precision:",precision_score(y_test,y_pred))
print("recall:",recall_score(y_test,y_pred))
print("confusion:",confusion_matrix(y_test,y_pred))
print("f1:",f1_score(y_test,y_pred))
print("accuracy:",accuracy_score(y_test,y_pred))


# In[67]:


# import naive_bayes form sklearn website
from sklearn.naive_bayes import GaussianNB
# create the model
nb_model=GaussianNB()
# fit LogisticRegression
nb_model.fit(X_train_scaled,y_train)
# # now prediction 


# In[68]:


# now prediction 
y_pred=nb_model.predict(X_test_scaled)


# In[69]:


# evaluation:-
from sklearn.metrics import precision_score,recall_score,confusion_matrix,accuracy_score,f1_score

print("nb  model")
print("precision:",precision_score(y_test,y_pred))
print("recall:",recall_score(y_test,y_pred))
print("confusion:",confusion_matrix(y_test,y_pred))
print("f1:",f1_score(y_test,y_pred))
print("accuracy:",accuracy_score(y_test,y_pred))


# In[70]:


# best  is nb for the loan approved here this is best model 


# In[71]:


# feature engineering 


# In[72]:


# add or transform features
loan_df["DTI_Ratio_sq"]=loan_df["DTI_Ratio"]**2
loan_df["Credit_Score_sq"]=loan_df["Credit_Score"]**2
# skewed some data 
loan_df["Applicant_Income_log"]=np.log1p(loan_df["Applicant_Income"])
# calu x,y
X=loan_df.drop(columns=["Loan_Approved","Credit_Score","DTI_Ratio"])
y=loan_df["Loan_Approved"]
# train test split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

# feature scaling
# create a scalled
scaler=StandardScaler()

# Fit the scaler on training data and transform it
X_train_scaled =scaler.fit_transform(X_train)

# Transform the test data using the same scaler (no fitting!)
X_test_scaled=scaler.transform(X_test)


# In[73]:


X_train.head()


# In[74]:


# logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score,recall_score,confusion_matrix,accuracy_score,f1_score
# create the model
log_model=LogisticRegression()
# fit LogisticRegression
log_model.fit(X_train_scaled,y_train)
# now prediction 
y_pred=log_model.predict(X_test_scaled)
# evaluation:-
print("logistic regression model")
print("precision:",precision_score(y_test,y_pred))
print("recall:",recall_score(y_test,y_pred))
print("confusion:",confusion_matrix(y_test,y_pred))
print("f1:",f1_score(y_test,y_pred))
print("accuracy:",accuracy_score(y_test,y_pred))


# In[75]:


# import KNeighborsClassifier form sklearn website
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score,recall_score,confusion_matrix,accuracy_score,f1_score
# create the model
knn_model=KNeighborsClassifier(n_neighbors=5)
# fit LogisticRegression
knn_model.fit(X_train_scaled,y_train)
# now prediction 
y_pred=knn_model.predict(X_test_scaled)
# evaluation:-
print("knn model")
print("precision:",precision_score(y_test,y_pred))
print("recall:",recall_score(y_test,y_pred))
print("confusion:",confusion_matrix(y_test,y_pred))
print("f1:",f1_score(y_test,y_pred))
print("accuracy:",accuracy_score(y_test,y_pred))



# In[76]:


# import naive_bayes form sklearn website
from sklearn.naive_bayes import GaussianNB
# create the model
nb_model=GaussianNB()
# fit LogisticRegression
nb_model.fit(X_train_scaled,y_train)
# now prediction 
y_pred=nb_model.predict(X_test_scaled)
# evaluation:-
from sklearn.metrics import precision_score,recall_score,confusion_matrix,accuracy_score,f1_score

print("nb  model")
print("precision:",precision_score(y_test,y_pred))
print("recall:",recall_score(y_test,y_pred))
print("confusion:",confusion_matrix(y_test,y_pred))
print("f1:",f1_score(y_test,y_pred))
print("accuracy:",accuracy_score(y_test,y_pred))

Naive Bayes is the best model for this loan approval dataset due to class imbalance. Logistic Regression also performs well when precision and recall are balanced. KNN performs poorly because it is sensitive to class imbalance and high dimensionality caused by one-hot encoding.best one is the nb but we are going with precision_score and recall score balance on that time logistic  regression also we can use but nnb is best in loan approvel data set there is inbalance calss and 2nd one is the logistic  regression give good performance given knn not given the good performnace becuse of the class is inbalance ,and 2nd one is the onehotencodeing is so many feauters come on that time its fials,on that time it poverly work
# Model Selection Explanation:
# Naive Bayes (NB) performs best on this loan approval dataset because
# the data is imbalanced and NB handles class imbalance efficiently.

# We evaluate models using precision and recall, and aim to keep a
# good balance between them. Logistic Regression also performs well
# under this evaluation strategy and can be used as an alternative.

# KNN does not give good performance because it is sensitive to
# class imbalance and distance-based learning is affected by skewed data.

# Additionally, one-hot encoding creates a large number of features.
# This increases dimensionality, which causes KNN to perform poorly,
# while Naive Bayes and Logistic Regression handle this situation better.

# In[ ]:




