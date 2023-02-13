#!/usr/bin/env python
# coding: utf-8

# #                     Bank Term Deposit Marketing

# ![micheile-dot-com-ZVprbBmT8QA-unsplash.jpg](attachment:micheile-dot-com-ZVprbBmT8QA-unsplash.jpg)

#  Bank Marketing :It is known for its nature of developing a unique brand image, which is treated as the capital reputation of the financial academy. It is very important for a bank to develop good relationship with valued customers accompanied by innovative ideas which can be used as measures to meet their requirements.
# 
# Customers expect quality services and returns. There are good chances that the quality factor will be the sole determinant of successful banking corporations. Therefore, Indian banks need to acknowledge the imperative of proactive Bank Marketing and Customer Relationship Management and also take systematic steps in this direction.

# What is a Term Deposit ?
# 
#    A time deposit or term deposit is a deposit in a financial institution with a specific maturity date or a period to maturity, commonly referred to as its "term". Time deposits differ from at call deposits, such as savings or checking accounts, which can be withdrawn at any time, without any notice or penalty. Deposits that require notice of withdrawal to be given are effectively time deposits, though they do not have a fixed maturity date
# 
#   A term deposit is a fixed-term investment that includes the deposit of money into an account at a financial institution. Term deposit investments usually carry short-term maturities ranging from one month to a few years and will have varying levels of required minimum deposits.
# 
# The investor must understand when buying a term deposit that they can withdraw their funds only after the term ends. In some cases, the account holder may allow the investor early termination or withdrawal if they give several days notification. Also, there will be a penalty assessed for early termination.
# 
# 

# Key Takeways:
#            
#   *A term deposit is a type of deposit account held at a financial institution where money is locked up for some set period of time.
#   
#   *Term deposits are usually short-term deposits with maturities ranging from one month to a few years.
#   
#   *Typically, term deposits offer higher interest rates than traditional liquid savings accounts, whereby customers can withdraw their money at any time.
#      
#      

# Objective:
#     
#    Business goal: Reducing marketing resources by identifying customers who would subscribe to term deposit and thereby direct marketing efforts to them.
# 
# 

# ### y-has the client subscribed a term deposit? (binary: 'yes','no')**
# 
# 

# # Features

# The dataset has the following attributes:
# 
# 1 .age (numeric)
# 
# 2 .job : type of job (categorical: "admin.","unknown","unemployed","management","housemaid","entrepreneur","student","blue-collar","self-employed","retired","technician","services")
# 
# 3 .marital : marital status (categorical: "married","divorced","single"; note: "divorced" meansdivorced or widowed)
# 
# 4 .education (categorical: "unknown","secondary","primary","tertiary")
# 
# 5 .default: has credit in default? (binary: "yes","no")
# 
# 6 .balance: average yearly balance, in euros (numeric)
# 
# 7 .housing: has housing loan? (binary: "yes","no")
# 
# 8 .loan: has personal loan? (binary: "yes","no")
# 
# 9 .contact: contact communication type (categorical: "unknown","telephone","cellular")
# 
# 10 .day: last contact day of the month (numeric)
# 
# 11 .month: last contact month of year (categorical: "jan", "feb", "mar", ..., "nov", "dec")
# 
# 12 .duration: last contact duration, in seconds (numeric)
# 
# 13 .campaign: number of contacts performed during this campaign and for this client (numeric,includes last contact)
# 
# 14 .pdays: number of days that passed by after the client was last contacted from a previouscampaign (numeric, -1 means client was not previously contacted)
# 
# 15 .previous: number of contacts performed before this campaign and for this client (numeric)
# 
# 16 .poutcome: outcome of the previous marketing campaign (categorical:"unknown","other","failure","success")
# 
# Target Variable:
# 
# 1 .y : has the client subscribed to a term deposit?(binary: "yes","no")
# 

# #  PROBLEM STATEMENT

# The data is related with direct marketing campaigns (phone calls) of a Portuguese banking institution. The classification goal is to predict if the client will subscribe (yes/no) a term deposit (variable y).

# # Importing Libraries

# In[786]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.offline import init_notebook_mode
import seaborn as sns
import datetime as dt
import warnings
import plotly.graph_objects as go
warnings.filterwarnings('ignore')


# In[787]:


df=pd.read_csv("bank.csv")


# In[788]:


df


# ###                                                            Data information

# In[789]:


df.info()


# # Explore and visualize the features
# 
# 

# ### Load the dataset
# 

# In[790]:


df=pd.read_csv("bank.csv")


# In[791]:


#show the top 5 records
df.head()


# In[792]:


#show the bottom 5 records
df.tail()


# In[793]:


#show the number of rows and columns
df.shape


# In[794]:


#check if null value present(0)
df.isnull().sum()


# In[795]:


#shows datatype and null values present for all columns
df.info() 


# In[796]:


#shows no.of unique values per column
df.nunique()


# In[797]:


#shows statistical summary for numerical columns
df.describe()


# ### Data Preprocessing: Label Encoder

# In[798]:


#converts categorical columns to numeric format
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['n_deposit']=le.fit_transform(df['deposit'])


# In[799]:


df


# In[800]:


#deleted 2 columns from dataset 
df.drop(['deposit','pdays'],axis=1,inplace=True)
df


# In[801]:


#checks no. of values in deposit column(balanced data)
df.n_deposit.value_counts()


# ### Categorical Feature Distribution

# In[802]:


#balanced output data
sns.countplot(x='n_deposit',data=df)


# In[803]:


#shows unique values for all columns
for col in df.select_dtypes(include='object').columns:
  print(col)
  print(df[col].unique()) 


# #### Client with job type as management records are high and housemaid are very less

# In[804]:


#client with job profile managemnent is higher and less for housemaid.
sns.countplot(y='job',data=df)


# In[805]:


#client who married are high in records and divorced are less
sns.countplot(y='marital',data=df)


# In[806]:


#client whoes education background is secondary are in high numbers
sns.countplot(y='education',data=df)


# In[807]:


#defualt feature looks unimportant  as it has value of no at high ratio to value yes(highly imbalance) which can drop
sns.countplot(y='default',data=df)


# In[808]:


##balance in housing loan for clients
sns.countplot(y='housing',data=df)


# In[809]:


#high no. of clients with no personal loan
sns.countplot(y='loan',data=df)


# In[810]:


#contact through
sns.countplot(y='contact',data=df)


# In[811]:


#monthly report
#data in month of may is high and less in dec
sns.countplot(y='month',data=df)


# In[812]:


#poutcome
sns.countplot(y='poutcome',data=df)


# In[813]:


#applying label encoder on categorical columns
df['n_job']=le.fit_transform(df['job'])
df['n_marital']=le.fit_transform(df['marital'])
df['n_education']=le.fit_transform(df['education'])
df['n_loan']=le.fit_transform(df['loan'])
df['n_contact']=le.fit_transform(df['contact'])
df['n_month']=le.fit_transform(df['month'])
df['n_poutcome']=le.fit_transform(df['poutcome'])
df['n1_deposit']=df['n_deposit']


# In[814]:


df.drop(['n_deposit'],axis=1,inplace=True)
df


# In[815]:


df.drop(['job','month','marital','education','default','housing','loan','contact','poutcome'],axis=1,inplace=True)


# In[816]:


df


# In[817]:


df.head()


# In[818]:


df.dtypes


# In[819]:


#shows all uniquevalues per column
for col in df.select_dtypes(include='int64').columns:
  print(col)
  print(df[col].unique()) 


# In[820]:


df.describe()


# # Outlier Detection

# In[821]:


figure, axis = plt.subplots(3, 4, figsize = (50,25))
sns.boxplot(x='n1_deposit',y='age',data=df,ax=axis[0,0])
sns.boxplot(x='n1_deposit',y='balance',data=df,ax=axis[0,1])
sns.boxplot(x='n1_deposit',y='duration',data=df,ax=axis[0,2])
sns.boxplot(x='n1_deposit',y='campaign',data=df,ax=axis[0,3])
sns.boxplot(x='n1_deposit',y='previous',data=df,ax=axis[1,0])
sns.boxplot(x='n1_deposit',y='n_job',data=df,ax=axis[1,1])
sns.boxplot(x='n1_deposit',y='n_marital',data=df,ax=axis[1,2])
sns.boxplot(x='n1_deposit',y='n_education',data=df,ax=axis[1,3])
sns.boxplot(x='n1_deposit',y='n_loan',data=df,ax=axis[2,0])
sns.boxplot(x='n1_deposit',y='n_contact',data=df,ax=axis[2,1])
sns.boxplot(x='n1_deposit',y='n_month',data=df,ax=axis[2,2])
sns.boxplot(x='n1_deposit',y='n_poutcome',data=df,ax=axis[2,3])


# In[822]:


#shows all record whose deposit value is 0 
outcome_zero=df[df['n1_deposit'] == 0]
outcome_zero


# In[823]:


#shows all record whose deposit value is 1
outcome_one=df[df['n1_deposit'] == 1]
outcome_one


# In[824]:


df.n1_deposit.value_counts()


# # Outlier Removal

# In[825]:


def Outdet(df):
    Q1=df.quantile(0.25)
    Q3=df.quantile(0.75)
    IQR=Q3-Q1
    LR=Q1-(IQR*1.5)
    UR=Q3+(IQR*1.5)
    return LR,UR


# In[826]:


LR,UR=Outdet(outcome_zero.age)
print(LR,UR)


# In[827]:


#removing outliers from feature(age)
outcome_zero=outcome_zero[(outcome_zero['age'] > LR)  &  (outcome_zero['age'] < UR)]
outcome_zero


# #### 5873 rows-Before outlier removal from age
# 
# #### 5818 rows-After outlier removal from age

# In[828]:


LR,UR=Outdet(outcome_zero.balance)
print(LR,UR)


# In[829]:


outcome_zero=outcome_zero[(outcome_zero['balance']>LR) & (outcome_zero['balance']< UR)]
outcome_zero


# #### 5818 rows-Before outlier removal from balance
# 
# #### 5205 rows-After outlier removal from balance

# In[830]:


LR,UR=Outdet(outcome_zero.day)
print(LR,UR)


# In[831]:


outcome_zero=outcome_zero[(outcome_zero['day']>LR) & (outcome_zero['day']< UR)]
outcome_zero


# In[832]:


LR,UR=Outdet(outcome_zero.duration)
print(LR,UR)


# In[833]:


outcome_zero=outcome_zero[(outcome_zero['duration']>LR) & (outcome_zero['duration']< UR)]
outcome_zero


# #### 5205
# 
# #### 4865

# In[834]:


LR,UR=Outdet(outcome_zero.campaign)
print(LR,UR)


# In[835]:


outcome_zero=outcome_zero[(outcome_zero['campaign']>LR) & (outcome_zero['campaign']< UR)]
outcome_zero


# #### 4865
# 
# #### 4339

# In[836]:


LR,UR=Outdet(outcome_one.age)
print(LR,UR)


# In[837]:


outcome_one=outcome_one[(outcome_one['age']>LR) & (outcome_one['age']< UR)]
outcome_one


# In[838]:


LR,UR=Outdet(outcome_one.balance)
print(LR,UR)


# In[839]:


outcome_one=outcome_one[(outcome_one['balance']>LR) & (outcome_one['balance']< UR)]
outcome_one


# In[840]:


LR,UR=Outdet(outcome_one.day)
print(LR,UR)


# In[841]:


outcome_one=outcome_one[(outcome_one['day']>LR) & (outcome_one['day']< UR)]
outcome_one


# In[842]:


LR,UR=Outdet(outcome_one.duration)
print(LR,UR)


# In[843]:


outcome_one=outcome_one[(outcome_one['duration']>LR) & (outcome_one['duration']< UR)]
outcome_one


# In[844]:


LR,UR=Outdet(outcome_one.campaign)
print(LR,UR)


# In[845]:


outcome_one=outcome_one[(outcome_one['campaign']>LR) & (outcome_one['campaign']< UR)]
outcome_one


# In[846]:


LR,UR=Outdet(outcome_one.n_poutcome)
print(LR,UR)


# In[847]:


outcome_one=outcome_one[(outcome_one['n_poutcome']>LR) & (outcome_one['n_poutcome']< UR)]
outcome_one


# In[848]:


LR,UR=Outdet(outcome_one.balance)
print(LR,UR)


# In[849]:


outcome_one=outcome_one[(outcome_one['balance']>LR) & (outcome_one['balance']< UR)]
outcome_one


# In[850]:


LR,UR=Outdet(outcome_zero.balance)
print(LR,UR)


# In[851]:


outcome_zero=outcome_zero[(outcome_zero['balance']>LR) & (outcome_zero['balance']< UR)]
outcome_zero


# In[852]:


LR,UR=Outdet(outcome_zero.duration)
print(LR,UR)


# In[853]:


outcome_zero=outcome_zero[(outcome_zero['duration']>LR) & (outcome_zero['duration']< UR)]
outcome_zero


# In[854]:


LR,UR=Outdet(outcome_one.duration)
print(LR,UR)


# In[855]:


outcome_one=outcome_one[(outcome_one['duration']>LR) & (outcome_one['duration']< UR)]
outcome_one


# In[856]:


LR,UR=Outdet(outcome_zero.balance)
print(LR,UR)
outcome_zero=outcome_zero[(outcome_zero['balance']>LR) & (outcome_zero['balance']< UR)]
outcome_zero


# In[857]:


LR,UR=Outdet(outcome_one.balance)
print(LR,UR)
outcome_one=outcome_one[(outcome_one['balance']>LR) & (outcome_one['balance']< UR)]
outcome_one


# In[858]:


LR,UR=Outdet(outcome_zero.balance)
print(LR,UR)
outcome_zero=outcome_zero[(outcome_zero['balance']>LR) & (outcome_zero['balance']< UR)]
outcome_zero


# In[859]:


LR,UR=Outdet(outcome_one.balance)
print(LR,UR)
outcome_one=outcome_one[(outcome_one['balance']>LR) & (outcome_one['balance']< UR)]
outcome_one


# In[860]:


LR,UR=Outdet(outcome_zero.balance)
print(LR,UR)
outcome_zero=outcome_zero[(outcome_zero['balance']>LR) & (outcome_zero['balance']< UR)]
outcome_zero


# In[861]:


LR,UR=Outdet(outcome_one.balance)
print(LR,UR)
outcome_one=outcome_one[(outcome_one['balance']>LR) & (outcome_one['balance']< UR)]
outcome_one


# In[862]:


df_1=pd.concat([outcome_zero,outcome_one],axis=0)
df_1


# In[863]:


df_1.n1_deposit.value_counts()


# In[864]:


figure, axis = plt.subplots(3, 4, figsize = (50,25))
sns.boxplot(x='n1_deposit',y='age',data=df_1,ax=axis[0,0])
sns.boxplot(x='n1_deposit',y='balance',data=df_1,ax=axis[0,1])
sns.boxplot(x='n1_deposit',y='duration',data=df_1,ax=axis[0,2])
sns.boxplot(x='n1_deposit',y='campaign',data=df_1,ax=axis[0,3])
sns.boxplot(x='n1_deposit',y='previous',data=df_1,ax=axis[1,0])
sns.boxplot(x='n1_deposit',y='n_job',data=df_1,ax=axis[1,1])
sns.boxplot(x='n1_deposit',y='n_marital',data=df_1,ax=axis[1,2])
sns.boxplot(x='n1_deposit',y='n_education',data=df_1,ax=axis[1,3])
sns.boxplot(x='n1_deposit',y='n_loan',data=df_1,ax=axis[2,0])
sns.boxplot(x='n1_deposit',y='n_contact',data=df_1,ax=axis[2,1])
sns.boxplot(x='n1_deposit',y='n_month',data=df_1,ax=axis[2,2])

sns.boxplot(x='n1_deposit',y='n_poutcome',data=df_1,ax=axis[2,3])


# In[865]:


sns.pairplot(df_1)


# # Correlation Matrix

# In[866]:


plt.figure(figsize=(20,20))
sns.heatmap(df.corr(),annot =True)


# #### High correlation between feature and target label 1)duration(0.56) 2)balance(0.30) 3)n_contact(-0.25) 4)campaign(-0.23) 5)n_loan(-0.12) 6)previous(0.11) 7)n_education(0.10)
# 
# 

# In[867]:


df


# In[868]:


x=df_1.drop(['n1_deposit','age','day','n_job','n_marital','n_month','n_poutcome'],axis=1).values
print(x)
y=df_1['n1_deposit'].values
print(y)


# # Splitting of Data

# In[881]:


from sklearn.model_selection import train_test_split
(x_train,x_test,y_train,y_test)=train_test_split(x,y,test_size=0.2)


# # Scale the data to improve model performance

# In[882]:


from sklearn.preprocessing import StandardScaler
std_model=StandardScaler()
x_train_std_features=std_model.fit_transform(x_train)
x_test_std_features=std_model.transform(x_test)


# In[883]:


x_train_std_features.shape


# In[884]:


x_test_std_features.shape


# # Model Building(Logistic Regression)

# In[885]:


from sklearn.linear_model import LogisticRegression
modelreg=LogisticRegression()


# In[886]:


modelreg.fit(x_train_std_features,y_train)


# In[887]:


ypred=modelreg.predict(x_test_std_features)
ypred


# In[888]:


modelreg.score(x_test_std_features,y_test)


# In[889]:


x_test_std_features.shape


# In[890]:


y_test.shape


# In[891]:


from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, ypred))
print("Precision:",metrics.precision_score(y_test, ypred))
print("Recall:",metrics.recall_score(y_test, ypred))


# In[892]:


cnf_matrix = metrics.confusion_matrix(y_test, ypred)
cnf_matrix


# # Model Building(KNeighborsClassifier)

# In[893]:


from sklearn.neighbors import KNeighborsClassifier
KNN_model = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
KNN_model.fit(x_train_std_features, y_train)


# In[894]:


y_predicted_KNN = KNN_model.predict(x_test_std_features)


# In[895]:


KNN_model.score(x_test_std_features,y_test)


# In[896]:


print("Accuracy:",metrics.accuracy_score(y_test, y_predicted_KNN))
print("Precision:",metrics.precision_score(y_test,y_predicted_KNN))
print("Recall:",metrics.recall_score(y_test, y_predicted_KNN))


# # Model Building(Naive Bayes)

# In[897]:


from sklearn.naive_bayes import GaussianNB
naive_bayes_model= GaussianNB()
naive_bayes_model.fit(x_train_std_features, y_train)


# In[898]:


y_predicted_naive = naive_bayes_model.predict(x_test_std_features)


# In[899]:


naive_bayes_model.score(x_test_std_features,y_test)


# In[900]:


print("Accuracy:",metrics.accuracy_score(y_test, y_predicted_naive))
print("Precision:",metrics.precision_score(y_test,y_predicted_naive))
print("Recall:",metrics.recall_score(y_test, y_predicted_naive))


# # Model Building(Decision Tree Classifier)

# In[901]:


from sklearn.tree import DecisionTreeClassifier
deseciontree_model=DecisionTreeClassifier()
deseciontree_model.fit(x_train_std_features, y_train)


# In[902]:


y_predicted_deseciontree = deseciontree_model.predict(x_test_std_features)
deseciontree_model.score(x_test_std_features,y_test)


# In[903]:


print("Accuracy:",metrics.accuracy_score(y_test,y_predicted_deseciontree))
print("Precision:",metrics.precision_score(y_test,y_predicted_deseciontree))
print("Recall:",metrics.recall_score(y_test, y_predicted_deseciontree))


# # Model Building(Random Forest Classifier)

# In[904]:


from sklearn.ensemble import RandomForestClassifier
randomforest_model= RandomForestClassifier(n_estimators = 10, criterion = 'entropy')
randomforest_model.fit(x_train_std_features, y_train)


# In[905]:


y_predicted_randomforest = randomforest_model.predict(x_test_std_features)
randomforest_model.score(x_test_std_features,y_test)


# In[906]:


print("Accuracy:",metrics.accuracy_score(y_test,y_predicted_randomforest))
print("Precision:",metrics.precision_score(y_test,y_predicted_randomforest))
print("Recall:",metrics.recall_score(y_test, y_predicted_randomforest))


# # Model Building(SVM using RBF kernel)

# In[907]:


from sklearn.svm import SVC
SVM_model_rbf=SVC(kernel='rbf')
SVM_model_rbf.fit(x_train_std_features,y_train)


# In[908]:


y_predicted_SVM_rbf = SVM_model_rbf.predict(x_test_std_features)
SVM_model_rbf.score(x_test_std_features,y_test)


# In[909]:


print("Accuracy:",metrics.accuracy_score(y_test,y_predicted_SVM_rbf))
print("Precision:",metrics.precision_score(y_test,y_predicted_SVM_rbf))
print("Recall:",metrics.recall_score(y_test, y_predicted_SVM_rbf))


# # Model Building(SVM using linear kernel)
# 

# In[910]:


from sklearn.svm import SVC
SVM_model_linear=SVC(kernel='linear')
SVM_model_linear.fit(x_train_std_features,y_train)


# In[911]:


y_predicted_SVM_linear = SVM_model_linear.predict(x_test_std_features)


# In[912]:


SVM_model_linear.score(x_test_std_features,y_test)


# In[913]:


print("Accuracy:",metrics.accuracy_score(y_test,y_predicted_SVM_linear))
print("Precision:",metrics.precision_score(y_test,y_predicted_SVM_linear))
print("Recall:",metrics.recall_score(y_test, y_predicted_SVM_linear))


# # Model Building(Gradient Boosting Classifier)

# In[914]:


from sklearn.ensemble import GradientBoostingClassifier
GradientB_model=GradientBoostingClassifier(n_estimators=100,learning_rate=1.0)


# In[915]:


GradientB_model.fit(x_train_std_features,y_train)
y_predicted_GradientB = GradientB_model.predict(x_test_std_features)


# In[916]:


GradientB_model.score(x_test_std_features,y_test)


# In[917]:


print("Accuracy:",metrics.accuracy_score(y_test,y_predicted_GradientB))
print("Precision:",metrics.precision_score(y_test,y_predicted_GradientB))
print("Recall:",metrics.recall_score(y_test, y_predicted_GradientB))


# # Model Building(AdaBoost Classifier)

# In[918]:


from sklearn.ensemble import AdaBoostClassifier


# In[919]:


Adaboost_model=AdaBoostClassifier(n_estimators=100)


# In[920]:


Adaboost_model.fit(x_train_std_features,y_train)
y_predicted_Adaboost = Adaboost_model.predict(x_test_std_features)


# In[921]:


Adaboost_model.score(x_test_std_features,y_test)


# In[922]:


print("Accuracy:",metrics.accuracy_score(y_test,y_predicted_Adaboost))
print("Precision:",metrics.precision_score(y_test,y_predicted_Adaboost))
print("Recall:",metrics.recall_score(y_test, y_predicted_Adaboost))


# # Final Report

# In[933]:


dfnew = pd.DataFrame()
dfnew['Names'] = ['LogisticRegression','KNeighborsClassifier','GaussianNB','DecisionTreeClassifier','RandomForestClassifier','SVM_RBF','SVM_Linear','GradientBoostingClassifier','AdaBoostClassifier']
dfnew['Score'] = [0.8316176470588236,0.8301470588235295,0.8080882352941177,0.8102941176470588,0.8323529411764706,0.8441176470588235, 0.8345588235294118,0.8389705882352941,0.8566176470588235]
dfnew['precision']=[0.8313458262350937,0.8274111675126904,0.8299445471349353,0.7866242038216561,0.8494623655913979,0.8664259927797834,0.8544520547945206, 0.828665568369028,0.8578680203045685]
dfnew['recall']=[ 0.7896440129449838,0.7912621359223301,0.7265372168284789,0.7993527508090615,0.7669902912621359,0.7765451664025357,0.7880258899676376,0.813915857605178,0.8203883495145631]


# In[934]:


dfnew


# In[935]:



cm = sns.light_palette('navy',as_cmap=True)
s = dfnew.style.background_gradient(cmap=cm)
s


# In[936]:


plt.figure(figsize=(20,5))
sns.set(style="whitegrid")
ax = sns.barplot(y ='Score',x = 'Names',data = dfnew)


# In[937]:


from sklearn.metrics import classification_report,roc_auc_score,roc_curve,auc
report_Adaboost = classification_report(y_test,y_predicted_Adaboost)
print(report_Adaboost)


# In[938]:


roc_auc_score(y_test,y_predicted_Adaboost)


# In[939]:


fpr,tpr,threshold =roc_curve(y_test,y_predicted_Adaboost)
auc = auc(fpr,tpr)


# In[940]:


plt.figure(figsize=(5,5),dpi=100)
plt.plot(fpr,tpr,linestyle='-',label = "(auc = %0.3f)" % auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()


# ### Conclusion:

# The dataset contained 16 features and 1 target variable for binary classification which determines if client will subscribe deposit or not.I have done feature extraction and got 7 important features, then applied various classification algorithms on the data which made it clear that Adaboost Classifier Model performed excellent with high accuracy(85%) compared to other algorithms.

# In[ ]:




