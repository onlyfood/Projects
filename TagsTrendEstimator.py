
# coding: utf-8

# In[1]:

#importation
import nltk
import octoparse
import sys
import os
import warnings
import pickle
import xgboost 
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import MinMaxScaler 
from sklearn.preprocessing import OneHotEncoder 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import AdaBoostRegressor


# In[2]:

#load the files

top_file=pd.read_csv(r'C:\Users\Steven\Desktop\Top100BeautyFashionI.csv')
common_words=pd.read_csv(r'C:\Users\Steven\Desktop\common_words.csv',engine='python')
df=pd.read_csv(r'C:/Users/Steven/Desktop/TrendyArgus4.csv')


# In[3]:

#create the datamining list for instagram

ins_list=pd.DataFrame(top_file)
ins_list['ins'] = ins_list['ins'].map(lambda x: str(x)[1:])
ins_list.drop_duplicates(subset=('ins'),inplace=True)
ins_list.reset_index(inplace=True)
ins_list.drop(columns='index',inplace=True)


# In[4]:

len(ins_list)


# In[ ]:




# In[5]:

i=ins_list['ins']
name=('https://www.instagram.com/'+i)
name=pd.DataFrame(name)
name.to_csv(r'C:\Users\Steven\Desktop\BeautyFahsionName.csv',index=False,header=False)


# In[6]:

df


# In[ ]:




# In[7]:

#data processing
   
    #drop duplicates
df.drop_duplicates(inplace=True)


# In[ ]:




# In[ ]:




# In[8]:

#fill nan
df.dropna(axis=0,inplace=True)




# In[9]:

#take out ago, add 2019,then to date
df=df[~df['date'].str.contains('ago')]

ii=df[df['date'].str.contains('2018')]
iii=df[df['date'].str.contains('2017')]
i=df[~df['date'].str.contains('|'.join(['2018','2017']))]
i.date=i.date+'/2019'

df=pd.concat([i,ii,iii],axis=0)

date=pd.to_datetime(df['date'],errors='ignore')
date=pd.Series(date,name='date')
df.drop(columns=['date'],axis=1,inplace=True)
df=pd.concat([df,date],axis=1)


# In[ ]:




# In[10]:

df_new=df.sample(frac=1).reset_index(drop=True)


# In[11]:

#word count
words=df_new['text'].str.split(expand=True).stack().value_counts()

words



# In[ ]:





# In[ ]:




# In[12]:



common_words


# In[13]:

common_words=common_words['words']


# In[ ]:




# In[14]:

words=pd.DataFrame(words)
words.reset_index(inplace=True)
words.rename(columns={0:'count','index':'word'},inplace=True)
words


# In[15]:

a=list(words.word[0:3000])
a=set(a)

b=set(common_words)

c=a-b

c


# In[16]:


e=words[words['word'].isin(c)]
e1=e[~e['word'].str.contains('#')]
e2=e[~e['word'].str.contains('@')]#?????????????????????????????????????!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
e=pd.concat([e1,e2],axis=0)
e.drop_duplicates(inplace=True)
e


# In[ ]:




# In[ ]:




# In[17]:

tag=e[e['word'].str.contains('#')]
tag.to_csv(r'C:/Users/Steven/Desktop/tagsFitness.csv')
words=e[~e['word'].str.contains('#')]
words.to_csv(r'C:/Users/Steven/Desktop/wordsFitness.csv')


# In[18]:

df_tag=pd.read_csv(r'C:/Users/Steven/Desktop/Programing/Data/15000.csv')


# In[19]:

df_tag.dropna(inplace=True)
df_tag.drop_duplicates(inplace=True)
df_tag.head(5)


# In[ ]:




# In[20]:

i=df_tag[df_tag['text'].str.contains('fitnessmotivation')]
i['text']=1

ii=df_tag[~df_tag['text'].str.contains('fitnessmotivation')]
ii['text']=0

df_fm=pd.concat([i,ii],axis=0)

df_fm.drop_duplicates(inplace=True)





# In[ ]:




# In[21]:

df_fm2=df_fm
df_fm=df_fm[~df_fm['date'].str.contains('ago')]

ii=df_fm[df_fm['date'].str.contains('2018')]

iii=df_fm[df_fm['date'].str.contains('2017')]

i=df_fm[~df_fm['date'].str.contains('|'.join(['2018','2017']))]

i.date=i.date+'/2019'

df_fm=pd.concat([i,ii,iii],axis=0)

date=pd.to_datetime(df_fm['date'],errors='ignore')

date=pd.Series(date,name='date')

df_fm.drop(columns=['date'],axis=1,inplace=True)

df_fm=pd.concat([df_fm,date],axis=1)


# In[22]:


df_fm.sort_values(by=['date'],inplace=True)

df_fm.reset_index(drop=True,inplace=True)

df_fm


# In[23]:

#a=max(df_fm['date'])

#i=df_fm2['date'][df_fm2['date'].str.contains('ago')]

#ii=i[i.str.contains('hours')]

#ii=pd.DataFrame(ii)

#b=a+pd.DateOffset(7)
#b=str(b)

#ii.replace(to_replace=ii['date'],value=b,inplace=True)
#ii


# In[24]:


df_fm['text']=df_fm['text'].astype('category')

df_fm.head(3)


# In[25]:

df_fm.ins=df_fm.ins.astype('category')

from sklearn.preprocessing import LabelBinarizer
encoder=LabelBinarizer()
ins_1hot=encoder.fit_transform(df_fm['ins'])



# In[26]:

ins_cat=list(df_fm['ins'].unique())


# In[27]:

ins_1hot=pd.DataFrame(ins_1hot,columns=[ins_cat])
df_fm1=pd.concat([df_fm,ins_1hot],axis=1)
df_fm1.drop(columns='ins',inplace=True)


# In[28]:


df_fm1



# In[29]:


i=pd.to_timedelta(df_fm1['date'])

ii=min(i)

i=i-ii

iii=i.dt.days

iv=iii-7

iv=pd.Series(iv,name='timedelta')

df_fm1.drop(columns=['date'],inplace=True)

df_fm1=pd.concat([df_fm1,iv],axis=1)







v=df_fm1['text'].cumsum(axis=0)

v=pd.Series(v,name='cumsum')

df_fm1=pd.concat([df_fm1,v],axis=1)



df_fm1


# In[30]:

df_fm1=df_fm1.sample(frac=1)

df_fm1.drop(columns='text',inplace=True)


# In[31]:

len(df_fm1)


# In[32]:

df_fm2=df_fm1.dropna()


# In[33]:

len(df_fm2)


# In[34]:

len(df_fm)


# In[35]:

df_fm3=df_fm.dropna()
len(df_fm3)


# In[36]:

from sklearn.model_selection import train_test_split
X=df_fm2
y=df_fm2['cumsum']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=42)
X_train


# In[ ]:




# In[ ]:




# In[ ]:




# In[40]:

from sklearn.ensemble import AdaBoostRegressor
Ada_reg=AdaBoostRegressor(random_state=42)
Ada_reg.fit(X_train,y_train)
Ada_reg.score(X_train,y_train)


# In[ ]:




# In[41]:

Ada_reg.get_params


# In[42]:

param_grid_xg=[
        {'n_estimators':[45,50,55],'learning_rate':[0.75,1,1.25]},
       
        
]
grid_search=GridSearchCV(Ada_reg,param_grid_xg,cv=5,scoring='neg_mean_squared_error',refit=True)

grid_search.fit(X_train,y_train)

#grid_search.transform(X_train,y_train)

grid_search.best_params_


# In[43]:

Ada_reg.get_params


# In[ ]:




# In[44]:

Ada_reg.score(X_test,y_test)


# In[ ]:




# In[46]:

result=Ada_reg.predict(X_test)


# In[63]:

def predict (data):
    data=Ada_reg.predict(X_test)
    #data=pd.to_csv(r'locations/result')
    print('Result Saved')
    print('The result should be between 1 to 100')
    print('check the array')
    return data[:3]


# In[64]:

predict(X_test)


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



