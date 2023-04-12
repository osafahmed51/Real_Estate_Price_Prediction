#!/usr/bin/env python
# coding: utf-8

# In[24]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
matplotlib.rcParams['figure.figsize']=(20,10)


# In[25]:


df1=pd.read_csv('Bengaluru_House_Data.csv')
df1.head()


# In[26]:


df1.shape


# In[27]:


df1.groupby('area_type')['area_type'].agg('count')


# In[28]:


df2=df1.drop(['area_type','society','balcony','availability'],axis='columns')
df2.head()


# In[29]:


df2.isnull().sum()


# In[30]:


df3=df2.dropna()
df3.isnull().sum()


# In[33]:


df3.shape


# In[34]:


df3['bhk']=df3['size'].apply(lambda x: int(x.split(' ')[0]))


# In[35]:


df3.head()


# In[36]:


df3['bhk'].unique()


# In[37]:


df3[df3.bhk>20]


# In[38]:


df3['total_sqft'].unique()


# In[39]:


def is_float(x):
    try:
        float(x)
    except:
        return False
    return True


# In[40]:


df3[~df3['total_sqft'].apply(is_float)].head()


# In[41]:


def convert_sqft_to_num(x):
    tokens=x.split('-')
    if len(tokens) ==2:
        return (float(tokens[0]) + float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None


# In[42]:


convert_sqft_to_num('2100-2850')


# In[43]:


df4=df3.copy()


# In[44]:


df4['total_sqft']=df4['total_sqft'].apply(convert_sqft_to_num)
df4.head(31)


# In[21]:


df4.loc[30]


# In[40]:


df5=df4.copy()
df5['price_per_sqft']=df5['price']*100000/df5['total_sqft']
df5.head()


# In[41]:


len(df5.location.unique())


# In[42]:


df5.location=df5.location.apply(lambda x: x.strip())
location_stats=df5.groupby('location')['location'].agg('count').sort_values(ascending=False)
location_stats


# In[43]:


len(location_stats[location_stats<=10])


# In[44]:


location_stats_less_than_10=location_stats[location_stats<=10]
location_stats_less_than_10


# In[45]:


len(df5.location.unique())


# In[46]:


df5.location=df5.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
len(df5.location.unique())


# In[47]:


df5.head(10)


# In[48]:


#Removing outliers
df5[df5.total_sqft/df5.bhk<300].head()


# In[49]:


df5.shape


# In[50]:


df6=df5[~(df5.total_sqft/df5.bhk<300)]
df6.shape


# In[51]:


df6.price_per_sqft.describe()


# In[52]:


def remove_pps_outliers(df):
    df_out=pd.DataFrame()
    for key,subdf in df.groupby('location'):
        m=np.mean(subdf.price_per_sqft)
        st=np.std(subdf.price_per_sqft)
        reduced_df=subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
        df_out=pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out
df7=remove_pps_outliers(df6)
df7.shape


# In[56]:


def plot_scatter_chart(df,location):
    bhk2=df[(df.location==location) & (df.bhk==2)]
    bhk3=df[(df.location==location) & (df.bhk==3)]
    matplotlib.rcParams['figure.figsize']=(15,10)
    plt.scatter(bhk2.total_sqft,bhk2.price,color='blue',label='2 BHK',s=50)
    plt.scatter(bhk3.total_sqft,bhk3.price,marker='+',color='green',label='3 BHK',s=50)
    plt.xlabel('Total square feet area')
    plt.ylabel('Price ')
    plt.title(location)
    plt.legend()
    
plot_scatter_chart(df7,'Hebbal')


# In[58]:


def remove_bhk_outliers(df):
    exclude_indices=np.array([])
    for location , location_df in df.groupby('location'):
        bhk_stats={}
        for bhk , bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk]={
                'mean':np.mean(bhk_df.price_per_sqft),
                'std':np.std(bhk_df.price_per_sqft),
                'count':bhk_df.shape[0]
            }
        for bhk , bhk_df in location_df.groupby('bhk'):
            stats=bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices=np.append(exclude_indices,bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
df8=remove_bhk_outliers(df7)
df8.shape


# In[59]:


plot_scatter_chart(df8,'Hebbal')


# In[60]:


df8.bath.unique()


# In[61]:


df8[df8.bath>10]


# In[62]:


plt.hist(df8.bath,rwidth=0.8)
plt.xlabel('Number of bathrooms')
plt.ylabel('count')


# In[63]:


df8[df8.bath>df8.bhk+2]


# In[64]:


df9=df8[df8.bath<df8.bhk+2]
df9.shape


# In[65]:


df10=df9.drop(['size','price_per_sqft'],axis='columns')
df10.head(20)


# In[67]:


dummies=pd.get_dummies(df10.location)
dummies.head(3)


# In[72]:


df11=pd.concat([df10,dummies.drop('other',axis='columns')],axis='columns')
df11.head(3)


# In[74]:


df12=df11.drop('location',axis='columns')
df12.head(3)


# In[80]:


X=df12.drop('price',axis='columns')
X.head()


# In[81]:


y=df12.price
y.head()


# In[82]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=10)


# In[83]:


from sklearn.linear_model import LinearRegression
lr_clf=LinearRegression()
lr_clf.fit(X_train,y_train)
lr_clf.score(X_test,y_test)


# In[84]:


X.columns


# In[85]:


def predict_price(location,sqft,bath,bhk):
    loc_index=np.where(X.columns==location)[0][0]
    
    x=np.zeros(len(X.columns))
    x[0]=sqft
    x[1]=bath
    x[2]=bhk
    if loc_index >=0:
        x[loc_index]=1
        
    return lr_clf.predict([x])[0]


# In[88]:


predict_price('1st Phase JP Nagar',1000,2,2)


# In[ ]:




