#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv(r"C:\Users\jsabh\OneDrive\Documents\ML programs\Wallmart dataset\train.csv")
features = pd.read_csv(r"C:\Users\jsabh\OneDrive\Documents\ML programs\Wallmart dataset\features.csv")
print("Train Data:")
print(train.head())
print("Features Data:")
print(features.head())


# In[2]:


test = pd.read_csv(r"C:\Users\jsabh\OneDrive\Documents\ML programs\Wallmart dataset\test.csv")
stores = pd.read_csv(r"C:\Users\jsabh\OneDrive\Documents\ML programs\Wallmart dataset\stores.csv")


# In[3]:


print("Store Data:")
print(stores.head())


# In[4]:


merge_df = pd.merge(train,features,on=["Store","Date"],how='inner')
merge_df.head()


# In[5]:


merge_df.describe().transpose()


# In[6]:


merge_df[(merge_df.Store==1)].plot(kind="scatter",x="Date",y="Weekly_Sales",alpha = 0.1)


# In[7]:


corr_matrix = merge_df.corr()
corr_matrix["Weekly_Sales"].sort_values(ascending=False)


# In[8]:


merge_df[(merge_df.Store==1)].plot(kind="bar",x="Dept",y="Weekly_Sales")


# In[9]:


attributes = ["IsHoliday_x","Fuel_Price","Temperature","CPI","Unemployment"]
merge_df = merge_df.drop(attributes,axis=1)
merge_df.head()


# In[10]:


X = merge_df.drop("Weekly_Sales",axis = 1)
X_label = merge_df["Weekly_Sales"].copy()


# In[11]:


X.head()


# In[12]:


#transforming the test data set and the features data set
merge_test = pd.merge(test,features,on=['Store','Date'],how='inner')


# In[13]:


attributes = ["IsHoliday_x","Fuel_Price","Temperature","CPI","Unemployment"]
merge_test = merge_test.drop(attributes,axis = 1)


# In[14]:


X_test = merge_test


# In[15]:


#filling the missing values and removing the non numeric attribute
X[X==np.inf]=np.nan
X.fillna(X.mean(),inplace = True)


# In[16]:


X.head()


# In[17]:


from sklearn.preprocessing import OrdinalEncoder
ord_encode = OrdinalEncoder()
X_cat = X[["IsHoliday_y"]]
X_cat_encoded = ord_encode.fit_transform(X_cat)
X_cat_encoded


# In[18]:


X_num = X.drop("IsHoliday_y",axis = 1)
X_num.head()


# In[20]:


X["IsHoliday"] = X_cat_encoded


# In[21]:


X = X.drop("IsHoliday_y",axis = 1)
X.head()


# In[22]:


from datetime import datetime as dt
X["DatetimeObj"] = [dt.strptime(x,'%Y-%m-%d') for x in list(X['Date'])]


# In[23]:


X = X.drop('Date',axis = 1)
X.head()


# In[28]:


X["Date"] = pd.to_datetime(X["DatetimeObj"])
X["Date"] = X["Date"].map(dt.toordinal)
X.head()


# In[29]:


X = X.drop("DatetimeObj",axis = 1)
X.head()


# In[38]:


from sklearn.linear_model import LinearRegression
lin_model = LinearRegression(normalize = True)
lin_model.fit(X,X_label)


# In[39]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(lin_model,X,X_label,scoring = "neg_mean_squared_error",cv = 10)
lin_reg_score = np.sqrt(-scores)
lin_reg_score.mean()


# In[33]:


X_test[X_test==np.inf]=np.nan
X_test.fillna(X_test.mean(),inplace = True)


# In[34]:


from datetime import datetime as dt
X_test["DatetimeObj"] = [dt.strptime(x,'%Y-%m-%d') for x in list(X_test['Date'])]


# In[35]:


X_test = X_test.drop('Date',axis = 1)


# In[36]:


X_test["Date"] = pd.to_datetime(X_test["DatetimeObj"])
X_test["Date"] = X_test["Date"].map(dt.toordinal)


# In[37]:


X_test = X_test.drop("DatetimeObj",axis = 1)


# In[40]:


from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(X,X_label)


# In[41]:


forest_scores = cross_val_score(forest_reg,X,X_label,scoring = "neg_mean_squared_error",cv = 10)
forest_rmse = np.sqrt(-forest_scores)
print("Scores:",forest_rmse)
print("Mean:",forest_rmse.mean())


# In[43]:


from sklearn.model_selection import GridSearchCV

param_grid = [
    {'n_estimators':[3,10,30], 'max_features':[2,4,6,8]},
    {'bootstrap':[False], 'n_estimators':[3,10], 'max_features':[2,3,4]}
]
grid_search = GridSearchCV(forest_reg,param_grid,cv=5,scoring = "neg_mean_squared_error",return_train_score = True)
grid_search.fit(X,X_label)


# In[44]:


grid_search.best_params_


# In[46]:


final_model = grid_search.best_estimator_
final_predictions = final_model.predict(X_test)


# In[48]:


final_predictions

