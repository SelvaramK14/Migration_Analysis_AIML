# In[6]:


import pandas as pd
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.naive_bayes import GaussianNB


# In[7]:


data = pd.read_csv("F:\My_Projects\AIML\migration_nz.csv")
data.head(10)


# In[8]:


data['Measure'].unique()


# In[9]:


data['Measure'].replace("Arrivals",0,inplace=True)
data['Measure'].replace("Departures",1,inplace=True)
data['Measure'].replace("Net",2,inplace=True)


# In[10]:


data['Measure'].unique()


# In[8]:


data['Country'].unique()


# In[11]:


data['CountryID'] = pd.factorize(data.Country)[0]
data['CitID'] = pd.factorize(data.Citizenship)[0]


# In[12]:


data['CountryID'].unique()


# In[13]:


data.isnull().sum()


# In[14]:


data["Value"].fillna(data["Value"].median(),inplace=True)


# In[15]:


data.isnull().sum()


# In[16]:


from sklearn.model_selection import train_test_split
X= data[['CountryID','Measure','Year','CitID']].values
Y= data['Value'].values
X_train, X_test, y_train, y_test = train_test_split(
  X, Y, test_size=0.3, random_state=9)


# In[17]:


print(data.columns)


# In[18]:


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=70,max_features = 3,max_depth=5,n_jobs=-1)
rf.fit(X_train ,y_train)
rf.score(X_test, y_test)


# In[23]:


X = data[['CountryID','Measure','Year','CitID']]
Y = data['Value']
X_train, X_test, y_train, y_test = train_test_split(
  X, Y, test_size=0.3, random_state=9)
grouped = data.groupby(['Year']).aggregate({'Value' : 'sum'})

grouped.plot(kind='line')
plt.axhline(0, color='g')
plt.show()


# In[36]:


grouped.plot(kind='bar')
plt.axhline(0, color='g')
plt.show()


# In[47]:


data['Country'].unique()


# In[53]:


# Prompt the user to input the desired year
user_year = int(input("Enter the year to find migration: "))
user_country=input("Enter the country to find the migration in : ")
user_in=int(input("Enter 1 for Arrivals to the country and 2 for departure from the country: "))
user_input=''
if user_in==1:
    user_input='Arrivals'
elif user_in==2:
    user_input='Departures'

migration_year_nz = data[(data['Year'] == user_year) & (data['Country'] == user_country) & (data['Measure']==user_input)]


total_migration_nz = migration_year_nz['Value'].sum()

print(f"Total migration to {user_country} in {user_year}: {total_migration_nz}")


# In[4]:


user_country = input("Enter the country name: ")
country_id = data[data['Country'] == user_country]['CountryID'].iloc[0]
predicted_values = []
years = range(data['Year'].min(), data['Year'].max() + 1)
for year in years:
    year_data = pd.DataFrame({'CountryID': [country_id] * len(data['Measure'].unique()),
                              'Measure': range(len(data['Measure'].unique())),
                              'Year': [year] * len(data['Measure'].unique()),
                              'CitID': range(len(data['CitID'].unique()))})
    predicted_values.append(rf.predict(year_data).sum())  # Aggregate predictions for each year

# Plot the predicted migration trends for the user-input country
plt.figure(figsize=(10, 6))
plt.plot(years, predicted_values, marker='o', linestyle='-')
plt.title(f"Predicted Migration Trends for {user_country}")
plt.xlabel("Year")
plt.ylabel("Migration Value")
plt.grid(True)
plt.show()
