#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline

from sklearn import metrics  
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
get_ipython().system('pip install xgboost')
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier


# In[2]:


data = pd.read_csv(r"C:\Users\drash\Documents\Drashti\Quantitative Research\LSE CFM\Credit Risk Neural Networks\Customer Loan Data.csv")


# In[3]:


pd.set_option("display.max_columns", None)


# In[4]:


data.shape


# In[5]:


data.head()


# In[6]:


data.tail()


# In[7]:


data.describe()


# In[8]:


data.isnull().sum()


# In[9]:


data.duplicated().sum


# In[10]:


plt.figure(figsize=(10,10))
plt.style.use('fivethirtyeight')
sns.countplot(x='default', data=data)
plt.title('Number of Default Customers\n (Default = 1, Not Default = 0)')


# In[11]:


#violin plot analysis


# In[12]:


sns.violinplot(x='default', y='income', data=data, hue='default', split=True)
plt.xlabel('Defaults (0: No, 1: Yes)')
plt.ylabel('Income')
plt.title('Distribution of Income by Defaults')
plt.show()


# In[13]:


sns.violinplot(x='default', y='loan_amt_outstanding', data=data, hue='default', split=True)
plt.xlabel('Defaults (0: No, 1: Yes)')
plt.ylabel('Income')
plt.title('Distribution of Outstanding Loan Amount by Defaults')
plt.show()


# In[14]:


sns.violinplot(x='default', y='total_debt_outstanding', data=data, hue='default', split=True)
plt.xlabel('Defaults (0: No, 1: Yes)')
plt.ylabel('Income')
plt.title('Distribution of Outstanding Total Debt Amount by Defaults')
plt.show()


# In[15]:


# Finding the correlation between different attribute
plt.figure(figsize=(22,12))
sns.heatmap(data.corr(),annot=True,cmap="coolwarm")


# In[16]:


#quartile analysis


# In[17]:


data['income_quartile']=pd.qcut(data['income'], q=4, labels=['IQ1', 'IQ2', 'IQ3', 'IQ4'])      


# In[18]:


defaults_by_quartile=data.groupby('income_quartile')['default'].sum().reset_index()


# In[19]:


defaults_by_quartile


# In[20]:


defaults_by_CLO=data.groupby('credit_lines_outstanding')['default'].sum().reset_index()


# In[21]:


defaults_by_CLO


# In[22]:


data['LAO_quartile'] = pd.qcut(data['loan_amt_outstanding'], q=4, labels=['LQ1', 'LQ2', 'LQ3', 'LQ4'])


# In[23]:


defaults_by_LAO_quartile=data.groupby('LAO_quartile')['default'].sum().reset_index()


# In[24]:


defaults_by_LAO_quartile


# In[25]:


data['TDO_quartile']=pd.qcut(data['total_debt_outstanding'], q=4, labels=['TQ1', 'TQ2', 'TQ3', 'TQ4'])      


# In[26]:


defaults_by_TDO_quartile=data.groupby('TDO_quartile')['default'].sum().reset_index()


# In[27]:


defaults_by_TDO_quartile


# In[28]:


data['F_quartile']=pd.qcut(data['fico_score'], q=4, labels=['FQ1', 'FQ2', 'FQ3', 'FQ4'])      


# In[29]:


defaults_by_FICO_quartile=data.groupby('F_quartile')['default'].sum().reset_index()


# In[30]:


defaults_by_FICO_quartile


# In[31]:


defaults_by_Years_Worked=data.groupby('years_employed')['default'].sum().reset_index()


# In[32]:


defaults_by_Years_Worked


# In[33]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[34]:


features = ['credit_lines_outstanding', 'debt_to_income', 'payment_to_income', 'years_employed', 'fico_score']


# In[35]:


data['payment_to_income'] = data['loan_amt_outstanding'] / data['income']


# In[36]:


data['debt_to_income'] = data['total_debt_outstanding'] / data['income']


# In[37]:


data


# In[38]:


x = data.drop(['customer_id', 'default', 'income_quartile', 'LAO_quartile', 'TDO_quartile', 'F_quartile' ], axis=1)


# In[39]:


y=data['default']


# In[40]:


# Standardize the data
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)


# In[41]:


import numpy as np
from sklearn.model_selection import StratifiedKFold

# Specify the number of folds (k)
num_folds = 10  # You can adjust this based on your preferences

# Initialize StratifiedKFold
kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

# Initialize lists to store evaluation results for each fold
accuracy_list = []
conf_matrix_list = []
class_report_list = []

# Perform k-fold cross-validation
for train_index, test_index in kf.split(x_scaled, y):
    x_train, x_test = x_scaled[train_index], x_scaled[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

  


# In[42]:


from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=x_train.shape[1]))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

   


# In[43]:


model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.15)


# In[44]:


y_pred_probs = model.predict(x_test)
y_pred = (y_pred_probs > 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

    
accuracy_list.append(accuracy)
conf_matrix_list.append(conf_matrix)
class_report_list.append(class_report)

from sklearn.metrics import f1_score
f1 = f1_score(y_test, y_pred)

print(f'F1 Score: {f1:.4f}')

# Print average results over all folds
print(f'Average Accuracy: {np.mean(accuracy_list):.4f}')
print(f'Average Confusion Matrix:\n{np.mean(conf_matrix_list, axis=0)}')
print(f'Average Classification Report:\n{np.mean(class_report_list, axis=0)}')


# In[46]:


from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras.callbacks import EarlyStopping

# Create a Sequential model
model = Sequential()

# Add the input layer and first hidden layer with batch normalization
model.add(Dense(units=64, activation='relu', input_dim=x_train.shape[1]))
model.add(BatchNormalization())

# Add additional hidden layers with batch normalization
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=16, activation='relu'))
model.add(BatchNormalization())

# Add the output layer (e.g., for binary classification with sigmoid activation)
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model with early stopping
history = model.fit(x_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# Evaluate the model on the testing set
accuracy = model.evaluate(x_test, y_test)
print(f'Test Accuracy: {accuracy[1]:.4f}')


# In[47]:


y_pred_probs = model.predict(x_test)
y_pred = (y_pred_probs > 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
    
accuracy_list.append(accuracy)
conf_matrix_list.append(conf_matrix)
class_report_list.append(class_report)



# Print average results over all folds
print(f'Average Accuracy: {np.mean(accuracy_list):.4f}')
print(f'Average Confusion Matrix:\n{np.mean(conf_matrix_list, axis=0)}')
print(f'Average Classification Report:\n{np.mean(class_report_list, axis=0)}')


# In[48]:


from sklearn.metrics import f1_score
f1 = f1_score(y_test, y_pred)
print(f1)


# In[49]:


from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras.callbacks import EarlyStopping
from keras.regularizers import l2

# Create a Sequential model
model = Sequential()

# Add the input layer and first hidden layer with batch normalization and L2 regularization
model.add(Dense(units=64, activation='relu', input_dim=x_train.shape[1], kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())

# Add additional hidden layers with batch normalization and L2 regularization
model.add(Dense(units=32, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dense(units=16, activation='relu', kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())

# Add the output layer (e.g., for binary classification with sigmoid activation)
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model with early stopping
history = model.fit(x_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# Evaluate the model on the testing set
accuracy = model.evaluate(x_test, y_test)
print(f'Test Accuracy: {accuracy[1]:.4f}')


# In[50]:


y_pred_probs = model.predict(x_test)
y_pred = (y_pred_probs > 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
    
accuracy_list.append(accuracy)
conf_matrix_list.append(conf_matrix)
class_report_list.append(class_report)



# Print average results over all folds
print(f'Average Accuracy: {np.mean(accuracy_list):.4f}')
print(f'Average Confusion Matrix:\n{np.mean(conf_matrix_list, axis=0)}')
print(f'Average Classification Report:\n{np.mean(class_report_list, axis=0)}')


# In[51]:


from sklearn.metrics import f1_score
f1 = f1_score(y_test, y_pred)
print(f1)


# In[ ]:




