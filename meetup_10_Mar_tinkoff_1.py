
# coding: utf-8

# In[1]:

get_ipython().magic('matplotlib inline')
import sys
print(sys.version_info)


# In[2]:

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


#  

# # Конкурс Тинькофф, задача 1
# ## 1. Постановка задачи
# ### Датасет содержит данные о кредитах на покупку электроники, которые были одобрены Tinkoff.ru. Необходимо предсказать, выберет ли покупатель кредит от Tinkoff.ru.
# 

# In[3]:

train=pd.read_csv('../data/Task1/credit_train.csv', sep=';', encoding='UTF-8', decimal=',')
test=pd.read_csv('../data/Task1/credit_test.csv', sep=';', encoding='UTF-8', decimal=',')


# In[4]:

print("train.shape: ", train.shape)
print("test.shape: ", test.shape)


# In[5]:

train.info()


#  

# ## 2. Первый подход - решение в лоб
# ### Берем небольшую выборку, только числовые фичи и тренируем SVM Classifier.
# ### результат на сабмите = 0.4991, т.е. чуть хуже чем рандом :)

#  

# ## 3. Что такое ROC AUC
# ### мерой качества решения определена ROC AUC, в примере данных для сабмита целевая переменная имеет значение "0" или "1".

# In[6]:

from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler


# In[7]:

train['tariff_id'] = train['tariff_id'].astype(np.float)


# In[8]:

numeric_fields = ['age', 'credit_sum', 'credit_month', 'tariff_id', 'score_shk',
        'monthly_income', 'credit_count', 'overdue_credit_count']
RScaler = RobustScaler()


# In[9]:

mean_values=train[numeric_fields].mean()
train_sc = RScaler.fit_transform(train[numeric_fields].fillna(value=mean_values))


# In[10]:

lm_ridge = Ridge(random_state=42, alpha=1e-08)
lm_ridge.fit(train_sc, train['open_account_flg'])


# In[11]:

y_predicted = lm_ridge.predict(train_sc)
y_predicted_bin = (y_predicted>0.20).astype(int)


# In[12]:

import pylab as pl
from sklearn.metrics import roc_curve, auc


# In[13]:

fpr, tpr, thresholds = roc_curve(train['open_account_flg'], y_predicted_bin)
roc_auc = auc(fpr, tpr)
print("Area under the ROC curve : %f" % roc_auc)


# In[14]:

# Plot ROC curve
pl.clf()
pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
pl.plot([0, 1], [0, 1], 'k--')
pl.xlim([0.0, 1.0])
pl.ylim([0.0, 1.0])
pl.xlabel('False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic')
pl.legend(loc="lower right")
pl.show()


# In[15]:

fpr, tpr, thresholds = roc_curve(train['open_account_flg'], y_predicted)
roc_auc = auc(fpr, tpr)
print("Area under the ROC curve : %f" % roc_auc)


# In[16]:

# Plot ROC curve
pl.clf()
pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
pl.plot([0, 1], [0, 1], 'k--')
pl.xlim([0.0, 1.0])
pl.ylim([0.0, 1.0])
pl.xlabel('False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic')
pl.legend(loc="lower right")
pl.show()


#  
#  
#  
#  

# ## 4. Добавление данных, инжиниринг фич, чистка данных, кодирование.
# ### были добавлены данные по регионам: 
# ### - средняя зарплата по возрастным группам
# ### - население по возрастным группам и уровню образования
# 
# Нормализованы названия регионов, заполнены пропуски (средние значения для социальной группы), 
# 
# категориальные переменные закодированы One-Hot кодированием.

# In[ ]:




# ## 5. Срез состояния и анализ
# 
# ### конкурс даёт дополнительную информацию в виде результатов других участников:
# 
# ![topic](meetup_10_Mar_img1.png)

# In[ ]:




# ## 6. Почему регионы написаны по разному

# In[17]:

train['living_region'][train['living_region'].str.match('МОСКОВСКА', as_indexer=True, na=False)].value_counts()


# In[18]:

train['living_region'][train['living_region'].str.match('ТОМСК', as_indexer=True, na=False)].value_counts()


# In[ ]:




# In[19]:

train['open_account_flg'][train['living_region'].str.match('^ОБЛ ', as_indexer=True, na=False)]    .describe()[['count', 'mean']]


# In[20]:

train['open_account_flg'][train['living_region'].str.match('.* ОБЛ$', as_indexer=True, na=False)]    .describe()[['count', 'mean']]


# In[21]:

train['open_account_flg'][train['living_region'].str.match('.*ОБЛАСТЬ', as_indexer=True, na=False)]    .describe()[['count', 'mean']]


# In[22]:

train['living_region'][train['living_region'].str.match('МОСКВА', as_indexer=True, na=False)].value_counts()


# In[23]:

train['open_account_flg'][train['living_region'].str.match('^МОСКВА$', as_indexer=True, na=False)]    .describe()[['count', 'mean']]


# In[24]:

train['open_account_flg'][train['living_region'].str.match('^МОСКВА Г$', as_indexer=True, na=False)]    .describe()[['count', 'mean']]


# ### ! способ написания региона несет дополнительную информацию !

#  

# ## 7. Визуальный анализ

# In[25]:

edu = train[['education','open_account_flg']].groupby('education').agg(['mean', 'count']).sort_values(('open_account_flg','mean'))
edu_bins = []
s = 0
for c in edu[('open_account_flg','count')]:
    edu_bins.append(s+c/2.0)
    s += c
fig = plt.figure(figsize=(12,6))

width = edu_bins[-1]
ind = np.arange(len(edu[('open_account_flg','count')]))
plt.bar(edu_bins, edu[('open_account_flg','mean')], width=0.9*edu[('open_account_flg','count')])
plt.xticks(edu_bins, list(edu.index))

fig.autofmt_xdate()


# In[26]:

job = train[['job_position','open_account_flg']]        .groupby('job_position')        .agg(['mean', 'count'])        .sort_values(('open_account_flg','mean'))
job_bins = []
s = 0
for c in job[('open_account_flg','count')]:
    job_bins.append(s+c/2.0)
    s += c
fig = plt.figure(figsize=(12,6))

#drop last three outliers
plt.bar(job_bins[:-3], job[('open_account_flg','mean')][:-3], width=0.95*job[('open_account_flg','count')][:-3])
plt.xticks(job_bins, list(job.index))

fig.autofmt_xdate()


# In[27]:

region = train[['living_region','open_account_flg']]        .groupby('living_region')        .agg(['mean', 'count'])        .sort_values(('open_account_flg','mean'))
region_bins = []
s = 0
for c in region[('open_account_flg','count')]:
    region_bins.append(s+c/2.0)
    s += c
fig = plt.figure(figsize=(15,6))

#drop single last outlier
plt.bar(region_bins[:-1], region[('open_account_flg','mean')][:-1], width=0.95*region[('open_account_flg','count')][:-1])
plt.xticks(region_bins, list(region.index))

fig.autofmt_xdate()


# Посмотрим на карте:
# 
# Процент
# ![map](meetup_10_Mar_img2.png)
# 
# Количество
# ![map](meetup_10_Mar_img3.png)

# In[ ]:




# In[28]:

train[['age', 'monthly_income', 'credit_sum', 'credit_month']].hist(figsize=(10,7))


# In[29]:

train[train['monthly_income'] < 100000][['monthly_income']].hist(bins=100, figsize=(12,7))


# In[30]:

train[train['credit_sum'] < 60000][['credit_sum']].hist(bins=200, figsize=(12,7))


# In[35]:

train[['score_shk']].hist(bins=200, figsize=(12,7))


# In[43]:

(np.log(1+train[['score_shk']])).hist(bins=200, figsize=(12,7))


# ### Новые фичи: 
# #### кратность monthly_income к 5000 и к 1000
# #### кратность credit_sum к 1000 и 100

# In[ ]:




# In[ ]:




# ## 8. Дубликаты записей
# 
# ### Да, мы подали заявку на получение кредита!
# ### Условия нам не понравились и мы подали еще одну.

# In[31]:

user_specific_columns = ['gender', 'age', 'marital_status', 'job_position', 'score_shk', 'education', 
                         'living_region', 'monthly_income', 'credit_count', 'overdue_credit_count']


# In[33]:

sum(train.duplicated(subset=user_specific_columns,keep=False))


# In[34]:

train['open_account_flg'][train.duplicated(subset=user_specific_columns,keep=False)]    .describe()[['count', 'mean']]


# ### Новые фичи: 
# #### флаг повторяемости записи
# 
# #### более продвинутый вариант - построить модель, определяющую вероятность взятия кредита из нескольких заявок

# In[ ]:




# ## 9. Стэкирование моделей

# ### блендинг - усреднение нескольких моделей
# ### R = 0.5 * M1 + 0.5 * M2
# 
# Пример - 12-ое место: https://github.com/TAPAKAH68/tinkoff_challenge_1
# 
# 
# 
# ### Более сложный пример - пять моделей с подбором коэффициэнтов (3-е место):
# https://github.com/VasiliyRubtsov/Tinkoff/blob/master/tinkof_1.ipynb
# 
