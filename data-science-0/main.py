#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# In[127]:





# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[4]:


def q1():
    return black_friday.shape


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[85]:


def q2():
    filter_age = ['26-35']
    filter_sex = ['F']
    return black_friday[black_friday['Age'].isin(filter_age) & black_friday['Gender'].isin(filter_sex)].shape[0]


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[78]:


def q3():
    feature_unique = ['User_ID']
    return int(black_friday[feature_unique].nunique()[0])


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[81]:


def q4():
    return int(black_friday.dtypes.nunique())


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[190]:


def q5():
    total_registros = black_friday.shape[0]
    num_inner = black_friday[black_friday['Product_Category_2'].isna() & black_friday['Product_Category_3'].isna()].shape[0]
    num_left = black_friday['Product_Category_2'].isna().sum()
    num_right = black_friday['Product_Category_3'].isna().sum()
    return float(num_right + num_left - num_inner)/float(total_registros)


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[126]:


def q6():
    most_na_feat = black_friday.columns[0]
    for feature in black_friday:
        if black_friday[feature].isna().sum() > black_friday[most_na_feat].isna().sum():
            most_na_feat = feature
    return int(black_friday[most_na_feat].isna().sum())


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[201]:


def q7():
    feature = ['Product_Category_3']
    return int(black_friday[feature].mode().iloc[0])


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[140]:


def q8():
    s1 = black_friday['Purchase']
    s1 = ((s1-s1.min())/(s1.max()-s1.min()))
    return float(s1.mean())


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[155]:


def q9():
    s1 = black_friday['Purchase']
    s1 = (s1 - s1.mean()) / s1.std()
    return s1[(s1 >= -1) & (s1 <= 1)].shape[0]


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[172]:


def q10():
    for index, row in black_friday.iterrows():
        if np.isnan(black_friday.iloc[index]['Product_Category_2']) and not np.isnan(black_friday.iloc[index]['Product_Category_3']):
            return False
    return True


# In[ ]:




