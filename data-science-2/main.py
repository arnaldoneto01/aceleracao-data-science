#!/usr/bin/env python
# coding: utf-8

# # Desafio 4
# 
# Neste desafio, vamos praticar um pouco sobre testes de hipóteses. Utilizaremos o _data set_ [2016 Olympics in Rio de Janeiro](https://www.kaggle.com/rio2016/olympic-games/), que contém dados sobre os atletas das Olimpíadas de 2016 no Rio de Janeiro.
# 
# Esse _data set_ conta com informações gerais sobre 11538 atletas como nome, nacionalidade, altura, peso e esporte praticado. Estaremos especialmente interessados nas variáveis numéricas altura (`height`) e peso (`weight`). As análises feitas aqui são parte de uma Análise Exploratória de Dados (EDA).
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns
import statsmodels.api as sm


# In[3]:


# %matplotlib inline

# from IPython.core.pylabtools import figsize


# figsize(12, 8)

# sns.set()


# In[4]:


athletes = pd.read_csv("athletes.csv")


# In[5]:


def get_sample(df, col_name, n=100, seed=42):
    """Get a sample from a column of a dataframe.
    
    It drops any numpy.nan entries before sampling. The sampling
    is performed without replacement.
    
    Example of numpydoc for those who haven't seen yet.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Source dataframe.
    col_name : str
        Name of the column to be sampled.
    n : int
        Sample size. Default is 100.
    seed : int
        Random seed. Default is 42.
    
    Returns
    -------
    pandas.Series
        Sample of size n from dataframe's column.
    """
    np.random.seed(seed)
    
    random_idx = np.random.choice(df[col_name].dropna().index, size=n, replace=False)
    
    return df.loc[random_idx, col_name]


# ## Inicia sua análise a partir daqui

# In[6]:


# Sua análise começa aqui.
athletes.head()


# In[7]:


athletes.describe()


# In[8]:


athletes.shape


# ## Questão 1
# 
# Considerando uma amostra de tamanho 3000 da coluna `height` obtida com a função `get_sample()`, execute o teste de normalidade de Shapiro-Wilk com a função `scipy.stats.shapiro()`. Podemos afirmar que as alturas são normalmente distribuídas com base nesse teste (ao nível de significância de 5%)? Responda com um boolean (`True` ou `False`).

# In[12]:


def q1():
    feature = "height"
    alpha = 0.05
    sample = get_sample(athletes, feature, n = 3000)
    pvalue = sct.shapiro(sample).pvalue
    return bool(pvalue > alpha)


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que? Não, o gráfico indica que os nossos dados seguem uma distribuição normal, enquanto o resultado do teste proveu forte evidência de que os dados não vêm de uma distribuição normal..
# * Plote o qq-plot para essa variável e a analise.
# * Existe algum nível de significância razoável que nos dê outro resultado no teste? (Não faça isso na prática. Isso é chamado _p-value hacking_, e não é legal). Poderíamos ter escolhido um alpha menor que o pvalue encontrado, que foi muito próximo a zero (5x10e-7). 

# In[19]:


sns.distplot(athletes["height"], bins=25);


# In[27]:


sm.qqplot(athletes["height"].dropna(), fit=True, line="45");


# ## Questão 2
# 
# Repita o mesmo procedimento acima, mas agora utilizando o teste de normalidade de Jarque-Bera através da função `scipy.stats.jarque_bera()`. Agora podemos afirmar que as alturas são normalmente distribuídas (ao nível de significância de 5%)? Responda com um boolean (`True` ou `False`).

# In[33]:


def q2():
    feature = "height"
    alpha = 0.05
    sample = get_sample(athletes, feature, n = 3000)
    pvalue = sct.jarque_bera(sample).pvalue
    return bool(pvalue > alpha)


# __Para refletir__:
# 
# * Esse resultado faz sentido? Ele não está condizente com o gráfico da amostra.

# ## Questão 3
# 
# Considerando agora uma amostra de tamanho 3000 da coluna `weight` obtida com a função `get_sample()`. Faça o teste de normalidade de D'Agostino-Pearson utilizando a função `scipy.stats.normaltest()`. Podemos afirmar que os pesos vêm de uma distribuição normal ao nível de significância de 5%? Responda com um boolean (`True` ou `False`).

# In[34]:


def q3():
    feature = "weight"
    alpha = 0.05
    sample = get_sample(athletes, feature, n = 3000)
    pvalue = sct.normaltest(sample).pvalue
    return bool(pvalue > alpha)


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que? Sim. O resultado do teste fornce evidência de que os dados não vêm de uma distribuição normal, e os gráficos corroboram com esse resultado.
# * Um _box plot_ também poderia ajudar a entender a resposta.

# In[37]:


sns.distplot(athletes["weight"], bins = 25);


# In[38]:


sm.qqplot(athletes["weight"].dropna(), fit=True, line="45");


# ## Questão 4
# 
# Realize uma transformação logarítmica em na amostra de `weight` da questão 3 e repita o mesmo procedimento. Podemos afirmar a normalidade da variável transformada ao nível de significância de 5%? Responda com um boolean (`True` ou `False`).

# In[13]:


def q4():
    feature = "weight"
    alpha = 0.05
    sample = get_sample(athletes, feature, n = 3000)
    sample_log = np.log(sample)
    pvalue = sct.normaltest(sample_log).pvalue
    return bool(pvalue > alpha)


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que? Não. O resultado do teste forneceu indícios de que os dados não vinham de uma distribuição normal, e o gráfico mostrou o contrário.
# * Você esperava um resultado diferente agora? Sim. Os dados estavam Skewed e era esperado que eles fosse normalizados após a transformação logaritmica.

# In[55]:


sns.distplot(np.log(athletes["weight"]), bins = 25);


# > __Para as questão 5 6 e 7 a seguir considere todos testes efetuados ao nível de significância de 5%__.

# ## Questão 5
# 
# Obtenha todos atletas brasileiros, norte-americanos e canadenses em `DataFrame`s chamados `bra`, `usa` e `can`,respectivamente. Realize um teste de hipóteses para comparação das médias das alturas (`height`) para amostras independentes e variâncias diferentes com a função `scipy.stats.ttest_ind()` entre `bra` e `usa`. Podemos afirmar que as médias são estatisticamente iguais? Responda com um boolean (`True` ou `False`).

# In[57]:


athletes.head()


# In[14]:


def q5():
    feature = "height"
    feature_filter = "nationality"
    nationalities = ["BRA", "USA"]
    alpha = 0.05
    samples = [athletes[athletes[feature_filter] == nationality][feature] for nationality in nationalities]
    pvalue = sct.ttest_ind(samples[0], samples[1], equal_var=False, nan_policy='omit').pvalue
    return bool(pvalue > alpha)


# ## Questão 6
# 
# Repita o procedimento da questão 5, mas agora entre as alturas de `bra` e `can`. Podemos afimar agora que as médias são estatisticamente iguais? Reponda com um boolean (`True` ou `False`).

# In[10]:


def q6():
    feature = "height"
    feature_filter = "nationality"
    nationalities = ["BRA", "CAN"]
    alpha = 0.05
    samples = [athletes[athletes[feature_filter] == nationality][feature] for nationality in nationalities]
    pvalue = sct.ttest_ind(samples[0], samples[1], equal_var=False, nan_policy='omit').pvalue
    return bool(pvalue > alpha)


# ## Questão 7
# 
# Repita o procedimento da questão 6, mas agora entre as alturas de `usa` e `can`. Qual o valor do p-valor retornado? Responda como um único escalar arredondado para oito casas decimais.

# In[12]:


def q7():
    feature = "height"
    feature_filter = "nationality"
    nationalities = ["USA", "CAN"]
    alpha = 0.05
    samples = [athletes[athletes[feature_filter] == nationality][feature] for nationality in nationalities]
    pvalue = sct.ttest_ind(samples[0], samples[1], equal_var=False, nan_policy="omit").pvalue
    return float(pvalue.round(8))


# __Para refletir__:
# 
# * O resultado faz sentido? Sim
# * Você consegue interpretar esse p-valor? Dado alpha = 0.05, nós reijeitamos a hipótese nula de que as médias das alturas são iguais para os casos BRA-USA e USA-CAN e nós não rejeitamos para os casos BRA-CAN. 
# * Você consegue chegar a esse valor de p-valor a partir da variável de estatística? Sim. t statistic e o p-value são relacionados matematicamente.

# In[ ]:




