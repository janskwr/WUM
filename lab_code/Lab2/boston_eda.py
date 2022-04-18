#!/usr/bin/env python
# coding: utf-8

# # Pakiety

import pandas as pd
import numpy as np
import sklearn 
from sklearn.datasets import load_boston
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import warnings
warnings.filterwarnings('ignore')
np.random.seed(23)

r"""
COLOR = 'white'
matplotlib.rcParams['text.color'] = COLOR
matplotlib.rcParams['axes.labelcolor'] = COLOR
matplotlib.rcParams['xtick.color'] = COLOR
matplotlib.rcParams['ytick.color'] = COLOR
matplotlib.rcParams['figure.figsize'] = (10, 6)
matplotlib.rcParams['font.size'] = 14"""


# # Boston - EDA


# wczytywanie danych
# tak jak w lab_01
boston_dict = load_boston()
boston_df = pd.DataFrame(boston_dict['data'], columns=boston_dict['feature_names'])
boston_df['MEDV'] = boston_dict['target']

# zawsze na początku patrzymy na "głowę" df
# to taki odruch bezwarunkowy \:)
print(boston_df.head())


# Metody .info() i .describe() to również nasi przyjaciele

boston_df.info()
# nie ma braków, tylko dane numeryczne

boston_df.describe()

# ### Zobaczmy rozkłady poszczególnych zmiennych
# Pomimo, że wszystkie zmienne są typu *float*, to nadal możemy mieć zmienne dyskretne/kategoryczne  
# Można to sprwadzić przy użyciu metody *value_counts()* oraz **opisu danych**  
# W tym przypadku, poza CHAS nie ma takiej sytuacji

#boston_df.hist(figsize=(25, 12), bins=40)
boston_df.hist(figsize=(18, 12), bins=30)
plt.show()

# tu występuje dość mało wartośći, ale to nie jest zmienna kategoryczna
boston_df['RAD'].value_counts()


# ### Uwaga! Z naszą zmienną targetową jest pewien problem

boston_df['MEDV'].value_counts()


# Zmienna MEDV została *zczapkowana* (*ang. capped*) - ucięta na wartości 50.0

# # Przegląd wizualizacji




plt.plot(boston_df['RM'], 'o')
plt.title('Zmienna RM (liczba pokoi) w kolejnych rekordach')
plt.xlabel('rekordy')
plt.ylabel('RM')
plt.show()


plot_dens=sns.distplot(boston_df['RM'], hist = True, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3})
plot_dens.set_title('Rozkład zmiennej RM')
plt.show()


box_plot = sns.boxplot(boston_df['RM'])
box_plot.set_title('Zmienna RM')

print('mediana = %s' % np.median(boston_df['RM']))
print('średnia = %s' % np.mean(boston_df['RM']))
print('Q1 = %s' %np.percentile(boston_df['RM'], 25),'Q3 = %s' %np.percentile(boston_df['RM'], 75))


sns.pairplot(boston_df.iloc[:,[0,5,10]])
plt.tight_layout()

# tak naprawdę wystarczyłoby narysowanie tylko części wykresów
# czy jest koleracja zmiennych?

corr=boston_df.iloc[:,np.r_[0:7,10]].corr()
ax=sns.heatmap(corr, 
            xticklabels=corr.columns,
            yticklabels=corr.columns, annot=True)

# below is a workaround for matrix truncation
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
ax.set_title('Korelacja zmiennych')
plt.show()


sns.lmplot( x='LSTAT', y='RM', data=boston_df, size=7, aspect=1.3)
plt.show()
#fit_reg=False, # No regression line
#dodaje automatycznie prostą regresji

