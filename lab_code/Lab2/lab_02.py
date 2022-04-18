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


# # Zbiór danych nt. butów męskich
# ## Naszym celem jest predykcja ceny - prices_amountmin

# wczytujemy dane 
shoes_df = pd.read_csv('menshoes.csv')

# w tym momencie następuje odruch bezwarunkowy
shoes_df.head()

# jakie wnioski?
# @conclusions

#  warto przyjrzeć się danym pod innym kątem
shoes_df.info()
# @conclusions

#usuwamy kolumny z NaN
threshold = len(shoes_df) * 0.7
shoes_df_drop = shoes_df.loc[:, shoes_df.apply(lambda x: x.isna().sum(), axis=0) < threshold]
shoes_df_drop.head(10)


# które kolumny jeszcze wydają się niepotrzebne?  
# które kolumny trzeba przekształcić?

# @conclusions

# popatrzmy na kolory butów

df_count = shoes_df_drop['colors'].value_counts()

# jakie jest ograniczenie metody .value_counts()?
# @conclusion

# wartości NA
pd.DataFrame(shoes_df_drop['colors']).info()

# alternatywny sposób

# jest sporo braków - sposoby na zaadresowanie tego problemu będą w dalszej części  
# 
# ### Najpierw wizualizacja  
# jak przedstawić zmienną ciągłą vs. kategoryczną?

Black_Brown = shoes_df_drop.loc[shoes_df_drop['colors'].isin(['Black', 'Brown'])]
violin_plot = sns.violinplot(Black_Brown['colors'], Black_Brown['prices_amountmin'])
violin_plot.set_title('Rozkład ceny względem koloru butów')
plt.show()

popular_colors=shoes_df_drop.loc[shoes_df_drop['colors'].isin(shoes_df_drop['colors'].value_counts().index[:5])]
mean_price_popular_colors=popular_colors.groupby('colors')['prices_amountmin'].mean()
mean_price_popular_colors.plot(kind='bar', title='Średnia cena')
plt.show()

# # Preprocessing danych

# # Imputacja zmiennych kategorycznych

# kolumna colors miała dużo nieuzupełnionych wartości  
# jak można je uzupełnić?

# możnaby najczęściej występującym kolorem, ale czy to nie wprowadza fałszywej informacji?
# inne podejście - nowa klasa 'Other'
shoes_df_drop['colors'].fillna('Missing_color')  

# inny sposób? tworzymy nową zmienną
shoes_df_drop['missing' + 'colors'] = shoes_df_drop['colors'].isna()*1
shoes_df_drop['colors'].fillna('Missing_color', inplace=True)


# ### Uwaga, zaprezentowane powyżej rozwiązanie nie jest optymalne
# ### przekonamy się o tym przy one hot encoding

# # Imputacja zmiennych ciągłych

# w naszym zbiorze nie ma nic do imputacji ciągłej
# zerknijmy na szutczny zbiór
fake_data=pd.DataFrame({'num':np.random.choice([None, 3,4], 100), 
                        'cat': np.random.choice([None, 'Puma','Nike','Adidas'], 100, p=[0.92, 0.03, 0.03, 0.02])})
fake_data.head()

# średnia czy mediana?
fake_data.num.fillna(fake_data.num.median(), inplace=True) #fake_data.num.mean()
fake_data.info()


# # Outliery

# dane jedynie z przedziału (średnia +- 3 sigma) - zasadne podejście gdy rozkład normalny
data=pd.DataFrame({'num':np.random.normal(2,0.4,1000)})
factor = 3
upper_lim = data['num'].mean () + data['num'].std () * factor
lower_lim = data['num'].mean () - data['num'].std () * factor

# Metoda 1
# usunięcie rekordów !!!  UWAGA NIE POWINNO SIĘ STOSOWAĆ
data = data[(data['num'] < upper_lim) & (data['num'] > lower_lim)]
data.shape

# Metoda 2
# zastąpienie wartosci outlierow wartosciami skrajnymi
data['num'] = np.where(data['num'] < upper_lim, data['num'], upper_lim)
data['num'] = np.where(data['num'] > lower_lim, data['num'], lower_lim)


# usuwanie na podstawie skrajnych percentyli
# zasadne dla każdego rozkładu

# w celu zaprezentowania tej metody posłużymy się zbiorem z poprzednich zajęć nt. domów w Bostonie
boston_dict = load_boston()
boston_df = pd.DataFrame(boston_dict['data'], columns=boston_dict['feature_names'])

dis_data = boston_df['DIS']
print('Wejściowy rozmiar: ', dis_data.shape[0])

upper_lim = dis_data.quantile(.95)
lower_lim = dis_data.quantile(.05)

data_percentile = dis_data[(dis_data < upper_lim) & (dis_data > lower_lim)]
print('Wyjściowy rozmiar: ', data_percentile.shape[0])
print('Usunięto %: ', round(data_percentile.shape[0]/dis_data.shape[0], 2))
# Czy zastosowane podejście było dobre?
# @conclusion

# A może boxplot?
# Okazuje się, że nie powinniśmy usuwać najmniejszych danych
# ale faktycznie wydaje się, że wartości maksymalne są outlierami
plot_box = sns.boxplot(boston_df['DIS'])
plot_box.set_title('Rozkład zmiennej DIS')
plt.show()


# Jednak dopiero analizowanie danych w postaci histogramu pokazuje, że wartości maksymalne nie są
# anomaliami - wpisują się w ten tym rozkładu
# wniosek: w przypadku tej zmiennej nie ma outlierów
plot_dens=sns.histplot(boston_df['DIS'])
plot_dens.set_title('Rozkład zmiennej DIS')
plt.show()

# W przypadku dużej liczby zmnienej nalezy stosowac Metode 2

# #  Grouping & Binning
# ### Agregujemy klasy do wyższego poziomu lub tniemy zmienną ciągłą na klasy

# czasami potrzebujemy zrobić ze zmiennej ciągłej kategoryczną  
# albo mamy zmienną kategoryczną o bardzo dużej liczbie klas  
# albo dużo klas mało licznych

# zobaczmy jak wygląda kolumna brand 
shoes_df_drop['brand'].value_counts()
# co można z tym zrobić?
# @conclusions

# jest aż 560 marek, które występują raz
shoes_df_drop['brand'].value_counts()[shoes_df_drop['brand'].value_counts() == 1].shape[0]

# czyli grupujemy w kategorię Other?
# @conclusion

# przyjrzyjmy się bliżej
brands = shoes_df_drop[['brand']].groupby(['brand']).size().sort_values(ascending=False).reset_index()
brands.columns = ['brand', 'count']
brands.loc[brands['brand'].apply(lambda x:'nike' in x.lower())]


# może warto najpierw zgrupować rekordy które de facto dotyczą marki Nike w jedną kategorię a 
# dopiero potem tworzyć kategorie Others?  
# Nike i NIKE to na pewno to samo, ale może NIKE - Kobe to dość niszowe obuwie i warto, żeby było Others?  
# to samodzielna decyzja

nike_synonyms = brands.loc[brands['brand'].apply(lambda x:'nike' in x.lower()), 'brand'].values

small_classes = shoes_df_drop['brand'].value_counts()[shoes_df_drop['brand'].value_counts() == 1].index

shoes_df_drop['brands' + '_processed'] = np.where(shoes_df_drop['brand'].isin(nike_synonyms), 'nike', 
                                              np.where(shoes_df_drop['brand'].isin(small_classes), 'Other', shoes_df_drop['brand']))

shoes_df_drop['brands' + '_processed'].value_counts()

################################################################################

# można to zrobić też przy pomocy słownika
geo=np.random.choice(("Poland",'Chile', 'France', 'Spain'), 100)
geo=pd.Series(geo)
geo

dict_geo={'Poland': "Europe", "Chile":"South America", "France":"Europe"}
geo.map(dict_geo)


# metoda z użyciem dict/defaultdict
from collections import defaultdict

countries_list = [('Poland','Europe'), ('France','Europe'), ('Chile','South America')]

countries_dict = defaultdict(lambda:'Other')
for continent, country in countries_list:
     countries_dict[continent]=country
geo.map(countries_dict)

################################################################################

# naszym celem było przewidywanie cen butów
# ale może wystarczy jeśli przewidzimy to z mniejszą granularnością? 
# Może wystarczy informacja, że coś jest tanie, średnio kosztowne, drogie
prices_hist = sns.histplot(shoes_df_drop['prices_amountmin'])
prices_hist.set_title('Rozkład cen butów')
plt.show()

# pd.cut(shoes_df_drop['prices_amountmin'], bins=[0, 100, 200, 250], labels=['cheap', 'affordable', 'expensive'])[17:]

cutted = pd.cut(shoes_df_drop['prices_amountmin'], bins=[0, 100, 200, np.inf], labels=['cheap', 'affordable', 'expensive'])
cutted[18:]

# zawsze warto sprawdzić
cutted[cutted.isna()]

# i spróbować zrozumieć dlaczego
shoes_df_drop.loc[4176, 'prices_amountmin']

# # Dla przypomnienia - Log Transform

dis_dist = sns.distplot(boston_df['DIS'])
dis_dist.set_title('Rozkład zmiennej DIS')
plt.show()


dis_log_dist = sns.distplot(np.log1p(boston_df['DIS']))
dis_log_dist.set_title('Rozkład logarytmu zmiennej DIS')
plt.show()
# wykres jest bliższy rozkładowi normalnemu 


# # Categorical variables encoding
# ### algorytmy często "nie lubią" zmiennych kategorycznych

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

data = ['cold', 'cold', 'warm', 'cold', 'hot', 'hot', 'warm', 'cold', 'warm', 'hot']
values = np.array(data)

# integer encode
le = LabelEncoder()
integer_encoded = le.fit_transform(values)
print(integer_encoded)

# Warto zauważyć, że to ma sens, tylko dla zmiennych, które reprezentują jakieś poziomy/kolejność (hierarchię)

# uwaga! 
# Nie mamy wpływu na kolejność przypisań
data = ['hot', 'cold', 'warm', 'cold', 'hot', 'hot', 'warm', 'cold', 'warm', 'hot']
values = np.array(data)
le = LabelEncoder()
integer_encoded = le.fit_transform(values)
print(integer_encoded)


# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded)


# przykład z naszych danych - kolumna categories
shoes_df_drop.head()

shoes_df_drop['categories'].value_counts()

categories = {}
def split_and_count(x, categories):
    cat_list = x.split(',')
    for cat in cat_list:
        categories.setdefault(cat, 0)
        categories[cat] += 1
    return categories

for row in shoes_df_drop['categories'].iteritems():
    print(row[1])
    split_and_count(row[1], categories)
categories_df = pd.DataFrame.from_dict(categories, orient='index').reset_index()
categories_df.columns = ['category', 'count']
categories_df.sort_values(by='count', ascending=False).head(20)

# kategorie typu athletic, *sport, *outwear mogą się być transformowane z użyciem na one-hot encoding


# # Scaling
# - min-max scaling
# - standarization

# niektóre algorytmy nie lubią dużych skal zmiennych - regresja liniowa z poprzednich zajęć   
# jakiś parametr musi "obsłużyć" bardzo małe i bardzo duże liczby  
# inne potrzebują mieć zmienne w konkretnym przedziale


# min-max scaling
from sklearn.preprocessing import MinMaxScaler
data = np.array([-1, 2, -0.5, 6, 0, 10, 1, 18]).reshape(-1, 1)
mm_scaler = MinMaxScaler()

print(mm_scaler.fit_transform(data))

# standarization
from sklearn.preprocessing import StandardScaler
data = np.array([-1, 2, -0.5, 6, 0, 10, 1, 18]).reshape(-1, 1)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
print(scaled_data)
print(np.round(np.mean(scaled_data), 4), np.std(scaled_data))


# # Extracting info from date

from datetime import date

data = pd.DataFrame({'date':
['01-01-2017',
'04-12-2008',
'23-06-1988',
'25-08-1999',
'20-02-1993',
]})

#Transform string to date
data['date'] = pd.to_datetime(data.date, format="%d-%m-%Y")

#Extracting Year
data['year'] = data['date'].dt.year

#Extracting Month
data['month'] = data['date'].dt.month

#Extracting passed years since the date
data['passed_years'] = date.today().year - data['date'].dt.year

#Extracting passed months since the date
data['passed_months'] = (date.today().year - data['date'].dt.year) * 12 + date.today().month - data['date'].dt.month

#Extracting the weekday name of the date
data['day_name'] = data['date'].dt.day_name()

data


# # warto poczytać
# pakiet category_encoders:
# - https://kiwidamien.github.io/encoding-categorical-variables.html
# - https://pbpython.com/categorical-encoding.html

# Ciekawa strona z przykładami wizualizacji (wraz z kodem):
#     https://www.machinelearningplus.com/plots/top-50-matplotlib-visualizations-the-master-plots-python/
