#!/usr/bin/env python
# coding: utf-8

# # Wstęp do Uczenia Maszynowego - Lab 5

# Wszystkie dzisiejsze metody można znaleźć dokładnie (i przystępnie!) omówione na StatQuest: https://www.youtube.com/user/joshstarmer




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
np.random.seed = 42





data = pd.read_csv('heart.csv')
data.head()





y = np.array(data['chd'])
X = data.drop(['chd'],axis=1)





map_dict = {'Present': 1, 'Absent':0}
X['famhist'] = X['famhist'].map(map_dict)
X.head()





from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


# ## Ensemble Methods - Komitety
# Często kilka osób, które myślą nad daną sprawą, potrafi dać lepszą odpowiedź niż jedna osoba. Nawet jeśli żadna z osób nie jest ekspertem.  
# To podejście zadziała, gdy błędny popełniane przez różne modele są od siebie niezależne (w miarę). 
# 
#   
# Na potrzeby stosowania różnych metod Ensemble Learningu załadujemy sobie już 3 modele z których będziemy potem korzystać.




from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

model1 = DecisionTreeClassifier(random_state=1)
model2 = KNeighborsClassifier()
model3 = LogisticRegression(random_state=1, max_iter=1000)
estimators=[('DecisionTree', model1), ('KNN', model2), ('LR', model3)]


# ### Max Voting
# lub Hard Voting - głosowanie większościowe 




from sklearn.ensemble import VotingClassifier





model_hard = VotingClassifier(estimators=estimators, voting='hard')
model_hard.fit(X_train,y_train)

y_hat = model_hard.predict(X_test)
print('accuracy: ', accuracy_score(y_test, y_hat))

# model też ma inną metodę ewaluacji i jest to też accuracy w przypadku klasyfikacji
print('model.score: ', model_hard.score(X_test,y_test))


# ### Averaging
# Soft Voting. Nie patrzymy na liczbę głosów, ale na "pewność". Patrzymy na ile pewny jest klasyfikator, że rekord należy do klasy 1. c1 - 90%, c2 - 49%, c3 - 49% 
# * hard voting: klasa 0 (1 głos na tak, 2 głosy na nie)
# * soft voting: klasa 1 ( (90 + 49 + 49)/3 $\approx$ 63% > 50%)
# 




model_soft = VotingClassifier(estimators=estimators, voting='soft')
model_soft.fit(X_train, y_train)

y_hat = model_soft.predict(X_test)
accuracy_score(y_test, y_hat)


# # Weights
# Nie musimy zawsze uważać, że każdy klasyfikator ma tyle samo do powiedzenia.




model_soft = VotingClassifier(estimators=estimators, voting='soft', weights=[0.05, 0.05, 0.90])
model_soft.fit(X_train, y_train)

y_hat = model_soft.predict(X_test)
accuracy_score(y_test, y_hat)


# ### Stacking
# <div>
# <img src="https://miro.medium.com/max/700/1*RP0pkQEOSrw9_EjFu4w3gg.png" />
# </div>
# source: https://miro.medium.com/max/700/1*RP0pkQEOSrw9_EjFu4w3gg.png

# Bierzemy kilka różnych modeli (base) i jeden meta-model. Meta-model uczy się przewidzieć wynik na podstawie wyników z base.
# 
# Zasada kciuka: ostatni model jest raczej prosty (regresja liniowa/logistyczna).  
# Więcej (np. o trenowaniu) tutaj: https://machinelearningmastery.com/stacking-ensemble-machine-learning-with-python/
# 
# 
# Czy można stackować kilka takich samych modeli?  
# Tak, omówimy to za chwilę.




from sklearn.ensemble import StackingClassifier





clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
clf.fit(X_train, y_train).score(X_test, y_test)


# ### Bagging (Bootstrap Aggregating)
# Bootstrap - to technika próbkowania, w której tworzymy podzbiory (próby) obserwacji z oryginalnego datasetu, **ze zwracaniem**. Rozmiar podzbiorów jest taki sam jak rozmiar oryginalnego datasetu.
# 
# 1. Losujemy N **bootstrapowych** prób ze zbioru treningowego
# 2. Trenujemy niezależnie N "słabych" klasyfikatorów
# 3. Składamy wyniki "słabych" modeli 
#     - **Klasyfikacja:** reguła większościowa / uśrednione prawdopodobieństwo
#     - **Regresja:** Uśrednione wartości




from sklearn.ensemble import BaggingClassifier





clf = BaggingClassifier(base_estimator=model1,
                        n_estimators=10, random_state=0)
clf.fit(X_train, y_train)
clf.score(X_test,y_test)


# ## Random Forest
# Najbardziej popularny algorytm Baggingowy. Wiele małych drzew.
# 
# Przypomnijmy
# #### Zalety drzew
# * interpretowalność
# * prostota + wizualizacja
# * nie trzeba normalizować danych \:)
# * odporne na wartości skrajne
# * z ich użyciem można wykrywać ważne cechy
# 
# #### Wady drzew
# * łatwo o overfitting
# * drzewo długo rośnie
# * starych drzew się nie przesadza - nie da się dotrenować algorytmu po jakimś czasie, trzeba zasadzić nowe 
# * jest duża losowość - na tych samych danych algorytm może dawać bardzo różne wyniki




from sklearn.ensemble import RandomForestClassifier

model_rf = RandomForestClassifier(n_estimators=1000, # Ilość słabych estymatorów
                                  max_depth=2, # Maksymalna wysokość drzewa w słabym estymatorze
                                  min_samples_split = 2, # Minimalna ilość obserwacji wymagana do podziału węzła
                                  max_features = 3, # Maksymalna ilość zmiennych brana pod uwagę przy podziale węzła
                                  random_state=0,
                                  n_jobs = -1)
model_rf.fit(X_train, y_train)
model_rf.score(X_test,y_test)


# ### Boosting
# Boosting działa podobnie jak Bagging z jedną różnicą. Każda kolejna próba bootstrap jest tworzona w taki sposób,  
# że losuje z większym prawdopodobieństwiem obserwacje **źle sklasyfikowane**.  
# W skrócie: Boosting uczy się na błędach, które popełnił w poprzednich iteracjach.

# #### AdaBoost
# Najprostsza metoda boostingowa.  
# Większe prawdopodobieństwo wylosowania próbek, na których podczas predykcji został popełniony błąd.




from sklearn.ensemble import AdaBoostClassifier

model = AdaBoostClassifier(random_state=1)
model.fit(X_train, y_train)
model.score(X_test,y_test)


# #### Gradient Boosting
# Zaczynamy od pojedynczej predykcji (np. średniej) i liczymy różnicę, tzw. residuum.  
# Następnie każdy model próbuje przewidzieć residuum.  
# W GB każdy model ma taki sam głos.




from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(random_state=1,
                                  learning_rate=0.01)
model.fit(X_train, y_train)
model.score(X_test,y_test)


# #### XGBoost
# Zaawansowana implementacja Gradient Boostingu




from xgboost import XGBClassifier # Inna paczka niż sklearn!





model=XGBClassifier(random_state=1,
                    learning_rate=0.01, # Szybkość "uczenia" się
                    booster='gbtree', # Jaki model wykorzystujemy (drzewo - gbtree, liniowe - gblinear)
                    max_depth=4 # Maksymalna głębokość drzewa 
                    )
model.fit(X_train, y_train)
model.score(X_test,y_test)


from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import StackingClassifier
from matplotlib import pyplot

import warnings
warnings.filterwarnings('ignore')

# przygotowanie stacking ensemble
def get_stacking():
    # define the base models
    level0 = list()
    level0.append(('cart', DecisionTreeClassifier(random_state=1)))
    level0.append(('knn', KNeighborsClassifier()))
    level0.append(('rf', RandomForestClassifier(n_estimators=1000, # Ilość słabych estymatorów
                                      max_depth=2, # Maksymalna wysokość drzewa w słabym estymatorze
                                      min_samples_split = 2, # Minimalna ilość obserwacji wymagana do podziału węzła
                                      max_features = 3, # Maksymalna ilość zmiennych brana pod uwagę przy podziale węzła
                                      random_state=0,
                                      n_jobs = -1)))
    level0.append(('gboost', GradientBoostingClassifier(random_state=1,
                                      learning_rate=0.01)))
    level0.append(('xgb', XGBClassifier(random_state=1,
                        learning_rate=0.01, # Szybkość "uczenia" się
                        booster='gbtree', # Jaki model wykorzystujemy (drzewo - gbtree, liniowe - gblinear)
                        max_depth=4 # Maksymalna głębokość drzewa 
                        )))
    level0.append(('svm', SVC()))
    level0.append(('bayes', GaussianNB()))
    # definicja meta learner model
    level1 = LogisticRegression()
    # definicja stacking ensemble
    model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
    
    return model

# lista modeli
def get_models():
    
    models = dict()
    models['cart'] = DecisionTreeClassifier()
    models['knn'] = KNeighborsClassifier()
    models['rf'] = RandomForestClassifier(n_estimators=1000, # Ilość słabych estymatorów
                                      max_depth=2, # Maksymalna wysokość drzewa w słabym estymatorze
                                      min_samples_split = 2, # Minimalna ilość obserwacji wymagana do podziału węzła
                                      max_features = 3, # Maksymalna ilość zmiennych brana pod uwagę przy podziale węzła
                                      random_state=0,
                                      n_jobs = -1)
    models['gboost'] = GradientBoostingClassifier(random_state=1,
                                      learning_rate=0.01)
    models['xgb'] = XGBClassifier(random_state=1,
                        learning_rate=0.01, # Szybkość "uczenia" się
                        booster='gbtree', # Jaki model wykorzystujemy (drzewo - gbtree, liniowe - gblinear)
                        max_depth=4 # Maksymalna głębokość drzewa 
                        )
    models['svm'] = SVC()
    models['bayes'] = GaussianNB()
    models['stacking'] = get_stacking()

    return models

# evaluate a give model using cross-validation
def evaluate_model(model, X, y):
    
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')

    return scores

# modele do evaluacji
models = get_models()
# ocena modeli
results, names = list(), list()
for name, model in models.items():
    scores = evaluate_model(model, X, y)
    results.append(scores)
    names.append(name)
    print('>%s %.3f (%.3f)' % (name, np.mean(scores), np.std(scores)))
# plot modele
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()


# ### Wszystkie dzisiejsze metody można znaleźć dokładnie (i przystępnie!) omówione na StatQuest: https://www.youtube.com/user/joshstarmer






