#!/usr/bin/env python
# coding: utf-8

# ## Klasyfikacja 
# Klasyfikacja to rodzaj algorytmu statystycznego, który przydziela obserwacje statystyczne do klas, bazując na atrybutach tych obserwacji.
# 
# **Definicja:**
# Dla danego zbioru danych trenujących $\{(x_1,y_1),\ldots,(x_n,y_n)\}$ algorytm potrafi znaleźć funkcję klasyfikującją $h: X -> Y$, która przydziela obiektowi $x\in X$ klasę $y \in Y$.
# 
# - prawdopodobieństwo aposteriori: $P(Y=i|X)$ *
# - funkcja klasyfikacyjna przyjmuje postać: $h(X) = argmax_{1,\ldots,y} P(Y=i|X)$
# 
# *większość klasyfikatorów modeluje prawdopodobieństwa, wyjątek stanowi SVM
# 
# Przykłady klasyfikacji:
# - wykrywanie czy pacjent jest chory na daną chorobę na podstawie wyników badań
# - klasyfikacja maili jako spam/nie-spam
# - czy transakcja dokonana na koncie klienta banku to oszustwo/kradzież czy też normalna transakcja
# - rozpoznawania na obrazu różnych rodzajów zwierząt
# - rozpoznawanie czy pasażer przeżyje katastrofę na Titanicu
# 
# **Na potrzeby uproszczenia wyjaśniania w dalszej części labów, skupimy się tylko na klasyfikacji binarnej.**
# 
# Zajmiemy się zbiorem gdzie klasyfikujemy u pacjentów czy występuje choroba serca czy nie.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

import shap

warnings.filterwarnings('ignore')
np.random.seed = 42

# South African Hearth data set

data = pd.read_csv('heart.csv')
data.head()

# A retrospective sample of males in a heart-disease high-risk region of the Western Cape, South Africa. 
# There are roughly two controls per case of CHD. 
# Many of the CHD positive men have undergone blood pressure reduction treatment and other programs to reduce their risk factors after their CHD event. 
# In some cases the measurements were made after these treatments. 
# These data are taken from a larger dataset, described in Rousseauw et al, 1983, South African Medical Journal.

# The class label indicates if the person has a coronary heart disease: negative (0) or positive (1).

# Attributes description:

# 1- sbp: systolic blood pressure 
# 2- tobacco: cumulative tobacco (kg) 
# 3- ldl: low densiity lipoprotein cholesterol 
# 4- adiposity 
# 5- famhist: family history of heart disease (1=Present, 0=Absent) 
# 6- typea: type-A behavior 
# 7- obesity (is this bmi) 
# 8- alcohol: current alcohol consumption 
# 9- age: age at onset


# Szybko sprawdzamy podstawowe cechy danych
na_ratio_cols = data.isna().mean(axis=0)
na_ratio_cols

y = np.array(data['chd'])
data["famhist"]=np.where(data["famhist"]=="Present",1,0)
X = data.drop(['chd','famhist'],axis=1)
y

# Szybkie ćwiczenie - wykonaj dowolne kodowanie zmiennej kategorycznej

X.head()
X.info()

# ## Sposoby podziału danych
# - Jak radzić sobie z overfitingiem?
# - Jakie znacie sposoby podziału danych na treningowe i testowe?

# ![image.png](https://media.geeksforgeeks.org/wp-content/cdn-uploads/20190523171258/overfitting_2.png)

# https://www.geeksforgeeks.org/underfitting-and-overfitting-in-machine-learning/
# 
# ## Zbiór treningowy, walidacyjny i testowy¶
# Prosty podział danych na część, na której uczymy model i na część która służy nam do sprawdzenia jego skuteczności.


from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X, y, stratify=y, test_size=0.3, random_state=42
)
# X_val, X_test, y_val, y_test = train_test_split(
#     X_val, y_val, stratify=y_val, test_size=0.3, random_state=42
# )

pd.Series(y).hist()

# pd.Series(y_test).hist()

# pd.Series(y_test).hist()

# print(X.shape,X_train.shape, X_val.shape, X_test.shape)

# from sklearn.metrics import roc_curve, auc
from sklearn import metrics

def gini_roc(y_test, y_pred_proba, tytul):
    
    fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
    
    plt.plot(fpr,tpr)
    plt.title(tytul)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    
    roc_auc = metrics.auc(fpr, tpr)
    gini = (2 * roc_auc) - 1

    return gini

def gini_train_val(model, X_train, y_train, X_val, y_val):
    
    y_pred_proba = model.predict_proba(X_train)[::,1]
    gini_train = gini_roc(y_train, y_pred_proba, "ROC Curve for Training Sample")
    print("gini_test: %.4f" % gini_train)
    
    y_pred_proba = model.predict_proba(X_val)[::,1]
    gini_val = gini_roc(y_val, y_pred_proba, "Roc Curve for Validation Sample")
    print("gini_val: %.4f" % gini_val)

    return

def shapley(model, X_train, X_val):
        
    # explain the model's predictions using SHAP
    # (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
    # model = lr
    
    explainer = shap.Explainer(model, X_train)
    
    shap_values = explainer(X_train)
    shap.summary_plot(shap_values, X_train)
    # visualize the first prediction's explanation
    shap.plots.waterfall(shap_values[0])

    # freature importance    
    shap.summary_plot(shap_values, X_train, plot_type="bar")
   
    shap_values = explainer(X_val)
    shap.plots.beeswarm(shap_values)
    # visualize the first prediction's explanation
    shap.plots.waterfall(shap_values[0])
    
    # freature importance
    shap.summary_plot(shap_values, X_val, plot_type="bar")    
    
    
# ### Jaki znacie najprostszy klasyfikator?
from sklearn.dummy import DummyClassifier

dc = DummyClassifier(strategy='uniform', random_state=32)
dc.fit(X_train,y_train)
y_proba = dc.predict_proba(X_val)
y_hat = dc.predict(X_val)
print("proba: " + str(y_proba[0:10,0]) + '\ny:     ' + str(y_hat[0:10]) + '\ny_hat: ' + str(y_val[0:10]))

np.corrcoef(y_hat, y_val)
np.corrcoef(X, X)

gini_train_val(dc, X_train, y_train, X_val, y_val)

# DummyClassfier

# Przetestujcie jaki będzie wynik działania algorytmu gdy zmienimy parametr *strategy* (oraz porównać accuracy) - podpowiedź: skorzystaj z dokumentacji funkcji

#TODO: policzyć accuracy dla baselinu (z inną strategią niż uniform) na train i validation


# - Jakieś inne proste klasyfikatory?

# ## Regresja logistyczna - czemu by nie prognozować prawdopodobieństwa za pomocą regresji liniowej?
# 
# **Przypomnienie:** uogólniony model liniowy: $y_{i}=\beta _{0}1+\beta _{1}x_{i1}+\cdots +\beta _{p}x_{ip} = x^T \beta$
# 
# - Jaki jest podstawowy problem z wykorzystaniem regresji do modelowania prawdopodobieństwa?
# - Jakie macie propozycje rozwiązania tego problemu?
# 
# $odds = \frac{P(Y=1|X)}{P(Y=0|X)} = \frac{p}{1-p}$ $\in (0,\infty)$
# 
# $\log({odds}) \in (-\infty, \infty)$
# 
# Co pozwala nam modelować powyższe równanie dzięki regresji liniowej, po przekształceniu równania, uzyskujemy prawdopodobieństwo sukcesu:
# 
# $x^T \beta = \log({\frac{p}{1-p}}) \Rightarrow p = \frac{1}{1+\exp({-x^T \beta})}$

# ![image](http://res.cloudinary.com/dyd911kmh/image/upload/f_auto,q_auto:best/v1534281070/linear_vs_logistic_regression_edxw03.png)

# https://www.datacamp.com/community/tutorials/understanding-logistic-regression-python

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=1000)

lr.fit(X_train, y_train)
y_hat = lr.predict(X_val)
print('y:     ' + str(y_hat[0:10]) + '\ny_hat: ' + str(y_val[0:10]))

gini_train_val(lr, X_train, y_train, X_val, y_val)
shapley(lr, X_train, X_val)

#TODO: policzyć accuracy dla logita z l1, l2, i bez regularyzacji na train i validation
#porównać z baselinem

lr.coef_
lr.intercept_

# ### Jak interpretować wyniki?

# jak się zmieni powyższy wynik gdy zwiększymy wartość czwartej cechy (tj. adiposity) dla pierwszej obserwacji o 1

#solution
experiment=X_val.iloc[0,:]
experiment[3]=experiment[3]+1
a = experiment.values.reshape(1,-1)
np.log(lr.predict_proba(experiment.values.reshape(1,-1))[0,1]/lr.predict_proba(experiment.values.reshape(1,-1))[0,0])

# a = lr.predict_proba(experiment.values)

# #### Dlaczego można było się przewidzieć, że taki właśnie będzie wynik?

#solution
np.log(lr.predict_proba(X_val)[0,1]/lr.predict_proba(X_val)[0,0])+lr.coef_[0,3]
# otrzymano taki sam wynik - nie trzeba było wykonywać metody predict

# TODO Jaki będzie wynik gdy wektor cech będzie miał tylko zerowe elementy?

#solution
# #### Dlaczego można było się przewidzieć, że taki właśnie będzie wynik?

#solution
1/(1+np.exp(-lr.intercept_))

# - Jakie są zalety regresji logistycznej?

# ## Drzewo decyzyjne
# - Jak wykorzystać model drzewa do predykcji klasyfikacji/regresji?
# - jakie problemy może to generować?

from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text, export_graphviz
## biblioteka poniżej może być problematyczna na Windows
#import graphviz

tree1 = DecisionTreeClassifier()

tree1.fit(X_train,y_train)
y_hat = tree1.predict(X_val)
print('y:     ' + str(y_hat[0:10]) + '\ny_hat: ' + str(y_val[0:10]))

gini_train_val(tree1, X_train, y_train, X_val, y_val)

explainer = shap.TreeExplainer(tree1, X_train)
shap_values = explainer(X_train)
tmp = shap.Explanation(shap_values[:, :, 1], data=X_train, feature_names=X_train.columns)
shap.plots.beeswarm(tmp)

shap_values = explainer(X_val)
tmp = shap.Explanation(shap_values[:, :, 1], data=X_val, feature_names=X_train.columns)
shap.plots.beeswarm(tmp)



# text_representation = export_text(tree1)
# print(text_representation)

# plt.figure(figsize=(25,20))
# splits = plot_tree(tree1, filled=True)

# opcja 2
# fig = plt.figure(figsize=(25,20))
# _ = plot_tree(tree1, 
#                    feature_names=X_train.columns,  
#                    class_names="target",
#                    filled=True)

# opcja 3
# from dtreeviz.trees import dtreeviz # remember to load the package

# viz = dtreeviz(tree1, X, y,
#                 target_name="target",
#                 feature_names=X_train.columns,
#                 class_names=list([0,1]))

# viz

tree1.get_params()

# TODO spróbujcie wytrenować model ze zmienionymi parametrami


# ## SVM
# Znalezienie równania hiperpłaszczyzny, która najlepiej dzieli nasz zbiór danych na klasy.
# 
# **Uwaga: w przypadku SVM nie modelujemy prawdopodobieństwa przynależności do danej klasy - domyślnym wyjściem jest informacja o konkretnej klasie**
# - Co jeżeli nie istnieje taka płaszczyzna?
# - Co jeżeli nasze dane nie są separowalne liniowo, tylko np. radialnie?

# ![image](https://upload.wikimedia.org/wikipedia/commons/thumb/7/72/SVM_margin.png/300px-SVM_margin.png)

# https://en.wikipedia.org/wiki/Support-vector_machine

# ### Kernel trick

# ![image](https://machine-learning-note.readthedocs.io/en/latest/_images/svm_kernel_trick.png)

# https://machine-learning-note.readthedocs.io/en/latest/algorithm/svm.html

from sklearn.svm import SVC
svm = SVC(probability=True)

svm.fit(X_train,y_train)
y_hat = svm.predict(X_val)
print('y:     ' + str(y_hat[0:10]) + '\ny_hat: ' + str(y_val[0:10]))

gini_train_val(svm, X_train, y_train, X_val, y_val)

# Jakie są wady?
# - trudno dobrać optymalne parametry
# - metoda wrażliwa na skalowanie danych
# - długo się "uczy"

# ## Naiwny Klasyfikator Bayesowski
# Jest oparty na założeniu o wzajemnej niezależności zmiennych. Często nie mają one żadnego związku z rzeczywistością i właśnie z tego powodu nazywa się je naiwnymi.

# ![image](https://wikimedia.org/api/rest_v1/media/math/render/svg/cae70e6035d9ac52c547bc1c666e372063b85324)

# Mianownik nie zależy od C więc nie będziemy go dalej analizować - skupimy się na liczniku.
# ![image](https://wikimedia.org/api/rest_v1/media/math/render/svg/2d0555690cd428cb6d6a52ea6b6391256125a45c) 

# Rekurencyjnie obliczenia będą kontynuowane. Teraz pora zrozumieć dokładniej dlaczego występuje słowo "naiwny" w nazwie metody.
#     Zakładamy bowiem że cechy $F_i$ są niezależne czyli ![image](https://wikimedia.org/api/rest_v1/media/math/render/svg/8898f2ee081f407669fdb7a4f60e390615513346)

# Ostatecznie wzór to: ![image](https://wikimedia.org/api/rest_v1/media/math/render/svg/a5978cc50b1c3d745ad304987a750aeb4a27df5b)

# https://pl.wikipedia.org/wiki/Naiwny_klasyfikator_bayesowski

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(X_train,y_train)
y_hat = nb.predict(X_val)
print('y:     ' + str(y_hat[0:10]) + '\ny_hat: ' + str(y_val[0:10]))

gini_train_val(nb, X_train, y_train, X_val, y_val)

# train an XGBoost model
import xgboost

model_xgboost = xgboost.XGBClassifier().fit(X_train, y_train)
y_hat = model_xgboost.predict(X_val)
print('y:     ' + str(y_hat[0:10]) + '\ny_hat: ' + str(y_val[0:10]))

gini_train_val(model_xgboost, X_train, y_train, X_val, y_val)

# ## RandomForest

# class sklearn.ensemble.RandomForestClassifier(n_estimators=100, *, criterion='gini', max_depth=None, 
# min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, 
# min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, 
# class_weight=None, ccp_alpha=0.0, max_samples=None)[source]

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
# rf = RandomForestClassifier(max_leaf_nodes=14, max_depth=2, n_estimators=7)
rf = RandomForestClassifier(n_estimators=500, # Ilość słabych estymatorów
                                  max_depth=2, # Maksymalna wysokość drzewa w słabym estymatorze
                                  min_samples_split = 4, # Minimalna ilość obserwacji wymagana do podziału węzła
                                  max_leaf_nodes = 14, # 
                                  max_features = 3, # Maksymalna ilość zmiennych brana pod uwagę przy podziale węzła
                                  random_state=0,
                                  n_jobs = -1)

rf.fit(X_train, y_train)

y_hat = rf.predict(X_val)

gini_train_val(rf, X_train, y_train, X_val, y_val)

def rf_feature_importances():
    
    print(rf.feature_importances_)
    importances = rf.feature_importances_
    indices = np.argsort(importances)
    features = X_train.columns
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()
    
rf_feature_importances()

explainer = shap.TreeExplainer(rf, X_train)
shap_values = explainer(X_train)
tmp = shap.Explanation(shap_values[:, :, 1], data=X_train, feature_names=X_train.columns)
shap.plots.beeswarm(tmp)

shap_values = explainer(X_val)
tmp = shap.Explanation(shap_values[:, :, 1], data=X_val, feature_names=X_train.columns)
shap.plots.beeswarm(tmp)

# TODO spróbujcie wytrenować model ze zmienionymi parametrami

# ## Lepszy sposób na podział danych na zbiory treningowe i testowe

# ### Crossvalidation
# - Czy możemy stosować CV dzieląc zbiór, tak by w zbiorze walidacyjnym pozostała tylko jedna obserwacja danych?
# - Czy sprawdzając performance modelu przez CV, możemy potem nauczyć model na całym zbiorze danych?

from sklearn.model_selection import cross_val_score

X_train_val=pd.concat((X_train,X_val))
y_train_val=np.concatenate((y_train,y_val), axis=0)
cross_val_score(lr, X_train_val, y_train_val, scoring='accuracy', cv = 10)

# ## Miary ocen jakości klasyfikatorów
# - Jakie znacie miary oceny klasyfikatorów?

# ### Accuracy
# $ACC = \frac{TP+TN}{ALL}$
# 
# Bardzo intuicyjna miara - ile obserwacji zakwalifikowaliśmy poprawnie.
# 
# - Jaki jest problem z *accuracy*?

# ### Precision & Recall
# $PRECISION = \frac{TP}{TP+FP}= \frac{TP}{\text{TOTAL PREDICTED POSITIVE}}$
# 
# $RECALL = \frac{TP}{TP+FN}$

# ### F1 Score
# $F1\_SCORE =\frac{2*PRECISION*RECALL}{PRECISION+RECALL}$

# ### ROC AUC

# ![Image](https://mathspace.pl/wp-content/uploads/2016/09/ROC-krzywa.png)

# https://mathspace.pl/matematyka/receiver-operating-characteristic-krzywa-roc-czyli-ocena-jakosci-klasyfikacji-czesc-7/

# ![image](https://cdn.analyticsvidhya.com/wp-content/uploads/2020/06/data-1.png)

# https://www.analyticsvidhya.com/blog/2020/06/auc-roc-curve-machine-learning/

# **Zadanie** - przetestować 3 modele przedstawione dziś na zajęciach i sprawdzić, który jest lepszy na podstawie wyżej wymienionych miar. Należy zastosować kroswalidację.






