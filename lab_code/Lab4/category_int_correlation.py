#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
np.random.seed(1)


numeric=np.random.normal(size=100)

category=np.random.choice(['Ania', 'Kasia', 'Tomek'], size=100)

dict1={"Ania":1, "Kasia": 2, "Tomek":3}
np.corrcoef(pd.Series(category).map(dict1), numeric)[0,1]

print("pearson %1.4f" % pd.Series(numeric).corr(pd.Series(category).map(dict1), method="pearson"))
print("spearman %1.4f" % pd.Series(numeric).corr(pd.Series(category).map(dict1), method="spearman"))

#przeskalowanie wartości
dict1={"Ania":10, "Kasia": 20, "Tomek":30}
np.corrcoef(pd.Series(category).map(dict1), numeric)[0,1]

print("pearson %1.4f" % pd.Series(numeric).corr(pd.Series(category).map(dict1), method="pearson"))
print("spearman %1.4f" % pd.Series(numeric).corr(pd.Series(category).map(dict1), method="spearman"))

# zamiana kolejności na odwrotną  -- analogia do zmiennej binarnej
dict1={"Ania":3, "Kasia": 2, "Tomek":1}
np.corrcoef(pd.Series(category).map(dict1), numeric)[0,1]

print("pearson %1.4f" % pd.Series(numeric).corr(pd.Series(category).map(dict1), method="pearson"))
print("spearman %1.4f" % pd.Series(numeric).corr(pd.Series(category).map(dict1), method="spearman"))

# zamiana kolejności
dict1={"Ania":3, "Kasia": 1, "Tomek":2}
np.corrcoef(pd.Series(category).map(dict1), numeric)[0,1]

print("pearson %1.4f" % pd.Series(numeric).corr(pd.Series(category).map(dict1), method="pearson"))
print("spearman %1.4f" % pd.Series(numeric).corr(pd.Series(category).map(dict1), method="spearman"))

# zamiana kolejności
dict1={"Ania":2, "Kasia": 1, "Tomek":3}
np.corrcoef(pd.Series(category).map(dict1), numeric)[0,1]

print("pearson %1.4f" % pd.Series(numeric).corr(pd.Series(category).map(dict1), method="pearson"))
print("spearman %1.4f" % pd.Series(numeric).corr(pd.Series(category).map(dict1), method="spearman"))

# zamiana kolejności
dict1={"Ania":2, "Kasia": 3, "Tomek":1}
np.corrcoef(pd.Series(category).map(dict1), numeric)[0,1]

print("pearson %1.4f" % pd.Series(numeric).corr(pd.Series(category).map(dict1), method="pearson"))
print("spearman %1.4f" % pd.Series(numeric).corr(pd.Series(category).map(dict1), method="spearman"))

