# -*- coding: utf-8 -*-
"""
Created on Wed May 25 10:06:19 2022

@author: esrar
"""

#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2.veri onisleme
#2.1.veri yukleme
veriler = pd.read_csv('belgium.csv')
#pd.read_csv("belgium.csv")
#test
print(veriler)
cases = veriler[['cases']]
print(cases)



x = veriler.iloc[:,1:4].values #bağımsız değişkenler
y = veriler.iloc[:,4:5].values #bağımlı değişken
print(y)



#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)



#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()




from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(x_train,y_train)



y_pred = logr.predict(x_test)
print(y_pred)
print(y_test)



plt.plot(x_train,y_train)
plt.plot(x_test,logr.predict(x_test))



plt.title("cases")
plt.xlabel("tarih??")
plt.ylabel("vaka sayıları")
