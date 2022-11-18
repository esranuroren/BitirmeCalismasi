# -*- coding: utf-8 -*-
"""
Created on Tue May 24 15:42:49 2022

@author: esrar
"""


#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2.veri onisleme
#2.1.veri yukleme
veriler = pd.read_csv('turkey-weekly.csv')
#pd.read_csv("turkey-weekly.csv")
#test
print(veriler)
cases = veriler[['weekly_count']]
print(cases)



x = veriler.iloc[:,3:5].values #bağımsız değişkenler
y = veriler.iloc[:,2:3].values #bağımlı değişken
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
