# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 16:02:24 2023

@author: Kenay
"""
from RegresionLogistic import RegresionLogistic 
import pandas as pd
from sklearn.model_selection import train_test_split as split




def ChooseFile():
    options = ['iris.data','Raisin.csv','wine.data']
    print("1) Iris")
    print("2) Raisin")
    print("3) Wine")
    option = int(input('Seleccione una opcion: ')) 
    df = pd.read_csv(options[option-1])
    if option == 3:
        df.insert(df.shape[1],"clases",df.iloc[:,0])
        df.drop(['a'],axis=1)
    return df
opcion = "S" 
while opcion == "S" or opcion == "s":       

    df =ChooseFile()
    X_train, X_test, y_train, y_test = split(df.iloc[:, 0:-1], df.iloc[:, -1], test_size=0.2)
    RL = RegresionLogistic()
    X_train.insert(0,'insert',1)
    X_test.insert(0,'insert',1)
    print("============== Entrenamiento ==============")
    print("----------------- X_train ---------------------")
    print(X_train)
    print("----------------- Y_train ---------------------")
    print(y_train)
    print("---------------- Clases --------------------")
    print(list(set(df.iloc[:,-1])))
    RL.fit(X_train, y_train)
    print("---------------- thethas -------------------")
    if len(list(set(df.iloc[:,-1]))) == 2:
        index = [1]
    else:
        index =list(set(df.iloc[:,-1]))
    print(pd.DataFrame(RL.thethas,index=index))
    print("================= Predicciones ==============")
    print("----------------- X_test ---------------------")
    print(X_test)
    print("----------------- Y_test ---------------------")
    print(y_test)
    y_predict = RL.predict(X_test )
    print("----------------- Y_predict ---------------------")
    print(y_predict)
    RL.CalculateAccuracy(y_predict, y_test)
    opcion = input('Desea Continuar (S,n): ')