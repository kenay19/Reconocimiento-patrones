#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UNIVERSIDAD AUTÓNOMA DEL ESTADO DE MÉXICO
CU UAEM ZUMPANGO

Ingeniería en Computación

Reconocimiento de Patrones

Alumno: Kevin Omar Lazaro Ortega

Profesor: Dr. Asdrúbal López Chau

DESCRIPCIÓN: Test of Naive Bayes classification method


Created on Thu Mar  2 09:41:08 2023

@author: asdruballopezchau
"""
from MyNaiveBayes import NaiveBayesReal as NBR, NaiveBayesCategorical as NBC
import pandas as pd
from sklearn.model_selection import train_test_split as split


    
def Categorical():
    X_train, X_test, y_train, y_test = split(df.iloc[:, 0:-1], df.iloc[:, -1], test_size=0.2)
    mod = NBC()
    if mod.fit(X_train,y_train) == "No es Categorico":
        print('los datos no son categoricos')
        return     
    Y_predict  = mod.predict(X_test)
    if str(type(Y_predict))!="<class 'pandas.core.series.Series'>":
        if Y_predict == "No es Categorico": 
            print('los datos no son categoricos')
            return  
    print("============== Xtrain ===============")
    print(X_train)
    print("============== Ytrain ===============")
    print(y_train)
    print("============== Xtest ===============")
    print(X_test)
    print("============== Ytest ===============")
    print(y_test)
    print("============== Ypredict ===============")
    print(Y_predict)
    mod.CalculateAccuracy(Y_predict, y_test)
    
    
def Numerico():
    X_train, X_test, y_train, y_test = split(df.iloc[:, 0:-1], df.iloc[:, -1], test_size=0.2)
    mod = NBR()
    if mod.fit(X_train,y_train) == "No es Numerico":
        print('los datos no son numericos')
        return     
    Y_predict  = mod.predict(X_test) 
    if str(type(Y_predict))!="<class 'pandas.core.series.Series'>":
        if Y_predict == " No es Numerico": 
            print('los datos no son numericos')
            return
    print("============== Xtrain ===============")
    print(X_train)
    print("============== Ytrain ===============")
    print(y_train)
    print("============== Xtest ===============")
    print(X_test)
    print("============== Ytest ===============")
    print(y_test)
    print("============== Ypredict ===============")
    print(Y_predict)
    mod.CalculateAccuracy(Y_predict, y_test)


def ChooseFile():
    options = ['carDataset.csv','naive.csv','Raisin.csv','iris.data']
    print("1) carDataset")
    print("2) naive")
    print("3) Raisin")
    print("4) Iris")
    option = int(input('Seleccione una opcion: '))
    global df 
    df = pd.read_csv(options[option-1])
    
def ChooseOption():
    option = "S"
    print('Escoja un archivo')
    ChooseFile()
    while option == "S" or option== "s":
        print("1) Atributos Numericos")
        print("2) Atributos Categoricos")
        option = int(input("Seleccione una opcion: ")) 
        if option == 1:
            Numerico()
        elif option ==2:
            
            Categorical()
        option = input('Desea camiar de archivo? S/n: ')
        if option == "S" or option== "s":
            ChooseFile()
        option = input("Desea Continuar? S/N: ")
    
ChooseOption()
