#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UNIVERSIDAD AUTÓNOMA DEL ESTADO DE MÉXICO
CU UAEM ZUMPANGO

Ingeniería en Computación

Reconocimiento de Patrones

Alumno: Kevin Omar Lazaro Ortega 

Profesor: Dr. Asdrúbal López Chau

DESCRIPCIÓN: Own implementation of Naive Bayes classification method  
    
Created on Thu Mar  2 09:58:20 2023

@author: asdruballopezchau
"""
import pandas as pd 
import numpy as np
class NaiveBayesReal:
    
    def fit(self, X, y):
        '''
        Entrena el modelo Naibe Bayes con datos numericos
        Parameters
        ----------
        X : DataFrame
            Son todos los atributos que componen una instancias.
        y : DataFrame
            Son las etiquetas de cada instancia.

        Returns
        -------
        None.

        '''
        msg = self.compruebaCat(X) 
        if msg == "No es Numerico":
            return msg
        self.x = X
        self.y = y 
        self.ProbsPriori(self.y)
        self.CalculateMm()
        
    def compruebaCat(self,X):
         '''
         Comprueba si los datos son string

         Parameters
         ----------
         X : DATAFRAME
             Son todas las instancias con cada atributo sin contar
             las etiquetas.

         Returns
         -------
         str
             Mensaje que indica si hay un dato numerico 

         '''
         test = []
         for j in range(X.shape[1]):
             for i in range(X.shape[0]):              
                 if str(type(X.iloc[i,j])) != "<class 'numpy.float64'>" and str(type(X.iloc[i,j])) != "<class 'numpy.int64'>" :
                     test.append(X.iloc[i,j])
         if(len(test) >0):
             return  "No es Numerico"
         return "Es Numerico"
    
    def predict(self, X):
        '''
        Clasifica las instancias dadas en base al entrenamiento previo
        Parameters
        ----------
        X : DataFrame
            Son las n instancias a clasificar con todos los atributos necesarios.

        Returns
        -------
        Pandas.Series || str
            Puede devolver un mensaje indicando que no es un atributo numerico
            O devolver una series de pandas en la que vienen las etiquetas predichas.

        '''
        msg = self.compruebaCat(X) 
        if msg == "No es Categorico":
            return msg
        tamano = X.shape
        etiquetas = []
        for i in range(tamano[0]):
            canti = []
            for col in range(len(self.clases)):
                mult = self.Probs_priori[self.clases[col]]
                for j in range(tamano[1]):
                    mult = mult *  self.probCond(X.iloc[i,j], self.statistic['mus'][j][self.clases[col]], self.statistic['sigmas'] [j][self.clases[col]]) 
                canti.append(mult)
            etiquetas.append(self.clases[canti.index(max(canti))])
        return pd.Series(etiquetas)

    def probCond(self,x,mu,sigma):
        '''
        Calcula la probabilidad condicionada
    Parameters
        ----------
        x : float
            Atributo al que se le calculara la probabilidad condicionada
        mu : float
            Media de todos los datos pertenecientes a dicho atributo .
        sigma : float
            Desviacione estandar de todos los datos pertenecientes a dicho atributo.

        Returns
        -------
        float
            Probabilidad condicionada calculada.

        '''
        return (1/(sigma*np.sqrt(2*np.pi)))*(np.exp(-np.power(x-mu,2)/(2*np.power(sigma,2))))
    
    def ProbsPriori(self,y):
        '''
        Calcula las probabilidad a priori de las clases del conjunto de datos

        Parameters
        ----------
        y : DataFrame
            Son todas las etiquetas dentro del conjunto de datos.

        Returns
        -------
        None.

        '''
        self.clases = list(set(y))
        tamano = y.shape[0]
        probs = []
        for i in range(len(self.clases)):
            probs.append(sum(y.iloc[:] ==self.clases[i])/tamano)
        self.Probs_priori = pd.Series(probs,index = self.clases)
        
    def CalculateMm(self):
        '''
        Calcula las medias y dsitribuciones estandar de cada atributo dentro del conjunto de datos

        Returns
        -------
        None.
        '''
        atributos = []
        for x in range(len(self.clases)):
            index = self.GetIndex(self.clases[x],list(self.y))
            at = []
            for ind in range(len(index)):
                at.append(self.x.iloc[ind,:])
            atributos.append(at)
    
        mus = []
        sigmas = []
        for x in range(len(self.x.columns.values)):
            mu = {}
            sigma = {}

            for atr in range(len(atributos)):   
                a = []
                for y in range(len(atributos[atr])):
                    a.append(atributos[atr][y].loc[self.x.columns.values[x]] )
                mu[self.clases[atr]] = np.mean(a)
                sigma[self.clases[atr]] = np.std(a)
            mus.append(mu)
            sigmas.append(sigma)
        
        self.statistic = pd.Series([mus,sigmas],index=['mus','sigmas'])
    def GetIndex(self,value,lista):
        '''
        retorna los indices de un elemento repetido dentro de una isla

        Parameters
        ----------
        value : TYPE
            DESCRIPTION.
        lista : TYPE
            DESCRIPTION.

        Returns
        -------
        index : TYPE
            DESCRIPTION.

        '''
        index = []
        for i in range(len(lista)):
            if value == lista[i]:
                index.append(i)
        return index
        
    def CalculateAccuracy(self,Ypredict,Ytest):
        '''
        Calcula la exactitud del conjunto de datos1

        Parameters
        ----------
        Ypredict : DataFrame
            Son las etiquetas predichas.
        Ytest : DataFrame
            Son las etiquetas del conjutno de datos.

        Returns
        -------
        None.

        '''
        size = Ytest.shape[0]
        Ypredict = list(Ypredict)
        Ytest = list(Ytest)
        accuracy = sum([1 for x in Ytest if x in Ypredict])/size
        print("La Exactitud es: ",accuracy*100,"%")
         
class NaiveBayesCategorical:
    
    def fit(self, X, y):
        '''
        Entrana al modelo con datos categoricos

        Parameters
        ----------
        X : DataFrame
            Son todas las instancias incluyendo todos sus atributos sin contar las
            clases.
        y : DataFrame
            Son las etiquetas de cada instancia.

        Returns
        -------
        None.

        '''
        self.x =X
        self.y = y
        msg = self.compruebaCat(X) 
        if msg == "No es Categorico":
            return msg
        self.ProbsPriori(y)
        self.ProbsCond()
        print(self.Probs_Cond)
    
    
    def compruebaCat(self,X):
        '''
        Comprueba si los datos son string

        Parameters
        ----------
        X : DATAFRAME
            Son todas las instancias con cada atributo sin contar
            las etiquetas.

        Returns
        -------
        str
            Mensaje que indica si hay un dato numerico 

        '''
        test = []
        for j in range(X.shape[1]):
            for i in range(X.shape[0]):
                if str(type(X.iloc[i,j])) != "<class 'str'>":
                    test.append(X.iloc[i,j])
        if(len(test) >0):
            return  "No es Categorico"
        return "Es categorico"
    
    def predict(self, X):
        '''
        Hace predicciones con los datos dados en base al entrenamiento previo
        Parameters
        ----------
        X : DataFrame
            Son las n instancias a predecirs.

        Returns
        -------
        Pandas.Series
            Son las etiquetas predichas por el algoritmo.

        '''
        tamano = X.shape
        etiquetas = []
        msg = self.compruebaCat(X) 
        if msg == "No es Categorico":
            return msg
        for i in range(tamano[0]):
            canti = []
            for col in range(len(self.clases)):
                mult = self.Probs_priori[self.clases[col]]
                for j in range(tamano[1]):
                    label = self.clases[col] +" | "+ X.iloc[i,j]
                    mult = mult * self.Probs_Cond.loc[self.x.columns.values[j]][label]
                canti.append(mult)
            etiquetas.append(self.clases[canti.index(max(canti))])
        return pd.Series(etiquetas)
    
    def ProbsPriori(self,y):
        '''
        Calcula las probabilidad a priori de las clases del conjunto de datos

        Parameters
        ----------
        y : DataFrame
            Son todas las etiquetas dentro del conjunto de datos.

        Returns
        -------
        None.

        '''
        self.clases = list(set(y))
        tamano = y.shape[0]
        probs = []
        for i in range(len(self.clases)):
            probs.append(sum(y.iloc[:] ==self.clases[i])/tamano)
        self.Probs_priori = pd.Series(probs,index = self.clases)
        
    def ProbsCond(self):
        '''
        Calcula las probabilidades condicionales de cada nivel por atributo y clase

        Returns
        -------
        None.

        '''
        variables = self.x.columns.values
        self.Probs_Cond = pd.Series(dtype=object)
        for h in range(len(variables)):
            variable = list(set(self.x[variables[h]]))
            inst = {}
            for i in range(len(self.clases)):
                for j in range(len(variable)):
                    a = list(self.x[variables[h]] == variable[j])
                    print(a)
                    b = list(self.y == self.clases[i])
                    print(b)
                    count = 0 
                    for x  in range(self.y.shape[0]):
                        if (a[x] and b[x]):
                            count = count + 1 
                    label = self.clases[i]+" | "+variable[j]
                    inst[label] = count/sum(self.y == self.clases[i])
            self.Probs_Cond.loc[variables[h]] = inst
            
    def CalculateAccuracy(self,Ypredict,Ytest):
        '''
        Calcula la exactitud del conjunto de datos1

        Parameters
        ----------
        Ypredict : DataFrame
            Son las etiquetas predichas.
        Ytest : DataFrame
            Son las etiquetas del conjutno de datos.

        Returns
        -------
        None.

        '''
        size = Ytest.shape[0]
        Ypredict = list(Ypredict)
        Ytest = list(Ytest)
        accuracy = sum([1 for x in Ytest if x in Ypredict])/size
        print("La Exactitud es: ",accuracy*100,"%")
        
        