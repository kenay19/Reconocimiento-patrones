# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 18:45:47 2023

@author: Kenay
"""
import pandas as pd
import numpy as np 
class RegresionLogistic:
    
    def fit(self,x,y):
        '''
        Se encarga de preparar los datos necesarios para entrenar el 
        modelo

        Parameters
        ----------
        x : dataframe
            Es el conjunto de n instancias con m atributos.
        y : dataframe
            Son las etiquetas de cada instancia de x.

        Returns
        -------
        None.

        '''
        self.thethas= []
        self.elementos = x
        self.nArrayThetas(y)
        for i in range(len(self.etiquetas)):
            self.realFit(self.etiquetas[i])
            
       
        
    
    def nArrayThetas(self,y):
        '''
        Hace la 'traduccion' de las etiquetas a unos y ceros

        Parameters
        ----------
        y : dataframe
            Contiene las etiquedas de las instancias dadas por x.

        Returns
        -------
        None.

        '''
        clases = list(set(y))
        self.etiquetas = []
        self.clases = clases
        x = len(self.clases)
        if x == 2:
            x = 1
        for i in range(x):
            nlabel = []
            for j in range(len(y)):
                if y.iloc[j] == clases[i]:
                    nlabel.append(1)
                else: 
                    nlabel.append(0)
            nlabel = pd.Series(nlabel,index=y.index.values)
            self.etiquetas.append(nlabel)
            
    def realFit(self,etiquetas):
        '''
        calcula el conjunto de thetas optimo para las etiquetas
        de unos y ceros

        Parameters
        ----------
        etiquetas : list
            Contiene las etiquetas de unos y ceros.

        Returns
        -------
        None.

        '''
        thetha = [1] * len(self.elementos.columns.values)
        cont  = 0 
        count = 0
        while count < 1000:
            count = count + 1
            propose = [0] * len(thetha)
            for i in range(len(thetha)):
                suma = 0
                for j in range(len(self.elementos)):
                    suma = suma + (etiquetas.loc[etiquetas.index.values[j]] - self.logisticFunction(thetha, self.elementos.loc[self.elementos.index.values[j]]))  * self.elementos.loc[self.elementos.index.values[j]].loc[self.elementos.columns[i]]              
                propose[i] = thetha[i] + 0.02*(suma)
            if self.test(thetha, propose) <= 2.5:
                cont = cont  + 1
                if cont == 5 :
                    break
            else:
                cont = 0
            thetha = propose
        self.thethas.append(thetha)
    
    def test(self,x,y):
        '''
        Da el parametro para definir si se para el entrenamiento

        Parameters
        ----------
        x : list
            thethas anteriores
        y : list
            thethas actualizados.

        Returns
        -------
        res : float
            DESCRIPTION.

        '''
        res = 0
        for i in range(len(x)):
            res = res + abs(y[i]-x[i])
        return res
            

    def logisticFunction(self,thetha,x):
        '''
        Calcula la funcion logistica para los thethas e instancia dada

        Parameters
        ----------
        thetha :  list
            thethas para las etiqueta dad.
        x : list
            Instancia a la que se le va a aplicar la funcion 
            logistica.

        Returns
        -------
        float
            Valor de la funcion logistica.

        '''
        suma = 0
        for i in range(len(thetha)):
            suma = suma + thetha[i] * x[i]
        return (1/(1+np.exp(-suma)))
    
    
    def predict(self,x):
        '''
        Hace las predicciones para las instancias dadas

        Parameters
        ----------
        x : dataframe
            son las instancias a predecirs.

        Returns
        -------
        pd.Series
            Contiene las clases predichas para las instancias dadas.

        '''
        predicts = []
        for i in range(len(self.etiquetas)):
            labels = []
            for j in range(x.shape[0]):
                if self.logisticFunction(self.thethas[i], x.loc[x.index.values[j]]) >= 0.5:
                    labels.append(1)
                else:
                    labels.append(0)
            predicts.append(labels)
        indices = self.indexMatches(predicts)
        return pd.Series(self.calculateLabel(indices, x),index=x.index.values)
        
    def indexMatches(self,lista):
        '''
        Si el problema dado es multiclase recopila la "clase" 
        con la que fue clasificado

        Parameters
        ----------
        lista : list
            lista de listas con las etiquetas predichas por 
            cada clase.

        Returns
        -------
        indices : list
            lista de listas que contiene las etiquetas
            predichas con la clase en el indice indicado.

        '''
        matches = []
        for j in range(len(lista[0])):
            index = []
            for i in range(len(lista)):
        
                index.append(lista[i][j])
            matches.append(index)
        indices = []
        for i in range(len(matches)):
            indices.append([indice for indice, dato in enumerate(matches[i]) if dato == 1])
        return indices
    
    def distance(self,thetha,x):
        '''
        calcula la distancia entre la instancia x con la recta formada
        por los parametros thetha 

        Parameters
        ----------
        thetha : list
            parametros thetha.
        x : list
            instancia de x.

        Returns
        -------
        suma : float
            distancia entre los parametros thetha y x.

        '''
        suma = 0
        for i in range(len(thetha)):
            suma = suma + thetha[i] * x[i]
        return suma
    
    def calculateDistance(self,indices,x):
        '''
        de los coincidencias de indices toma la distancia mas alta
        y retorna el indice de dicha distancia

        Parameters
        ----------
        indices : list
            indices de coincidencias.
        x : list
            instancia a calcular la etiqueta.

        Returns
        -------
        str
            nombre de la clase asignada.

        '''
        if len(indices) != 0:
            dic = {}
            for i in range(len(indices)):
                dic[i] = self.distance(self.thethas[indices[i]], x)
            values =  list(dic.values())
            maxi = values.index(max(values))
            keys = list(dic.keys())
            index = keys[maxi]
            return self.clases[index]
        else:
            return self.clases[np.random.randint(0,len(self.clases))]
    
    def calculateLabel(self,indices,x):
        '''
        Calcula la eitqueta para la instancia x

        Parameters
        ----------
        indices :  list
            lista de coincidencias de prediccion.
        x : list
            instancia a calcular etiqueta.

        Returns
        -------
        labels : list
            etiquetas predichas.

        '''
        labels = []
        for i in range(len(indices)):
            if len(indices[i]) == 1 :
                labels.append(self.clases[indices[i][0]])
            else:
                labels.append(self.calculateDistance(indices[i],x.loc[x.index.values[i]]))
        return labels
    
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
        
            