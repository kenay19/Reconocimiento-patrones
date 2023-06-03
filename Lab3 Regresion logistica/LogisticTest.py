# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 16:02:24 2023

@author: Kenay
"""
from RegresionLogistic import RegresionLogistic 
import pandas as pd
import numpy as np
import matplotlib.pyplot  as plt
from sklearn.model_selection import train_test_split as split
from sklearn import svm


def choosemin(x1,x2):
    if len(x1) == 0:
        xmax = max(x2)
        xmin = min(x2)
    elif len(x2) ==0:
        xmax = max(x1)
        xmin = min(x1)
    else:
        if max(x1) > max(x2):
            xmax = max(x1)
        else:
            xmax = max(x2)
            
        if min(x1) < min(x2):
            xmin = min(x1)
        else:
            xmin = min(x2)
    return [xmin,xmax]
        

def graficarRL(df,thepro,thepre,accuracy):
    x1true = []
    y1true = []
    x1false = []
    y1false = []
    x2true = []
    y2true = []
    x2false = []
    y2false = []
    
    atributos = df.columns.values
    fig,ax = plt.subplots()
    for j in range(df.shape[1]-3):
        for i in range(df.shape[0]):
            if df.iloc[i,-2] == df_filtros[0]:
               if df.iloc[i,-2] == df.iloc[i,-1]:
                   x1true.append(df.iloc[i,j])
                   y1true.append(df.iloc[i,j+1])
               else:
                   x1false.append(df.iloc[i,j])
                   y1false.append(df.iloc[i,j+1])
            else:
                if df.iloc[i,-2] == df.iloc[i,-1]:
                    x2true.append(df.iloc[i,j])
                    y2true.append(df.iloc[i,j+1])
                else:
                    x2false.append(df.iloc[i,j])
                    y2false.append(df.iloc[i,j+1])
    plt.scatter(x1true,y1true,color='g',marker='x',label=df_filtros[0] +' correcto')
    plt.scatter(x1false,y1false,color='r',marker='x',label=df_filtros[0] +' incorrecto')
    plt.scatter(x2true,y2true,color='g',marker='s',label=df_filtros[1] +' correcto')
    plt.scatter(x2false,y2false,color='r',marker='s',label=df_filtros[1] +' incorrecto')
    limits = choosemin(x1true+x1false, x2true+x2false)
    x = np.linspace(limits[0], limits[1],1000)
    ypro = x + 1
    ypre = (-(x*thepre[0][1])/thepre[0][2])-(thepre[0][0]/thepre[0][2])
    plt.plot(x,ypro,label='sin ajustar')
    plt.plot(x,ypre,label='ajustado')
    plt.xlabel(atributos[0])
    plt.ylabel(atributos[1])
    plt.title('Accurracy: '+str(accuracy*100)+'% (RL)')

def graficarSVM(df,clf,accuracy):
       x1true = []
       y1true = []
       x1false = []
       y1false = []
       x2true = []
       y2true = []
       x2false = []
       y2false = []
       atributos = df.columns.values
       fig,ax = plt.subplots()
       for j in range(df.shape[1]-3):
              for i in range(df.shape[0]):
                  if df.iloc[i,-2] == df_filtros[0]:
                     if df.iloc[i,-2] == df.iloc[i,-1]:
                         x1true.append(df.iloc[i,j])
                         y1true.append(df.iloc[i,j+1])
                     else:
                         x1false.append(df.iloc[i,j])
                         y1false.append(df.iloc[i,j+1])
                  else:
                      if df.iloc[i,-2] == df.iloc[i,-1]:
                          x2true.append(df.iloc[i,j])
                          y2true.append(df.iloc[i,j+1])
                      else:
                          x2false.append(df.iloc[i,j])
                          y2false.append(df.iloc[i,j+1])
       plt.scatter(x1true,y1true,color='g',marker='x',label=df_filtros[0] +' correcto')
       plt.scatter(x1false,y1false,color='r',marker='x',label=df_filtros[0] +' incorrecto')
       plt.scatter(x2true,y2true,color='g',marker='s',label=df_filtros[1] +' correcto')
       plt.scatter(x2false,y2false,color='r',marker='s',label=df_filtros[1] +' incorrecto')
       w = clf.coef_[0]
       a = -w[0] / w[1]
       xx = np.linspace(min(df.iloc[:,0]), max(df.iloc[:,0]))
       yy = a * xx - (clf.intercept_[0]) / w[1]
       plt.plot(xx,yy,label='frontera de desicion')
       plt.xlabel(atributos[0])
       plt.ylabel(atributos[1])
       plt.title('Accurracy: '+str(accuracy*100)+'% (SVM)')
     
         
dfe = pd.read_csv('iris.data')

clases = list(set(dfe.iloc[:,-1]))
df_filtros = ['','']
while df_filtros[0] == df_filtros[1]:
    df_filtros = [ clases[np.random.randint(0,3)] for i in range(2)]
print(df_filtros)
isin = dfe.iloc[:,-1] == df_filtros[0]  
isin2 = dfe.iloc[:,-1]== df_filtros[1]
index = [i for i in range(len(isin2)) if isin2[i]]
for i in range(len(index)):
    isin[index[i]] = True 
dfe = dfe[isin]
RL = RegresionLogistic()
clf = svm.SVC(kernel='linear')
atributos = dfe.columns.values[0:-1]
for i in range(len(atributos)-1):
    for j in range(i+1,len(atributos)):
        df = pd.DataFrame([dfe.iloc[:,i],dfe.iloc[:,j],dfe.iloc[:,-1]])
        df = df.transpose()
        X_train, X_test, y_train, y_test = split(df.iloc[:, :-1], df.iloc[:, -1], test_size=0.2)
        clf.fit(X_train,y_train)
        ypre = clf.predict(X_test)
        accSVM = RL.CalculateAccuracy(pd.Series(ypre,index=y_test.index.values), y_test)
        X_train.insert(0,'insert',1)
        X_test.insert(0,'insert',1)
        RL.fit(X_train, y_train)
        y_predict = RL.predict(X_test )
        accRL=RL.CalculateAccuracy(y_predict, y_test)
        graficarRL(pd.DataFrame([X_test.iloc[:,1],X_test.iloc[:,2],y_predict,y_test]).transpose(),[1]*len(X_train.columns.values),RL.thethas,accRL)
        graficarSVM(pd.DataFrame([X_test.iloc[:,1],X_test.iloc[:,2],pd.Series(ypre,index=y_test.index.values),y_test]).transpose(), clf, accSVM)