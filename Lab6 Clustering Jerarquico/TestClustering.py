# -*- coding: utf-8 -*-
"""
Created on Mon May 29 16:38:20 2023

@author: Kenay
"""       
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
datos = pd.read_csv('toy01.csv')
dataCopy = pd.read_csv('toy01.csv')
clusters = []

def calculateDistancePoints(p1,p2):
    if p2 == [0,0] or p1 == [0,0]: 
        return 10000000000
    return np.sqrt(np.power(p2[0]-p1[0],2)+np.power(p2[1]-p1[1],2))    

def calculateMatrixDistances(p1,points):
    distances = []
    for i in range(len(points)):
        distances.append(calculateDistancePoints(p1, points[i]))
    return distances

def calculateClusterPointsDistances(p,clusters):
    distancesList = []
    for i in range(len(clusters)):
        dis = []
        for j in range(len(clusters[i])):
            dis.append(calculateDistancePoints(p, list(dataCopy.iloc[clusters[i][j],:])))
        distancesList.append(min(dis))
    return distancesList

def leftPoints(data,index):
    point = []
    for i in range(index,len(data)):
        point.append(list(data.loc[i]))
    return point

def poinTopoitnMin(matrix):
    minimos = list(matrix.min())
    #print(matrixDistances.iloc[:,minimos.index(min(minimos))].idxmin()) # filas
    #print(minimos.index(min(minimos))) # columnas
    return [matrixDistances.min().min(),matrix.iloc[:,minimos.index(min(minimos))].idxmin(),minimos.index(min(minimos))]
    
def validateCluster(index):
    val = [0,0]
    for i in range(len(clusters)):
        if index[0] in clusters[i]:
            val[0] = i
        if index[1] in clusters[i]:
            val[1] = i 
    return val
    
def promCluster():
    proms = []
    for i in range(len(clusters)):
        x = 0
        y = 0
        for j in range(len(clusters[i])):
            x = x + datos.iloc[clusters[i][j],0]
            y = y + datos.iloc[clusters[i][j],1]
        proms.append([x/j,y/j])
    return proms


matrixDistances = pd.DataFrame([[0]*len(datos)]*len(datos))
if len(datos) > 0 :
    for i in range(len(datos)-1):
        matrixDistances.iloc[:,i]= [0]*(i+1) + calculateMatrixDistances(list(dataCopy.iloc[i,:]), leftPoints(dataCopy,i+1))

matrixDistances[matrixDistances == 0 ] = np.NaN


nivel = int(input('Da un nivel (entero): '))
k =0
cont = 0
while k < nivel and cont < len(dataCopy) :
    k = k + 1 
    cont = cont + 1
    candi = poinTopoitnMin(matrixDistances)
    matrixDistances.iloc[candi[1],candi[2]] = 100000000
    if len(clusters) == 0: 
        clusters.append([candi[1],candi[2]])
        dataCopy.iloc[candi[1],:] = [0,0]
        dataCopy.iloc[candi[2],:] = [0,0]
    else:
        validate = validateCluster([candi[1],candi[2]])
        if validate == [0,0]:
            clusters.append([candi[1],candi[2]])
            dataCopy.iloc[candi[1],:] = [0,0]
            dataCopy.iloc[candi[2],:] = [0,0]
        else:
            if validate[0] != validate[1]:
                if validate[0] == 0:
                    clusters[validate[1]].append(candi[1])
                    dataCopy.iloc[candi[1],:] = [0,0]
                elif validate[1] == 0 :
                    clusters[validate[0]].append(candi[2])
                    dataCopy.iloc[candi[2],:] = [0,0]
                else:
                    union = clusters[validate[0]-1] + clusters[validate[1]-1]
                    clusters.pop(validate[0]-1)
                    if validate[1] > validate[0]:
                        clusters.pop(validate[1]-2)
                    else:
                        clusters.pop(validate[1]-1)
                    clusters.append(union)

while k < nivel :
    if len(clusters) == 1:
        break
    k= k +1
    promedios = promCluster()
    matrixDistances = pd.DataFrame([[0]*len(promedios)]*len(promedios))
    for i in range(len(promedios)-1):
        matrixDistances.iloc[:,i]= [0]*(i+1) + calculateMatrixDistances(promedios[i],promedios[(i+1):])
    matrixDistances[matrixDistances == 0 ] = np.NaN
    candi = poinTopoitnMin(matrixDistances)
    print(candi)
    matrixDistances.iloc[candi[1],candi[2]] = 100000000
    union = clusters[validate[0]-1] + clusters[validate[1]-1]
    clusters.pop(validate[0]-1)
    if validate[1] > validate[0]:
        clusters.pop(validate[1]-2)
    else:
        clusters.pop(validate[1]-1)
    clusters.append(union)
    
colors = ['r','g','b']
markers = ['.','o','v','<','>','h']
option = []

if len(clusters)>= 2:
    no = []
    for i in range(len(datos)):
        contador = 0
        for j in range(len(clusters)):
            if i not in (clusters[j]) :
                contador = contador+1
        if contador == len(clusters):
            no.append(i)
    no = list(set(no))
    print(no)
    x = []
    y = []
    for h in range(len(no)):
        x.append(datos.iloc[no[h],0])
        y.append(datos.iloc[no[h],1])
    color = colors[np.random.randint(0,len(colors))]
    marker = markers[np.random.randint(0,len(markers))]
    cad = color+marker
    while cad in option:
        color = colors[np.random.randint(0,len(colors))]
        marker = markers[np.random.randint(0,len(markers))]
        cad = color+marker
    option.append(cad)
    plt.scatter(x, y, color='black',marker='s',label='without')

for i in range(len(clusters)):
    x = []
    y = []
    for j in range(len(clusters[i])):
        x.append(datos.iloc[clusters[i][j],0])
        y.append(datos.iloc[clusters[i][j],1])
    color = colors[np.random.randint(0,len(colors))]
    marker = markers[np.random.randint(0,len(markers))]
    cad = color+marker
    while cad in option:
        color = colors[np.random.randint(0,len(colors))]
        marker = markers[np.random.randint(0,len(markers))]
        cad = color+marker
    option.append(cad)
    plt.scatter(x, y, color=color,marker=marker,label='cluster '+ str(i+1))
    plt.legend()
