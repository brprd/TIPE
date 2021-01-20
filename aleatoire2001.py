import random
import csv
import numpy as np

def deplacement(X,V):
    Vect=[X]
    U=X
    for i in range(1,100):
        U=U+V[i]
        Vect.append(np.array([U[0],U[1]]))
    return Vect
def vitesse(V):

    Vect=[V[0]]
    U=V[0]
    for i in range(1,100):

        U=U+np.array([random.choice([-1,1])*V[i][0],random.choice([-1,1])*V[i][1]])
        Vect.append(np.array([U[0],U[1]]))
    return Vect
def acceleration(V):
    U=[V]
    x=random.random()
    for i in range(1,100):
        V=V+np.array([0.00005*random.choice([-x,x]) ,0.00005*random.choice([-x,x])])
        U.append(V)
    return U
def heure(j):
    if j<10:
        return "00:0"+str(j)+":00"
    elif j<60:
        return "00:"+str(j)+":00"
    elif j<70:
        return "01:0"+str(j-60)+":00"
    else:
        return "01:"+str(j-60)+":00"
def mismatch():
    AIS=[]
    n=10
    g=random.random()
    for i in range(n):
        A=acceleration(np.array([1*random.random(),1*random.random()]))
        U=vitesse(A)
        X=np.array([10*random.uniform(10-g,10+g),10*random.uniform(10-g,10+g)])
        V=deplacement(X,U)
        for j in range(len(V)):
            AIS.append(["2018-12-31T"+heure(j),3337640+i,V[j][0],V[j][1],U[j][0],U[j][1]])
    return AIS

AIS=mismatch()
with open('D:/Documents/AIS_2018_12_20/bbd.csv', 'w', newline='') as csvfile:
    c = csv.writer(csvfile, delimiter=',')
    for i in range(len(AIS)):
        print(AIS[i])
        c.writerow(AIS[i])