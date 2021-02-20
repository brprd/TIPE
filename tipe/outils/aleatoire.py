import random
import csv
import numpy as np

output_file="D:/brieu/Documents/MP/TIPE/bdd_aleatoire.csv"

def deplacement(X,V):
    Vect=[X]
    U=X
    for i in range(1,100):
        U=U+V[i]
        Vect.append(np.array([U[0],U[1]]))
    return Vect
def vitesse(V):
    U=[V]
    for i in range(1,100):
        V=V+np.array([random.choice((-1, 1)),random.choice((-1, 1))])
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
    n=50
    for i in range(n):
        U=vitesse(np.array([10*random.random(),10*random.random()]))
        X=np.array([10*random.random(),10*random.random()])+np.array([10*random.random(),10*random.random()])
        V=deplacement(X,U)
        for j in range(len(V)):
            AIS.append(["2018-12-31T"+heure(j),3337640+i,V[j][0],V[j][1],U[j][0],U[j][1]])
    return AIS

AIS=mismatch()
with open(output_file, 'w', newline='') as csvfile:
    c = csv.writer(csvfile, delimiter=',')
    for i in range(len(AIS)):
        print(AIS[i])
        c.writerow(AIS[i])