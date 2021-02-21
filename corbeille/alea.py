import random
import csv
import numpy as np

output_file="/tmp/guest-xhulru/Bureau/bbd_aleatoire.csv"

def deplacement(X,V):
    Vect=[X]
    U=X
    for i in range(1,len(V)):
        U=U+V[i]
        Vect.append(np.array([U[0],U[1]]))
    return Vect
def vitesse(V):
    U=[V]
    for i in range(1,100):
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
        
def inverse(X):
    X0=[]
    for i in range(len(X)):
        X0.append([-X[i][1],X[i][0]])
    return np.array(X0)
    
def AIS(U,V,i):
    AIS=[]
    for j in range(len(V)):
        AIS.append(["2018-12-31T"+heure(j),3337640+i,V[j][0],V[j][1],U[j][0],U[j][1]])
    return AIS

def dispatch():
    n=2
    U=vitesse(np.array([10*random.random(),10*random.random()]))
    X=np.array([-20,-20])
    V=deplacement(X,U)
    V2=inverse(V)
    U2=(-1)*np.array(U)
    V3=inverse(V2)
    U3=inverse(U2)
    return AIS(U2,V2,0)+AIS(U,V,1)+AIS(U3,V3,2)

AIS=dispatch()
with open(output_file, 'w', newline='') as csvfile:
    c = csv.writer(csvfile, delimiter=',')
    for i in range(len(AIS)):
        print(AIS[i])
        c.writerow(AIS[i])