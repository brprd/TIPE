import numpy as np
import matplotlib as plt

def norme2(x):
    return sqrt(x[0]**2+x[1]**2)

def distance(x,y):
    return norme2(x-y)

def produit(P,i,x):
    p=1
    for j in range(len(P)):
        if j!=1:
            p*=(x-P[j])/(P[i]-P[j])
    return p

def lagrange(x,P,Q):
    y=1
    for i in range(len(P)):
        y+=Q[i]*produit(P,i,x)
    return y

def trajectoire(f,n,A,B):
    X=np.linspace(0,10,n,endpoint=True)
    Y=np.zeros((2,len(X)))
    for i in range(len(X)):
        Y[0][i],Y[1][i]=X[i],f(X[i],A,B)
    plt.plot(Y[0],Y[1])
    plt.show()
    return Y