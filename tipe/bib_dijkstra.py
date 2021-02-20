import numpy as np
import matplotlib.pyplot as plt

def trajet(G,R,s,d):
    Dij = __Dijkstra_tas(G,R,s)
    distances, chemins = Dij[0], Dij[1]
    return distances[d[0]][d[1]], chemins[d[0]][d[1]]

## Implémentation des __graphes :

# Les sommets sont les couples (i,j) pour i=0,...,n-1 , j=0,...,n-1, où n désigne la taille de la matrice.
# Pour un __graphe G, G[i][j] est la liste des voisins de (i,j).

# Implémentation 1 : chaque sommet est relié à quatre sommets au maximum (haut, bas, gauche, droite).

def graphe4(n):
    L = [[[] for j in range(n)] for i in range(n)]
    L[0][0] = [(1,0),(0,1)]
    L[0][n-1] = [(0,n-2),(1,n-1)]
    L[n-1][n-1] = [(n-1,n-2),(n-2,n-1)]
    L[n-1][0] = [(n-1,1),(n-2,0)]
    for i in range(1,n-1):
        L[i][0] = [(i-1,0),(i,1),(i+1,0)]
        L[i][n-1] = [(i-1,n-1),(i,n-2),(i+1,n-1)]
    for j in range(1,n-1):
        L[0][j] = [(0,j-1),(1,j),(0,j+1)]
        L[n-1][j] = [(n-1,j-1),(n-2,j),(n-1,j+1)]
    for i in range(1,n-1):
        for j in range(1,n-1):
            L[i][j] = [(i-1,j),(i,j-1),(i,j+1),(i+1,j)]
    return L

# Implémentation 2 : chaque sommet est relié à huit sommets au maximum (haut, bas, gauche, droite, plus les sommets en diagonale).

def graphe(n):
    L = [[[] for j in range(n)] for i in range(n)]
    L[0][0] = [(1,0),(0,1),(1,1)]
    L[0][n-1] = [(0,n-2),(1,n-1),(1,n-2)]
    L[n-1][n-1] = [(n-1,n-2),(n-2,n-1),(n-2,n-2)]
    L[n-1][0] = [(n-1,1),(n-2,0),(n-2,1)]
    for i in range(1,n-1):
        L[i][0] = [(i-1,0),(i,1),(i+1,0),(i-1,1),(i+1,1)]
        L[i][n-1] = [(i-1,n-1),(i,n-2),(i+1,n-1),(i-1,n-2),(i+1,n-2)]
    for j in range(1,n-1):
        L[0][j] = [(0,j-1),(1,j),(0,j+1),(1,j-1),(1,j+1)]
        L[n-1][j] = [(n-1,j-1),(n-2,j),(n-1,j+1),(n-2,j-1),(n-2,j+1)]
    for i in range(1,n-1):
        for j in range(1,n-1):
            L[i][j] = [(i-1,j),(i,j-1),(i,j+1),(i+1,j),(i-1,j-1),(i-1,j+1),(i+1,j-1),(i+1,j+1)]
    return L




## Fonctions élémentaires sur les tas :

def __descente(t,p,k):
    if 2*k > p:
        return t
    else:
        if 2*k == p or (2*k < p and t[2*k][1] < t[2*k+1][1]):
            j = 2*k
        else:
            j = 2*k+1
        if t[k][1] < t[j][1]:
            t[k] , t[j] = t[j] , t[k]
        return __descente(t,p,j)

def __montee(t,k):
    if k < 2:
        return t
    else:
        j = k//2
        if t[k][1] < t[j][1]:
            t[k] , t[j] = t[j] , t[k]
        return __montee(t,j)

def __ajouter_tas(t,x):
    t.append(x)
    k = len(t)
    return __montee(t,k-1)

def __retire_tas(t):
    min = t.pop(1)
    n = len(t)
    __descente(t,n-1,1)
    return min

## Algorithme de Dijkstra :

def __Dijkstra_tas(G,R,s):
    n = len(G)
    i0, j0 = s[0], s[1]
    dist = [[np.Infinity for j in range(n)] for i in range(n)]
    chemins = [[[s] for j in range(n)] for i in range(n)]
    dist[i0][j0] = 0
    t = [0,(s,0)]
    p = len(t)
    visit = [[False for j in range(n)] for i in range(n)]
    while p > 1:
        m = __retire_tas(t)
        c = m[0]
        c0 , c1 = c[0] , c[1]
        if not visit[c0][c1]:
            V = G[c0][c1]
            visit[c0][c1] = True
            for v in V:
                v0 , v1 = v[0] , v[1]
                dist[v0][v1] = min(dist[v0][v1],dist[c0][c1] + R[v0][v1])
                if not visit[v0][v1]:
                    __ajouter_tas(t,(v,dist[v0][v1]))
                if dist[v0][v1] >= dist[c0][c1] + R[v0][v1]:
                    ch = chemins[c0][c1] + [v]
                    chemins[v0][v1] = ch
        p = len(t)
    return dist , chemins

##Affichage

def affiche_trajet(G,R,s,d):
    n = len(G)
    X = []
    Y = []
    chemin = trajet(G,R,s,d)[1]
    p = len(chemin)
    plt.axis([1,n,n,1])
    plt.grid()
    plt.plot(s[0],s[1], marker = "o", color = "g", label = "Origine")
    if p > 1:
        for k in range(1,p-1):
            c = chemin[k]
            plt.plot(c[0], c[1], marker = "o", markersize=2, color = "y")
        plt.plot(d[0],d[1], marker = "o", color = "r", label = "Destination")
    plt.legend()
    plt.show()