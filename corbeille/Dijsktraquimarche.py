import numpy as np
import random as rd
import matplotlib.pyplot as plt

## Implémentation du graphe :

# Les sommets sont les couples (i,j) pour i=0,...,n-1 , j=0,...,n-1, où n désigne la taille de la matrice
# Les sommets sont classés dans l'ordre (0,0),(0,1),(0,2),...,(0,n-1),(1,0),...,(n-1,n-2),(n-1,n-1)

def graphe(n):
    L = [[[] for j in range(n)] for i in range(n)] # on implémente le graphe par une liste d'adjacence
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
    
# Si M est la matrice obtenue par l'appel graphe(n), alors pour tout (i,j) élément de {0,...,n-1}**2, M[i][j] est la liste des voisins du sommet (i,j)

## Algorithme de Dijkstra :

def minimum_matrice(M,L):
    C = []
    n = len(M)
    for i in range(n):
        for j in range(n):
            if not (i,j) in L:
                C.append((i,j))
    c0 = C[0]
    for c in C:
        if M[c[0]][c[1]] < M[c0[0]][c0[1]]:
            c0 = c
    return c0

def Dijkstra(G,R,c): # G est un graphe, R la matrice des risques associée et c un sommet (un couple)
    n = len(G)
    i0, j0 = c[0], c[1]
    dist = [[np.Infinity for j in range(n)] for i in range(n)] 
    dist[i0][j0] = 0
    Sg = [] # liste des sommets déjà visités
    while len(Sg) < n**2 :
        m = minimum_matrice(dist,Sg)
        Sg.append(m)
        V = G[m[0]][m[1]] # liste des voisins de m
        for v in V:
            v0, v1 = v[0], v[1]
            dist[v0][v1] = min(dist[v0][v1],dist[m[0]][m[1]] + R[v0][v1]) # formule de l'algorithme de Dijkstra
    return dist
    
def risque_minimum(G,R,s,d):
    D = Dijkstra(G,R,s)
    return D[d[0]][d[1]]
        
G = graphe(5)
R = [[0,9,1,1,12],
     [4,2,1,4,6],
     [8,0,0,0,488],
     [0,3,0,50,0],
     [3,0,9,0,1]]
s = (0,0)
d = (4,4)

## Adaptation de l'algorithme pour retourner le chemin :

def Dijkstra_chemins(G,R,c): # G est un graphe, R la matrice des risques associée et c un sommet (un couple)
    n = len(G)
    i0, j0 = c[0], c[1]
    dist = [[np.Infinity for j in range(n)] for i in range(n)]
    chemins = [[[] for j in range(n)] for i in range(n)]
    dist[i0][j0] = 0
    chemins[i0][j0] = [c] # matrice de listes répertoriant les chemins entre la source et les autres sommets
    Sg = [] # liste des sommets déjà visités
    while len(Sg) < n**2 :
        m = minimum_matrice(dist,Sg)
        Sg.append(m)
        V = G[m[0]][m[1]] # liste des voisins de m
        for v in V:
            v0, v1 = v[0], v[1]
            dist[v0][v1] = min(dist[v0][v1],dist[m[0]][m[1]] + R[v0][v1]) # formule de l'algorithme de Dijkstra
            if dist[v0][v1] >= dist[m[0]][m[1]] + R[v0][v1]:
                ch = chemins[m[0]][m[1]] + [v]
                chemins[v0][v1] = ch
    return chemins
    
def chemin_le_moins_risque(G,R,c,d):
    D = Dijkstra_chemins(G,R,c)
    return D[d[0]][d[1]]
   
# Complexité : proportionnelle au nombre de sommets au carré, donc ici en O(n**4). C'est trop si on a une grande taille de matrice
    
## Essais :

n = 200
Gn = graphe(n)
Rn = [[rd.randint(0,100) for j in range(n)] for i in range(n)]
sn = (0,0)
dn = (n-1,n-1)

## Implémentation avec des tas:

# L'implémentation avec des tas (file de priorité codée avec un tas) permet de limiter la complexité du programme à O(n**2*log(n)) dans le cas d'un graphe peu dense, ce qui est le cas ici ! Nous adopterons cette implémentation.

# Nous codons le tas avec une liste. On numérote les sommets de 1 à n**2; on prend donc une liste de taille n**2+1, dont le premier élément est un 0. Les fils d'un élément d'indice k sont ceux d'indices 2k et 2k+1.

# Modélisation de la liste des sommets à visiter par un tas

# def tas_ini(p,c): # création du tas initial
#     l=[]
#     for i in range(p):
#         for j in range(p):
#             if (i,j) != c:
#                 l.append((i,j))
#     return [0] + [c] + l
# 
# # Fonction descente : on met un élément à sa place dans le tas par descente. Tant que l'élément est plus grand que l'un de ses fils, on échange les deux.
# 
# def descente(t,p,k,m):
#     if 2*k > p:
#         return t
#     else:
#         c1 = t[2*k]
#         c2 = t[2*k+1]
#         if 2*k == p or m[c1[0]][c1[1]] < m[c2[0]][c2[1]]:
#             j = 2*k
#         else:
#             j = 2*k+1
#         t[k], t[j] = t[j], t[k]
#         return descente(t,p,j,m)
#     
# def suppression_minimum(t,p,m): # supprime le minimum d'un tas, le renvoie et conserve la structure de tas
#     min = t[1]
#     t[1], t[p] = t[p], t[1]
#     descente(t,p-1,1,m)
#     return min
#         
# T = tas_ini(7,(0,0))
# 
# # Nouvelle version de Dijkstra :
# 
# def Dijkstra_tas(G,R,s):
#     n = len(G)
#     i0, j0 = s[0], s[1]
#     dist = [[np.Infinity for j in range(n)] for i in range(n)]
#     chemins = [[[] for j in range(n)] for i in range(n)]
#     dist[i0][j0] = 0
#     chemins[i0][j0] = [s]
#     t = tas_ini(n,s)
#     p = n**2
#     while p > 1:
#         m = suppression_minimum(t,p,dist)
#         m0, m1 = m[0], m[1]
#         V = G[m0][m1] 
#         for v in V:
#             v0, v1 = v[0], v[1]
#             dist[v0][v1] = min(dist[v0][v1],dist[m[0]][m[1]] + R[v0][v1])
#             descente(t,p,
#             if dist[v0][v1] >= dist[m[0]][m[1]] + R[v0][v1]:
#                 ch = chemins[m[0]][m[1]] + [v]
#                 chemins[v0][v1] = ch
#         p = p - 1
#         
#     return dist, chemins
#     
# #    return dist,chemins
# 
# def chemin_le_moins_risque_tas(G,R,c,d):
#     D = Dijkstra_tas(G,R,c)
#     di = D[0]
#     ch = D[1]
#     return di[d[0]][d[1]], ch[d[0]][d[1]]

def descente(t,p,k):
    if 2*k > p:
        return t
    else:
        if 2*k == p or (2*k < p and t[2*k][1] < t[2*k+1][1]):
            j = 2*k
        else:
            j = 2*k+1
        if t[k][1] < t[j][1]:
            t[k] , t[j] = t[j] , t[k]
        return descente(t,p,j)
        
def montee(t,k):
    if k < 2:
        return t
    else:
        j = k//2
        if t[k][1] < t[j][1]:
            t[k] , t[j] = t[j] , t[k]
        return montee(t,j)

def ajouter_tas(t,x):
    t.append(x)
    k = len(t)
    return montee(t,k-1)

def retire_tas(t):
    min = t.pop(1)
    n = len(t)
    descente(t,n-1,1)
    return min
    
def Dijkstra_tas(G,R,s):
    n = len(G)
    i0, j0 = s[0], s[1]
    dist = [[np.Infinity for j in range(n)] for i in range(n)]
    chemins = [[[] for j in range(n)] for i in range(n)]
    dist[i0][j0] = 0
    chemins[i0][j0] = [s]
    t = [0,(s,0)]
    p = len(t)
    visit = [[False for j in range(n)] for i in range(n)]
    while p > 1:
        m = retire_tas(t)
        c = m[0]
        c0 , c1 = c[0] , c[1]
        V = G[c0][c1]
        visit[c0][c1] = True
        for v in V:
            v0 , v1 = v[0] , v[1]
            dist[v0][v1] = min(dist[v0][v1],dist[c0][c1] + R[v0][v1])
            if not visit[v0][v1]:
                ajouter_tas(t,(v,dist[v0][v1]))
            if dist[v0][v1] >= dist[c0][c1] + R[v0][v1]:
                ch = chemins[c0][c1] + [v]
                chemins[v0][v1] = ch
        p = len(t)
    return dist , chemins
    
def trajet(G,R,s,d):
    Dij = Dijkstra_tas(G,R,s)
    distances, chemins = Dij[0], Dij[1]
    return distances[d[0]][d[1]], chemins[d[0]][d[1]]
    
## Affichage des résultats:

def affiche_trajet(G,R,s,d):
    n = len(G)
    X = []
    Y = []
    chemin = trajet(G,R,s,d)[1]
    p = len(chemin)
    plt.axis([-1,n,-n,1])
    plt.grid()
    plt.plot(s[1],-s[0], marker = "o", color = "g", label = "Origine")
    if p > 1:
        for k in range(1,p-1):
            c = chemin[k]
            plt.plot(c[1], -c[0], marker = "x", color = "b")
        plt.plot(d[1],-d[0], marker = "o", color = "r", label = "Destination")
    plt.legend()
    plt.show()
    
Gp = graphe(5)
Rp = [[0,0,0,0,0],
      [0,0,1,1,1],
      [0,1,2,3,2],
      [0,1,2,3,2],
      [5,0,1,1,1]]
sp = (0,0)
dp = (4,4)
