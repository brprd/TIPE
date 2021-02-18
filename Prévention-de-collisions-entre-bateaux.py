## Importation des bibliothèques :

import numpy as np
import random as rd
import matplotlib.pyplot as plt
import matplotlib.animation as a
import matplotlib.colors 
import time

## Implémentation des graphes :

# Les sommets sont les couples (i,j) pour i=0,...,n-1 , j=0,...,n-1, où n désigne la taille de la matrice.
# Pour un graphe G, G[i][j] est la liste des voisins de (i,j).

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
    
## Algorithme de Dijkstra : 

def Dijkstra_tas(G,R,s):
    n = len(G)
    i0, j0 = s[0], s[1]
    dist = [[np.Infinity for j in range(n)] for i in range(n)]
    chemins = [[[s] for j in range(n)] for i in range(n)]
    dist[i0][j0] = 0
    t = [0,(s,0)]
    p = len(t)
    visit = [[False for j in range(n)] for i in range(n)] 
    while p > 1:
        m = retire_tas(t)
        c = m[0]
        c0 , c1 = c[0] , c[1]
        if not visit[c0][c1]:
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
    
# Pour les tests :
n = 200
Gn = graphe(n)
Rn = [[rd.randint(1,100) for j in range(n)] for i in range(n)]
sn = (0,0)
dn = (n-1,n-1)     
    
## Affichage de la carte des risques et du trajet :

def affiche_carte(G,R,s,d):
    n = len(G)
    plt.axis([-1,n,-n,1])
    plt.grid()
    plt.plot(s[1],-s[0], marker = "o", color = "g", label = "Origine")
    x = np.linspace(-1,n,n)
    y = np.linspace(-1,n,n)
    xx, yy = np.meshgrid(x,y)
    contour_level = [0,20,40,60,80,200]
    diffmap = ["#FFFFFF","#F8DEDE","#F1BEBE","#EB9E9E","#EB7D82"]
    cs = plt.contourf(xx,-yy,R,contour_level,colors = diffmap)
    plt.colorbar(cs,label = 'risque')
    plt.plot(d[1],-d[0], marker = "o", color = "r", label = "Destination")
    plt.legend()
    plt.show()
    
def affiche_trajet(G,R,s,d):
    n = len(G)
    chemin = trajet(G,R,s,d)[1]
    p = len(chemin)
    plt.axis([-1,n,-n,1])
    plt.grid()
    plt.plot(s[1],-s[0], marker = "o", color = "g", label = "Origine")
    x = np.linspace(-1,n,n)
    y = np.linspace(-1,n,n)
    xx, yy = np.meshgrid(x,y)
    contour_level = [0,20,40,60,80,200]
    diffmap = ["#FFFFFF","#F8DEDE","#F1BEBE","#EB9E9E","#EB7D82"]
    cs = plt.contourf(xx,-yy,R,contour_level,colors = diffmap)
    plt.colorbar(cs,label = 'risque')
    if p > 1:
        for k in range(1,p-1):
            c = chemin[k]
            plt.plot(c[1], -c[0], marker = "x", color = "b")
        plt.plot(d[1],-d[0], marker = "o", color = "r", label = "Destination")
    plt.legend()
    plt.show()
    
## Eviter un ou des obstacles fixes :

# Fonctions utiles :

def carre_centre(c,r):
    s = []
    i, j = c[0], c[1]
    s.append((i-r,j))
    s.append((i+r,j))
    s.append((i,j-r))
    s.append((i,j+r))
    for p in range(1,r+1):
        s.append((i-r,j+p))
        s.append((i-r,j-p))
        s.append((i+r,j+p))
        s.append((i+r,j-p))
    for p in range(1,r):
        s.append((i+p,j-r))
        s.append((i-p,j-r))
        s.append((i+p,j+r))
        s.append((i-p,j+r))
    return s

def in_matrice(c,n):
    i, j = c[0], c[1]
    return (i >= 0 and i < n and j >= 0 and j < n)
    
# Ajouter ou retirer les obstacles :

def obstacle(R,o):
    n = len(R)
    i, j = o[0][0], o[0][1]
    k = o[1]
    rm = o[2]
    R[i][j] = R[i][j] + rm
    for r in range(1,k):
        s = carre_centre((i,j),r)
        for c in s:
            if in_matrice(c,n):
                R[c[0]][c[1]] = R[c[0]][c[1]] + (rm//(r+1)) 
    return R
    
def retire_obstacle(R,o):
    n = len(R)
    i, j = o[0][0], o[0][1]
    k = o[1]
    rm = o[2]
    R[i][j] = R[i][j] - rm
    for r in range(1,k):
        s = carre_centre((i,j),r)
        for c in s:
            if in_matrice(c,n):
                R[c[0]][c[1]] = R[c[0]][c[1]] - (rm//(r+1)) 
    return R

# Eviter un ou plusieurs obstacles : 

def evite_obstacle(G, R, s, d, o):
    R = obstacle(R,o)
    plt.plot(o[0][1],-o[0][0], marker = '^', markersize = 15, color = 'orange', label = "Obstacle")
    affiche_trajet(G,R,s,d)
    
def evite_liste_obstacles(G, R, s, d, O):
    X = []
    Y = []
    for o in O:
        R = obstacle(R,o)
        X.append(o[0][1])
        Y.append(-o[0][0])
    plt.plot(X,Y, marker = '^', markersize = 10, color = 'orange', label = "Obstacles", linestyle = '')
    affiche_trajet(G,R,s,d)
    
## Animations : 

# Fonctions utiles :

def position_aleatoire(d,n):
    d0, d1 = d[0], d[1]
    l = []
    for i in (-1,1):
        for j in (-1,1):
            if in_matrice((d0+i,d1+j),n):
                l.append((d0+i,d1+j))
    return rd.choice(l)

def map(Li,n):
    for i in range(len(Li)):
        Li[i] = position_aleatoire(Li[i],n)
    return(Li)

def absord(L):
    X = []
    Y = []
    for c in L:
        X.append(c[1])
        Y.append(-c[0])
    return X,Y

#################
# Evitement de bateaux aléatoires :

m = 100
G = graphe(m)
Ri = [[1 for j in range(m)] for i in range(m)]
s0 = (0,0)
d = (m-1,m-1)
x0, y0 = s0[0], s0[1]
t = 5
r = 1000

cso = absord(trajet(G,Ri,s0,d)[1])
chefin = [s0]

#Li = [(m//2,m//2)]
Li = [(m//4,m//4),(m//4,3*m//4),(3*m//4,m//4),(3*m//4,3*m//4),(m//2,m//2),(m//2,m//4),(m//2,3*m//4),(m//4,m//2),(3*m//4,m//2)]
for s in Li:
    Ri = obstacle(Ri,(s,t,r))
fig = plt.figure()
ax = plt.axes(xlim = (-1,m), ylim = (1,-m))
ax.set_aspect('equal')
plt.grid()

bateau1, = ax.plot([y0],[-x0], marker = 'o', color = 'g', label = 'Bateau', markersize = '10')
tso, = ax.plot(cso[0],cso[1], marker ='x', color = 'g', label = 'Trajet sans obstacle')
plt.plot(d[1], -d[0], color = 'black', marker = 'v', markersize = 15, label = 'Destination')
plt.plot(y0, -x0, color = 'gray', marker = 'v', markersize = 15, label = 'Depart')

chemin = trajet(G,Ri,s0,d)[1]
C = absord(chemin)
Ci = absord(Li)

bateaux, = ax.plot(Ci[0],Ci[1], linestyle = '', color = 'orange', marker = '^', label = 'Bateaux inconnus')
traj, = ax.plot(C[0],C[1], linestyle = '', color = 'b', marker = 'x', label = 'Trajectoire') 
cf, = ax.plot([s0[1]], [-s0[0]], color = 'r', label = 'Chemin final')

plt.legend()

x = np.linspace(-1,m,m)
y = np.linspace(-1,m,m)
xx, yy = np.meshgrid(x,y)
contour_level = [0,200,400,600,800,10000]
diffmap = ["#FFFFFF","#F8DEDE","#F1BEBE","#EB9E9E","#EB7D82"]
cs = plt.contourf(xx,-yy,Ri,contour_level,colors = diffmap)
col = plt.colorbar(cs,label = 'risque')
B = [ [1 for i in range(m)] for j in range(m)]
blank = [0,np.Infinity]
difblank = ["#FFFFFF"]


def update(frame):
    global s0
    global bateaux
    global Ri
    global chemin
    global traj
    global Li
    global cf
    p = len(chemin)
    for s in Li:
        Ri = retire_obstacle(Ri,(s,t,r))
    if p > 1:
        plt.contourf(xx,-yy,B,blank,colors = difblank)
        s0 = chemin[1]
        chefin.append(s0)
        Li = map(Li,m)
        bateau1.set_data([s0[1]],[-s0[0]])
        for s in Li:
            Ri = obstacle(Ri,(s,t,r))
        chemin = trajet(G,Ri,s0,d)[1]
        C = absord(chemin)
        Ci = absord(Li)
        Cf = absord(chefin)
        bateaux.remove()
        bateaux, = ax.plot(Ci[0],Ci[1], linestyle = '', color = 'orange', marker = '^', label = 'Méchants bateaux')
        traj.remove()
        traj, = ax.plot(C[0],C[1], linestyle = '', color = 'b', marker = 'x', label = 'trajectoire')
        cf.remove()
        cf, = ax.plot(Cf[0], Cf[1], color = 'r', label = 'Chemin final')
        cs = plt.contourf(xx,-yy,Ri,contour_level,colors = diffmap)
    else:
        return ()

anim = a.FuncAnimation(fig, update, interval = 500)

plt.show()

#############
# Prévention de la collision entre deux bateaux : 

dim = 150
Gr = graphe4(dim)

s1 = (13*dim//32,0)
d1 = (0,dim-1)
a1 , b1 = s1[0] , s1[1]
R1 = [[1 for j in range(dim)] for i in range(dim)]
t1 = 12
r1 = 5000

s2 = (0,dim//2)
d2 = (dim-1,dim-1)
a2 , b2 = s2[0] , s2[1]
R2 = [[1 for j in range(dim)] for i in range(dim)]
t2 = 10
r2 = 5000
#s = s1

figp = plt.figure()
ax = plt.axes(xlim = (-1,dim), ylim = (1,-dim))
ax.set_aspect('equal')
plt.grid()

plt.plot(b1, -a1, marker = '^', color = 'gray', label = 'Depart1' )
plt.plot(d1[1], -d1[0], marker = 'v', color = 'gray', label = 'Destination1' )
bateau1, = ax.plot([b1],[-a1], marker = 'o', color = 'green', label = 'Bateau1', markersize = '10')

plt.plot(b2, -a2, marker = '^', color = 'magenta', label = 'Depart2' )
plt.plot(d2[1], -d2[0], marker = 'v', color = 'magenta', label = 'Destination2' )
bateau2, = ax.plot([b2],[-a2], marker = 'o', color = 'purple', label = 'Bateau2', markersize ='10')

R1 = obstacle(R1, (s2,t2,r2))
R2 = obstacle(R2,(s1,t1,r1))
chemin1 = trajet(Gr,R1,s1,d1)[1]
chemin2 = trajet(Gr,R2,s2,d2)[1]

Rt = np.array(R1) + np.array(R2)

C1 = absord(chemin1)
C2 = absord(chemin2)

traj1, = ax.plot(C1[0],C1[1], linestyle = '', color = 'g', marker = 'x', label = 'Trajectoire1')
traj2, = ax.plot(C2[0],C2[1], linestyle = '', color = 'purple', marker = 'x', label = 'Trajectoire2')

chefin1 = [s1]
chefin2 = [s2]

cf1, = ax.plot([s1[1]], [-s1[0]], color = 'g', label = 'Chemin final 1')
cf2, = ax.plot([s2[1]], [-s2[0]], color = 'purple', label = 'Chemin final 2')

plt.legend()

x = np.linspace(-1,dim,dim)
y = np.linspace(-1,dim,dim)
xx, yy = np.meshgrid(x,y)
contour_level = [0,200,400,600,800,10000]
diffmap = ["#FFFFFF","#F8DEDE","#F1BEBE","#EB9E9E","#EB7D82"]
cs = plt.contourf(xx,-yy,Rt,contour_level,colors = diffmap)
col = plt.colorbar(cs,label = 'risque')
B = [ [1 for i in range(dim)] for j in range(dim)]
blank = [0,np.Infinity]
difblank = ["#FFFFFF"]

def update2(frame):
    global s1, s2, R1, R2, chemin1, chemin2, traj1, traj2, C1, C2, Rt, cf1, cf2 
    p1 , p2 = len(chemin1), len(chemin2)
    if p1 > 1 and p2 > 1:
        plt.contourf(xx,-yy,B,blank,colors = difblank)
        R2 = [[1 for j in range(dim)] for i in range(dim)]
        s1 = chemin1[1]
        chefin1.append(s1)
        bateau1.set_data([s1[1]],[-s1[0]])
        R2 = obstacle(R2,(s1,t1,r1))
        for s in chemin1[2:6]:
            R2[s[0]][s[1]] = 500
        chemin1 = trajet(Gr,R1,s1,d1)[1]
        C1 = absord(chemin1)  
        Cf1 = absord(chefin1)      
        traj1.remove()
        traj1, = ax.plot(C1[0],C1[1], linestyle = '', color = 'g', marker = 'x', label = 'Trajectoire1')        
        R1 = retire_obstacle(R1,(s2,t2,r2))        
        s2 = chemin2[1]
        chefin2.append(s2)
        bateau2.set_data([s2[1]],[-s2[0]])
        R1 = obstacle(R1,(s2,t2,r2))
        chemin2 = trajet(Gr,R2,s2,d2)[1]
        C2 = absord(chemin2)
        Cf2 = absord(chefin2)   
        traj2.remove()
        traj2, = ax.plot(C2[0],C2[1], linestyle = '', color = 'purple', marker = 'x', label = 'Trajectoire2')
        cf1.remove()
        cf1, = ax.plot(Cf1[0], Cf1[1], color = 'g', label = 'Chemin final 1')
        cf2.remove()
        cf2, = ax.plot(Cf2[0], Cf2[1], color = 'purple', label = 'Chemin final 2')
        Rt = np.array(R1) + np.array(R2)
        cs = plt.contourf(xx,-yy,Rt,contour_level,colors = diffmap)
    if p1 > 1 and p2 == 1:
        plt.contourf(xx,-yy,B,blank,colors = difblank)
        s1 = chemin1[1]
        chefin1.append(s1)
        bateau1.set_data([s1[1]],[-s1[0]])
        chemin1 = trajet(Gr,R1,s1,d1)[1]
        C1 = absord(chemin1)
        Cf1 = absord(chefin1)
        traj1.remove()
        traj1, = ax.plot(C1[0],C1[1], linestyle = '', color = 'g', marker = 'x', label = 'Trajectoire1')
        cf1.remove()
        cf1, = ax.plot(Cf1[0], Cf1[1], color = 'g', label = 'Chemin final 1')
        Rt = np.array(R1) + np.array(R2)
        cs = plt.contourf(xx,-yy,Rt,contour_level,colors = diffmap)
    elif p1 == 1 and p2 > 1:
        plt.contourf(xx,-yy,B,blank,colors = difblank)
        s2 = chemin2[1]
        chefin2.append(s2)
        bateau2.set_data([s2[1]],[-s2[0]])
        chemin2 = trajet(Gr,R2,s2,d2)[1]
        C2 = absord(chemin2)
        Cf2 = absord(chefin2)
        traj2.remove()
        traj2, = ax.plot(C2[0],C2[1], linestyle = '', color = 'g', marker = 'x', label = 'Trajectoire2')
        cf2.remove()
        cf2, = ax.plot(Cf2[0], Cf2[1], color = 'purple', label = 'Chemin final 2')
        Rt = np.array(R1) + np.array(R2)
        cs = plt.contourf(xx,-yy,Rt,contour_level,colors = diffmap)
    else:
        return ()
        
animp = a.FuncAnimation(figp, update2, interval = 1000)

plt.show()        