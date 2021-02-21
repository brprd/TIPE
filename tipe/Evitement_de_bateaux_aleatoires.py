import numpy as np
import random as rd
import matplotlib.pyplot as plt
import matplotlib.animation as anim

import bib_dijkstra as dij
import bib_outils_Tangi as ot


def map(Li,n):
    for i in range(len(Li)):
        Li[i] = position_aleatoire(Li[i],n)
    return(Li)


def position_aleatoire(d,n):
    d0, d1 = d[0], d[1]
    l = []
    for i in (-1,1):
        for j in (-1,1):
            if ot.in_matrice((d0+i,d1+j),n):
                l.append((d0+i,d1+j))
    return rd.choice(l)

m = 100
G = dij.graphe(m)
Ri = [[1 for j in range(m)] for i in range(m)]
s0 = (0,0)
d = (m-1,m-1)
x0, y0 = s0[0], s0[1]
t = 5
r = 1000

cso = ot.absord(dij.trajet(G,Ri,s0,d)[1])
chefin = [s0]

#Li = [(m//2,m//2)]
Li = [(m//4,m//4),(m//4,3*m//4),(3*m//4,m//4),(3*m//4,3*m//4),(m//2,m//2),(m//2,m//4),(m//2,3*m//4),(m//4,m//2),(3*m//4,m//2)]
for s in Li:
    Ri = ot.obstacle(Ri,(s,t,r))
fig = plt.figure()
ax = plt.axes(xlim = (-1,m), ylim = (1,-m))
ax.set_aspect('equal')
plt.grid()

bateau1, = ax.plot([y0],[-x0], marker = 'o', color = 'g', label = 'Bateau', markersize = '10')
tso, = ax.plot(cso[0],cso[1], marker ='x', color = 'g', label = 'Trajet sans obstacle')
plt.plot(d[1], -d[0], color = 'black', marker = 'v', markersize = 15, label = 'Destination')
plt.plot(y0, -x0, color = 'gray', marker = 'v', markersize = 15, label = 'Depart')

chemin = dij.trajet(G,Ri,s0,d)[1]
C = ot.absord(chemin)
Ci = ot.absord(Li)

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
        Ri = ot.retire_obstacle(Ri,(s,t,r))
    if p > 1:
        plt.contourf(xx,-yy,B,blank,colors = difblank)
        s0 = chemin[1]
        chefin.append(s0)
        Li = map(Li,m)
        bateau1.set_data([s0[1]],[-s0[0]])
        for s in Li:
            Ri = ot.obstacle(Ri,(s,t,r))
        chemin = dij.trajet(G,Ri,s0,d)[1]
        C = ot.absord(chemin)
        Ci = ot.absord(Li)
        Cf = ot.absord(chefin)
        bateaux.remove()
        bateaux, = ax.plot(Ci[0],Ci[1], linestyle = '', color = 'orange', marker = '^', label = 'Méchants bateaux')
        traj.remove()
        traj, = ax.plot(C[0],C[1], linestyle = '', color = 'b', marker = 'x', label = 'trajectoire')
        cf.remove()
        cf, = ax.plot(Cf[0], Cf[1], color = 'r', label = 'Chemin final')
        cs = plt.contourf(xx,-yy,Ri,contour_level,colors = diffmap)
    else:
        return ()

ani = anim.FuncAnimation(fig, update, interval = 500)

plt.show()