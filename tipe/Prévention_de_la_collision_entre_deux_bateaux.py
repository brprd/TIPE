import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

import bib_dijkstra as dij
import bib_outils_Tangi as ot


dim = 150
Gr = dij.graphe4(dim)

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

R1 = ot.obstacle(R1, (s2,t2,r2))
R2 = ot.obstacle(R2,(s1,t1,r1))
chemin1 = dij.trajet(Gr,R1,s1,d1)[1]
chemin2 = dij.trajet(Gr,R2,s2,d2)[1]

Rt = np.array(R1) + np.array(R2)

C1 = ot.absord(chemin1)
C2 = ot.absord(chemin2)

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
        R2 = ot.obstacle(R2,(s1,t1,r1))
        for s in chemin1[2:6]:
            R2[s[0]][s[1]] = 500
        chemin1 = dij.trajet(Gr,R1,s1,d1)[1]
        C1 = ot.absord(chemin1)
        Cf1 = ot.absord(chefin1)
        traj1.remove()
        traj1, = ax.plot(C1[0],C1[1], linestyle = '', color = 'g', marker = 'x', label = 'Trajectoire1')
        R1 = ot.retire_obstacle(R1,(s2,t2,r2))
        s2 = chemin2[1]
        chefin2.append(s2)
        bateau2.set_data([s2[1]],[-s2[0]])
        R1 = ot.obstacle(R1,(s2,t2,r2))
        chemin2 = dij.trajet(Gr,R2,s2,d2)[1]
        C2 = ot.absord(chemin2)
        Cf2 = ot.absord(chefin2)
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
        chemin1 = dij.trajet(Gr,R1,s1,d1)[1]
        C1 = ot.absord(chemin1)
        Cf1 = ot.absord(chefin1)
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
        chemin2 = dij.trajet(Gr,R2,s2,d2)[1]
        C2 = ot.absord(chemin2)
        Cf2 = ot.absord(chefin2)
        traj2.remove()
        traj2, = ax.plot(C2[0],C2[1], linestyle = '', color = 'g', marker = 'x', label = 'Trajectoire2')
        cf2.remove()
        cf2, = ax.plot(Cf2[0], Cf2[1], color = 'purple', label = 'Chemin final 2')
        Rt = np.array(R1) + np.array(R2)
        cs = plt.contourf(xx,-yy,Rt,contour_level,colors = diffmap)
    else:
        return ()

ani = anim.FuncAnimation(figp, update2, interval = 1000)

plt.show()