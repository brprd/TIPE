input_file="D:/brieu/Documents/MP/TIPE/tipe/bdd_aleatoire.csv"
n=100
map_color='seismic'


import numpy as np
import matplotlib.pyplot as plt
import bib_dijkstra as dij
import bib_ais as ais


def centre_bateau(boats, mmsi,t, M):
    def norme2(x):
        return np.sqrt(x[0]**2+x[1]**2)
    def distance(x,y):
        return norme2(x-y)
    boats_out , boats_kal_out = boats[mmsi].get_data() , boats[mmsi].get_kal_data_pred()
    U=np.array([boats_out[t][1],boats_out[t][2]])
    V=np.array([boats_kal_out[t+1][0],boats_kal_out[t+1][1]])
    L=16/3*distance(V,U)
    pas=L/n
    repartition(U,t,mmsi, boats, pas, M)


def survoisin(J, p, M):
    q=0.5
    N=np.zeros((n,n))
    if int(J[0])<n and int(J[1])<n:
        N[int(J[0]),int(J[1])]=p
        for i in range(int(J[0])-10,int(J[0])+10):
            for j in range(int(J[1])-10,int(J[1])+10):
                N[i,j]=int(p*np.exp(-q/2*((J[0]-i)**2+(J[1]-j)**2)))
    M+=N

def repartition(U,t,mmsi, boats, pas, M):
    def conversion(X):
        return np.array([round(X[0]/pas),round(X[1]/pas)])

    for key in boats.keys():#on parcourt tous les bateaux
        if key != mmsi:#si ce n'est pas le bateau qu'on étudie alors on ajoute du risque
            i=0
            X=np.array([boats[key].get_data()[t][1],boats[key].get_data()[t][2]])
            if np.abs(conversion(X-U)[0])<n//2 and np.abs(conversion(X-U)[1])<n//2:
                points=[]
                Y=np.array([boats[key].get_kal_data_pred()[t+1][0],boats[key].get_kal_data_pred()[t+1][1]])
                J=conversion(X-U)
                while i<10 and (np.abs(conversion(J)[0])<n//2-1 and np.abs(conversion(J)[1])<n//2-1):
                    points.append((J+np.array([n//2,n//2])))
                    survoisin((J+np.array([n//2,n//2])), 1/(boats[mmsi].get_kal_cov_pred()[t][0,0]), M)
                    J=conversion((i/9)*X+(1-i/9)*Y-U)
                    i+=1


boats=ais.calculs_bateaux(input_file)

M=np.zeros((n,n))

centre_bateau(boats, "3337640",4, M)
X = np.linspace(0,n,n)
t = np.linspace(0,n,n)
X, t = np.meshgrid(X, t, indexing = 'ij')

plt.pcolor(X, t, M, cmap=map_color, shading="auto")
#plt.colorbar()

plt.xlabel("Latitude")
plt.ylabel("Longitude")

#plus court chemin
Gn = dij.graphe(n)
sn = (58,40)
dn = (10,20)
dij.affiche_trajet(Gn,M,sn,dn)
