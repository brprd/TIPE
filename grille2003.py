import csv
import numpy as np
import time
import matplotlib.pyplot as plt

input_file='/tmp/guest-dh4vsh/Bureau/bbd.csv'
RT = 6311000 #rayon de la Terre (en m)
pi=np.pi

time_scale=1000#time_scale=x => le temps s'écoule x fois plus vite qu'en vrai

interval=int(1000/time_scale) #ne pas toucher, modifier time_scale au-dessus



data=[] #contient les données du fichier d'entrée
boats={} #se remplit des bateaux détectés au fur et à mesure
boats_out={}
boats_kal_out={}
boats_covariance={}
pas,L,no,M,qfact=0, 0, 100,np.zeros(0),0.5
delta_t =120

def norme2(x):
    return np.sqrt(x[0]**2+x[1]**2)

def distance(x,y):
    return norme2(x-y)
##Définition matrice
def grille(Li,a):
    global L
    global no
    global M
    global X
    global pas
    L=Li
    no=a
    pas=L/no

    M=np.zeros((no,no))


def voisinage(M,a,b):
    max=M[a][b]
    x,y=a,b

    for i in range(a-1,a+1):
        for j in range (b-1,b+1):
            try:
                if M[i][j]>max:
                    max=M[i][j]
            except IndexError:
                None
    M[a][b]=max-1



def survoisin(J, p):
    global no
    global M
    global qfact
    q=qfact
    N=np.zeros((no,no))
    if int(J[0])<no and int(J[1])<no:
        N[int(J[0]),int(J[1])]=p
        for i in range(no):
            for j in range(no):
                N[i,j]=int(p*np.exp(-q/2*((J[0]-i)**2+(J[1]-j)**2)))
    M+=N

def conversion(X):
    return np.array([round(X[0]/pas),round(X[1]/pas)])

##charger données
with open(input_file, 'r', newline='') as input:
    csvreader = csv.reader(input, delimiter=',')
    for row in csvreader:

        hours=int(row[0][11:13])
        minuts=int(row[0][14:16])
        seconds=int(row[0][17:19])
        temps=int(hours*3600+minuts*60+seconds)

        mmsi=row[1]

        latitude=float(row[2])
        longitude=float(row[3])

        #speed over ground (SOG) : vitesse par rapport au sol (pas à la mer)
        SOG=float(row[4])*0.514 #1 noeud ~= 0.514 m/s
        #course over ground (COG) : direction de déplacement du bateau (pas son orientation), par rapport au Nord
        COG=float(row[5])*pi/180 #l'angle est donné en degrés

        vitesse_lat=SOG*np.cos(COG)/RT
        vitesse_lon=(SOG*np.sin(COG))/(RT*np.cos(latitude*pi/180)) #on considèrera que cos(lat) ne varie pas entre 2 mesures consécutives

        while temps > len(data)-1:
            data.append([]) #secondes pendant lesquelles rien est reçu
        data[temps].append([mmsi,[temps, latitude, longitude, vitesse_lat, vitesse_lon]])


len_data=len(data)

class Boat:
    def __init__(self, mmsi, vecteur): #vecteur = [instant, latitude, longitude, vitesse_latitudinale, vitesse_longitudinale]
        self.mmsi = mmsi
        self.liste_vecteurs = []
        self.liste_vecteurs_kalman=[]
        self.liste_vecteurs_kalman_prediction=[]
        self.covariances=[]
        #pour le filtre de Kalman
        self.kal_vecteur=vecteur[1:] #[latitude, longitude, vitesse_latitudinale, vitesse_longitudinale]
        self.kal_Q = 1e-1*np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]) #matrice de covariance du bruit du modèle physique
        #self.kal_Q = np.array([[0.1,0,0,0],[0,0.1,0,0],[0,0,1.0,0],[0,0,0,1.0]]) #matrice de covariance du bruit du modèle physique
        self.kal_H = np.identity(4) #matrice de transition entre le repère du capteur et le notre
        self.kal_R = 1e-5*np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]) #matrice de covariance liée aux bruits des capteurs (donné par le constructeur du capteur)
        #self.kal_R = np.array([[0.1,0,0,0],[0,0.1,0,0],[0,0,1.0,0],[0,0,0,1.0]]) #matrice de covariance liée aux bruits des capteurs (donné par le constructeur du capteur)
        self.kal_P = np.identity(4) #matrice de covariance de l'état estimé, arbitrairement grande au départ
        self.append(vecteur)
    def append(self, vecteur):
        self.liste_vecteurs.append(vecteur)
        if len(self.liste_vecteurs) >= 2:
            self.kalman()
            self.prediction()
    def kalman(self):
        #"prediction"
        vecteur1 = self.liste_vecteurs[-2]
        vecteur2 = self.liste_vecteurs[-1]
        delta_t = vecteur2[0]-vecteur1[0]
        F = np.array([[1,0,delta_t,0],[0,1,0,delta_t],[0,0,1,0],[0,0,0,1]]) #matrice représentant le modèle physique
        kal_vecteur_prime = np.dot(F, self.kal_vecteur)
        kal_P_prime = np.dot(np.dot(F, self.kal_P), F.transpose()) + self.kal_Q
        #mise a jour
        Y=vecteur2[1:]-np.dot(self.kal_H, kal_vecteur_prime)
        S=np.dot(np.dot(self.kal_H, kal_P_prime), self.kal_H.transpose()) + self.kal_R
        K=np.dot(np.dot(kal_P_prime, self.kal_H.transpose()), np.linalg.inv(S))#gain de Kalman
        self.kal_vecteur = kal_vecteur_prime + np.dot(K, Y)
        self.kal_P = np.dot((np.identity(4)-np.dot(K, self.kal_H)), kal_P_prime)
        self.liste_vecteurs_kalman.append(self.kal_vecteur)
        self.covariances.append(self.kal_P)
    def prediction(self):
        global delta_t
        F = np.array([[1,0,delta_t,0],[0,1,0,delta_t],[0,0,1,0],[0,0,0,1]]) #matrice représentant le modèle physique
        kal_vecteur_prime = np.dot(F, self.kal_vecteur)
        self.liste_vecteurs_kalman_prediction.append(kal_vecteur_prime)

        #kal_P_prime = np.dot(np.dot(F, self.kal_P), F.transpose()) + self.kal_Q
    def get_data(self):
        return self.liste_vecteurs, self.liste_vecteurs_kalman_prediction
    def covariance(self):
        return self.covariances

for frame in range(len_data):
    temps=frame #permet de recommencer l'animation au début plutôt que de planter #frame_offset : DEBUG
    if not data[temps] == []: #si il y a des données "reçues" durant la seconde représentée par la frame
        for infos in data[temps]:
            mmsi=infos[0] #le bateau est identifié par son mmsi dans le programme
            if mmsi in boats: #Si le bateau est déjà enregistré,
                boats[mmsi].append(infos[1]) #on met a jour sa position sa vitesse et son angle,
            else: #sinon,
                boats[mmsi]=Boat(mmsi,infos[1]) #on en crée un nouveau.
##Carte des risques
def repartition(U,t,mmsi):
    for k in boats.keys():
        if k!=mmsi:
            i=0
            boats_out[k],boats_kal_out[k]=boats[k].get_data()
            covariance=boats[k].covariance()
            X=np.array([boats_out[k][t][1],boats_out[k][t][2]])
            if np.abs(conversion(X-U)[0])<no//2 and np.abs(conversion(X-U)[1])<no//2:
                points=[]
                Y=np.array([boats_kal_out[k][t+1][0],boats_kal_out[k][t+1][1]])
                J=conversion(X-U)
                while i<36 and (np.abs(conversion(J)[0])<no//2-1 and np.abs(conversion(J)[1])<no//2-1):
                    
                    points.append((J+np.array([no//2,no//2])))
                    survoisin((J+np.array([no//2,no//2])),1/(covariance[t][0][0]))
                    J=conversion((i/35)*X+(1-i/35)*Y-U)
                    i+=1


def centre_bateau(mmsi,t):
    global no
    boats_out[mmsi],boats_kal_out[mmsi]=boats[mmsi].get_data()
    U=np.array([boats_out[mmsi][t][1],boats_out[mmsi][t][2]])
    V=np.array([boats_kal_out[mmsi][t+1][0],boats_kal_out[mmsi][t+1][1]])
    L=16/3*distance(V,U)
    grille(L,no)
    repartition(U,t,mmsi)

##Graphes
def comparaison_kalman(mmsi):
    g=delta_t/60
    boats_out[mmsi],boats_kal_out[mmsi]=boats[mmsi].get_data()
    TrX=[boats_out[mmsi][t][1] for t in range(len(boats_out[mmsi]))]
    TrY=[boats_out[mmsi][t][2] for t in range(len(boats_out[mmsi]))]
    KX=[boats_kal_out[mmsi][t][0] for t in range(1,len(boats_kal_out[mmsi])-int(g)+1)]
    KY=[boats_kal_out[mmsi][t][1] for t in range(1,len(boats_kal_out[mmsi])-int(g)+1)]
    plt.subplot(311)
    plt.plot(TrX,TrY,'ob',label='trajectoire')
    plt.plot(KX,KY,'.r',label='prédiction Kalman')
    plt.legend()

    D=[distance(np.array([boats_kal_out[mmsi][t][0],boats_kal_out[mmsi][t][1]]),np.array([boats_out[mmsi][int(t+g)][1],boats_out[mmsi][int(t+g)][2]])) for t in range(min(len(boats_kal_out[mmsi]),len(boats_out[mmsi]))-int(g)+1)]
    plt.subplot(312)
    plt.grid()
    plt.plot(D,label='distance Kalman/trajectoire')
    plt.legend()

    Dx=[(np.array([boats_kal_out[mmsi][t][0],boats_kal_out[mmsi][t][1]])-np.array([boats_out[mmsi][t+int(g)][1],boats_out[mmsi][t+int(g)][2]]))[0] for t in range(min(len(boats_kal_out[mmsi]),len(boats_out[mmsi]))-int(g)+1)]
    Dy=[(np.array([boats_kal_out[mmsi][t][0],boats_kal_out[mmsi][t][1]])-np.array([boats_out[mmsi][t+int(g)][1],boats_out[mmsi][t+int(g)][2]]))[1] for t in range(min(len(boats_kal_out[mmsi]),len(boats_out[mmsi]))-int(g)+1)]
    moyenne=[(Dx[0]+Dy[0])/2]
    for t in range(1,min(len(boats_kal_out[mmsi]),len(boats_out[mmsi]))-int(g)+1):
        moyenne.append((moyenne[t-1]*2*t+Dx[t]+Dy[t])/(2*(t+1)))
    plt.subplot(313)
    plt.grid()
    plt.plot(Dx,'--g',label='écart en x')
    plt.plot(Dy,':r',label='écart en y')
    plt.plot(moyenne,'-b',label="moyenne")
    plt.legend()

    plt.show()
## Résultats
testre="hr"
if testre=="r":
    comparaison_kalman("3337641")
else:
    centre_bateau("3337640",4)
    X = np.linspace(0,no,no)
    t = np.linspace(0,no,no)
    X, t = np.meshgrid(X, t, indexing = 'ij')

    plt.pcolor(X, t, M,cmap='plasma')
    plt.colorbar()
    plt.xlabel("Latitude")
    plt.ylabel("Longitude")
    #plt.show()
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

Gn = graphe(no)
sn = (30,50)
dn = (0,2)

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
#     n = (G)
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
    plt.plot(s[0],s[1], marker = "o", color = "g", label = "Origine")
    if p > 1:
        for k in range(1,p-1):
            c = chemin[k]
            plt.plot(c[0], c[1], marker = "x", color = "b")
        plt.plot(d[0],d[1], marker = "o", color = "r", label = "Destination")
    plt.legend()
    plt.show()

"""Gp = graphe(5)
Rp = [[0,0,0,0,0],
      [0,0,1,1,1],
      [0,1,2,3,2],
      [0,1,2,3,2],
      [5,0,1,1,1]]
sp = (0,0)
dp = (4,4)"""
affiche_trajet(Gn,M,sn,dn)