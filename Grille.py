import csv
import numpy as np
import time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

input_file="D:/brieu/Documents/MP/TIPE/bbd_aleatoire.csv"
RT = 6311000 #rayon de la Terre (en m)
pi=np.pi
n="h"

time_scale=1000#time_scale=x => le temps s'écoule x fois plus vite qu'en vrai

interval=int(1000/time_scale) #ne pas toucher, modifier time_scale au-dessus



data=[] #contient les données du fichier d'entrée
boats={} #se remplit des bateaux détectés au fur et à mesure
boats_out={}
boats_kal_out={}

pas,L,no,M=0, 0, 100,np.zeros(0)

def norme2(x):
    return np.sqrt(x[0]**2+x[1]**2)

def distance(x,y):
    return norme2(x-y)

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

"""def survoisin(a,b,G):
    global no
    global M
    print(a,b)
    N=np.zeros((no,no))
    N[a,b]=G
    for i in range(1,min(G+1,no+1)):
        for j in range(max(0,a-i),min(no,a+i+1)):
            try :
                if N[j,b+i]==0 and G-(2**i)>0:
                    N[j,b+i]=G-(2**i)
            except IndexError:
                None
            if b-i>=0:
                if N[j,b-i]==0  and G-(2**i)>0:
                    N[j,b-i]=G-(2**i)
        for j in range(max(0,b-i),min(no,b+i)):
            try:
                if N[a+i,j]==0  and G-(2**i)>0:
                    N[a+i,j]=G-(2**i)
            except IndexError:
                None
            if a-i>=0:
                if N[a-i,j]==0  and G-(2**i)>0:
                    N[a-i,j]=G-(2**i)
    M+=N"""


def survoisin(J, p):
    global no
    global M
    q=0.5
    N=np.zeros((no,no))
    N[int(J[0]),int(J[1])]=p
    for i in range(no):
        for j in range(no):
            N[i,j]=p*np.exp(-q*((J[0]-i)**2+(J[1]-j)**2))
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
        #pour le filtre de Kalman
        self.kal_vecteur=vecteur[1:] #[latitude, longitude, vitesse_latitudinale, vitesse_longitudinale]
        self.kal_Q = np.array([[0.00001,0,0,0],[0,0.00001,0,0],[0,0,0.0000005,0],[0,0,0,0.0000005]]) #matrice de covariance du bruit du modèle physique
        #self.kal_Q = np.array([[0.1,0,0,0],[0,0.1,0,0],[0,0,1.0,0],[0,0,0,1.0]]) #matrice de covariance du bruit du modèle physique
        self.kal_H = np.identity(4) #matrice de transition entre le repère du capteur et le notre
        self.kal_R = np.array([[0.00001,0,0,0],[0,0.00001,0,0],[0,0,0.0000005,0],[0,0,0,0.0000005]]) #matrice de covariance liée aux bruits des capteurs (donné par le constructeur du capteur)
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
    def prediction(self):
        delta_t = 3600
        F = np.array([[1,0,delta_t,0],[0,1,0,delta_t],[0,0,1,0],[0,0,0,1]]) #matrice représentant le modèle physique
        kal_vecteur_prime = np.dot(F, self.kal_vecteur)
        #kal_P_prime = np.dot(np.dot(F, self.kal_P), F.transpose()) + self.kal_Q
    def get_data(self):
        return self.liste_vecteurs, self.liste_vecteurs_kalman


for frame in range(len_data):
    temps=frame #permet de recommencer l'animation au début plutôt que de planter #frame_offset : DEBUG
    if not data[temps] == []: #si il y a des données "reçues" durant la seconde représentée par la frame
        for infos in data[temps]:
            mmsi=infos[0] #le bateau est identifié par son mmsi dans le programme
            if mmsi in boats: #Si le bateau est déjà enregistré,
                boats[mmsi].append(infos[1]) #on met a jour sa position sa vitesse et son angle,
            else: #sinon,
                boats[mmsi]=Boat(mmsi,infos[1]) #on en crée un nouveau.

def repartition(U,t):
    for k in boats.keys():
        i=0
        boats_out[k],boats_kal_out[k]=boats[k].get_data()
        X=np.array([boats_out[k][t][1],boats_out[k][t][2]])
        if np.abs(conversion(X-U)[0])<no//2 and np.abs(conversion(X-U)[1])<no//2:
            print("yes")
            points=[]
            Y=np.array([boats_kal_out[k][t+1][0],boats_kal_out[k][t+1][1]])
            J=conversion(X-U)
            while i<36 and (np.abs(conversion(J)[0])<no//2 and np.abs(conversion(J)[1])<no//2):
                print(J)
                points.append((J+np.array([50,50])))
                survoisin((J+np.array([50,50])),1)
                J=conversion((i/35)*X+(1-i/35)*Y-U)
                i+=1


def centre_bateau(mmsi,t):
    global no
    boats_out[mmsi],boats_kal_out[mmsi]=boats[mmsi].get_data()
    U=np.array([boats_out[mmsi][t][1],boats_out[mmsi][t][2]])
    print(U)
    V=np.array([boats_kal_out[mmsi][t+1][0],boats_kal_out[mmsi][t+1][1]])
    print(V)
    print(distance(U,V))
    L=16/3*distance(V,U)
    grille(L,no)
    repartition(U,t)

def comparaison_kalman(mmsi):
    boats_out[mmsi],boats_kal_out[mmsi]=boats[mmsi].get_data()
    TrX=[boats_out[mmsi][t][1] for t in range(len(boats_out[mmsi]))]
    TrY=[boats_out[mmsi][t][2] for t in range(len(boats_out[mmsi]))]
    KX=[boats_kal_out[mmsi][t][0] for t in range(len(boats_kal_out[mmsi]))]
    KY=[boats_kal_out[mmsi][t][1] for t in range(len(boats_kal_out[mmsi]))]
    plt.subplot(311)
    plt.plot(TrX,TrY,'ob',label='trajectoire')
    plt.plot(KX,KY,'.r',label='prédiction Kalman')
    plt.legend()

    D=[distance(np.array([boats_kal_out[mmsi][t][0],boats_kal_out[mmsi][t][1]]),np.array([boats_out[mmsi][t+1][1],boats_out[mmsi][t+1][2]])) for t in range(min(len(boats_kal_out[mmsi]),len(boats_out[mmsi])))]
    plt.subplot(312)
    plt.grid()
    plt.plot(D,label='distance Kalman/trajectoire')
    plt.legend()

    Dx=[(np.array([boats_kal_out[mmsi][t][0],boats_kal_out[mmsi][t][1]])-np.array([boats_out[mmsi][t+1][1],boats_out[mmsi][t+1][2]]))[0] for t in range(min(len(boats_kal_out[mmsi]),len(boats_out[mmsi])))]
    Dy=[(np.array([boats_kal_out[mmsi][t][0],boats_kal_out[mmsi][t][1]])-np.array([boats_out[mmsi][t+1][1],boats_out[mmsi][t+1][2]]))[1] for t in range(min(len(boats_kal_out[mmsi]),len(boats_out[mmsi])))]
    moyenne=[(Dx[0]+Dy[0])/2]
    for t in range(1,min(len(boats_kal_out[mmsi]),len(boats_out[mmsi]))):
        moyenne.append((moyenne[t-1]*2*t+Dx[t]+Dy[t])/(2*(t+1)))
    plt.subplot(313)
    plt.grid()
    plt.plot(Dx,'--g',label='écart en x')
    plt.plot(Dy,':r',label='écart en y')
    plt.plot(moyenne,'-b',label="moyenne")
    plt.legend()

    plt.show()

#comparaison_kalman("3337665")

centre_bateau("3337640",4)
ax = Axes3D(plt.figure())
X = np.linspace(0,L,no)
t = np.linspace(0,L,no)
X, t = np.meshgrid(X, t, indexing = 'ij')
ax.plot_surface(X, t, M, cmap='plasma')
x=np.linspace(0,L,no)
y=np.linspace(0,L,no)



plt.show()