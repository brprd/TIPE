input_file='ais_data.csv'

time_scale=100 #time_scale=x => le temps s'écoule x fois plus vite qu'en vrai
echelle_vitesse=0.5 #echelle d'affichage du vecteur vitesse
rayon_terrestre=6371000

#changer la localisation de l'étude :
latitude_min=-90#55
latitude_max=90#65
longitude_min=-180#-155
longitude_max=180#-145

#DEBUG
frame_offset=0 #temps en secondes

interval=int(1000/time_scale) #ne pas toucher, modifier time_scale au-dessus

import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.widgets as widget
import time

import cartopy.crs as ccrs #à installer avec "pip install Cartopy-0.18.0-cp38-cp38-win_amd64.whl" (problèmes de dépendances si installé depuis les dépots de Python)

#Bases
data=[] #contient les données du fichier d'entrée
len_data=0
boats={} #se remplit des bateaux détectés au fur et à mesure

#Trajectoires
tempstemps=[]
latitude_mesure=[]
latitude_kalman=[]

#Ecarts
dreel=0 #distance réelle par rapport à l'origine
d=0 #distance Kalman par rapport à l'origine
M=[] #liste des écarts pour faire moyenne
MM=[] #liste des moyennes

interval_modelisation=50 #intervalle entre deux appels de la fonction 'onclick' par le timer (en ms)

class Boat:
    def __init__(self, mmsi, z, t):
        self.__mmsi = mmsi
        self.__Z = [z] #mesures AIS pour chaque instant listé dans self.__temps : latitude, longitude, latitude_point, longitude_point
        self.__temps = [t] #liste des temps auxquelles le bateau envoie des données AIS
        self.__X = z #état du bateau donné par le filtre (au début, on prend l'état donné par les mesures)
        self.__dot_mesures, = plt.plot([],[], marker='o',color='blue',markersize=5) #le point représentant le bateau sur la carte (coordonnées mesurées)
        self.__dot_kalman, = plt.plot([],[], marker='x',color='green',markersize=5) #le point représentant le bateau sur la carte (coordonnées mesurées)
        self.__compteur=0 #compteur pour next()
        self.__traj_mesures=[[],[]]
        self.__traj_kalman=[[],[]]
    def append(self, z, t):
        self.__Z.append(z)
        self.__temps.append(t)
    def centrer(self): #centre la carte sur le bateau lui-même
        donnes_instant=self.__Z[self.__compteur]
        plt.xlim(donnes_instant[1]-1, donnes_instant[1]+1)
        plt.ylim(donnes_instant[0]-1,donnes_instant[0]+1)
    def next(self):
        self.__compteur+=1
        self.plot_mesures(self.__compteur)
        self.prediction_et_maj(self.__compteur)
        return self.__temps[self.__compteur]
    def description(self):
        print("mmsi : " + self.__mmsi)
        print("temps : " + str(self.__temps))
        print("Z : " + str(self.__Z))
        print("compteur : " + str(self.__compteur))
    def organiser(self):
        new_Z=[]
        temps_sorted =sorted(self.__temps)
        for i in range(len(self.__temps)):
            new_Z.append(self.__Z[self.__temps.index(temps_sorted[i])])
        self.__temps=temps_sorted
        self.__Z=new_Z
    def plot_mesures(self, compteur):
        global dreel
        donnes_instant=self.__Z[compteur]
        self.__dot_mesures.set_data(donnes_instant[1], donnes_instant[0])

        self.__traj_mesures[1].append(donnes_instant[1])
        self.__traj_mesures[0].append(donnes_instant[0])
        plt.plot(self.__traj_mesures[1],self.__traj_mesures[0], color='blue')

        dreel=np.sqrt(donnes_instant[1]**2+ donnes_instant[0]**2)

    def prediction_et_maj(self, compteur):
        global d
        U = np.array([[0.1],[0.1],[0.001],[0.001]]) #matrice des incertitudes du modèle
        P = np.identity(4) #matrice de covariance arbitrairement grande
        Q = np.identity(4) #matrice de covariance incluant le bruit
        H = np.identity(4) #matrice de transition entre le repère du capteur et le notre
        R = 0.1*np.identity(4) #matrice de covariance liée aux bruits des capteurs (donné par le constructeur du capteur)

        #prediction
        delta_t = (self.__temps[compteur]-self.__temps[compteur-1])*0.001
        F = np.array([[1,0,delta_t,0],[0,1,0,delta_t],[0,0,1,0],[0,0,0,1]]) #matrice représentant le modèle physique
        Xprime = np.dot(F, self.__X) + U
        print("Xprime : " + str(Xprime))
        Pprime = np.dot(np.dot(F, P), F.transpose())+Q
        print("Pprime : " + str(Pprime))
        #mise a jour
        Y=self.__Z[compteur]-np.dot(H, Xprime)
        print("Y : " + str(Y))
        S=np.dot(np.dot(H, Pprime), H.transpose()) + R
        K=np.dot(np.dot(Pprime, H.transpose()), np.linalg.inv(S))#gain de Kalman
        print("K : " + str(K))
        self.__X = Xprime + np.dot(K, Y)
        P = np.dot((np.identity(4)-np.dot(K, H)), Pprime)
        self.__dot_kalman.set_data(self.__X[1], self.__X[0])
        d=np.sqrt(self.__X[1]**2+ self.__X[0]**2)

        self.__traj_kalman[1].append(self.__X[1])
        self.__traj_kalman[0].append(self.__X[0])
        plt.plot(self.__traj_kalman[1],self.__traj_kalman[0], color='green')

def load_data(input_file, data):
    with open(input_file, 'r', newline='') as input:
        csvreader = csv.reader(input, delimiter=',')
        for row in csvreader:
            mmsi=row[0]

            hours=int(row[1][11:13])
            minuts=int(row[1][14:16])
            seconds=int(row[1][17:19])

            lat=float(row[2])
            lon=float(row[3])

            #speed over ground (SOG) : vitesse par rapport au sol (pas à la mer)
            SOG=float(row[4])*0.514 #1 noeud ~= 0.514444 m/s
            #course over ground (COG) : direction de déplacement du bateau (pas son orientation)
            COG=float(row[5])*0.017 #1 deg ~= 0.01745329251994329576923690768489 rad

            lat_point=SOG*np.cos(COG)/rayon_terrestre
            lon_point=(SOG*np.sin(COG))/(rayon_terrestre*np.cos(lat)) #on considèrera que cos(lat) ne varie pas entre 2 mesures consécutives


            t=int(hours*3600+minuts*60+seconds)
            z=np.array([[lat], [lon], [lat_point], [lon_point]])
            if mmsi in boats:
                boats[mmsi].append(z,t)
            else:
                boats[mmsi]=Boat(mmsi, z, t)


    for keys in boats:
        boats[keys].organiser()

def ecart():
    ec=d[0]-dreel[0]
    M.append(float(ec))
    print("écart :"+str(ec))
def moyenne(M):
    som=0
    for i in M:
        som+=i
    return som/len(M)

def onclick(event):
    bateau=boats["351925000"]
    try :
        temps = bateau.next()
        bateau.centrer()
        ecart()
        print("Moyenne:"+str(moyenne(M)))
        MM.append(float(moyenne(M)))
        plt.title(temps)
    except IndexError:
        print("Fin de liste atteinte")
        timer.stop()

def graph(event):
    timer.stop()
    plt.clf()
    plt.cla()
    plt.close()
    plt.plot(np.linspace(0,len(MM),len(MM)),np.array(MM))
    plt.title("La moyenne des écarts en fonction des instants de mesure")
    ax=plt.axes()
    ax.set_ylabel('Moyenne des écarts (en m)')
    ax.set_xlabel("nombre d'instants de mesure")
    plt.show()

fig = plt.figure() #il faut créer une figure pour l'animation
ax = plt.axes(projection=ccrs.PlateCarree(), autoscale_on=False)
ax.coastlines() #dessine la carte
ax.set_aspect('equal') #évite la déformation de la carte

timer = fig.canvas.new_timer(interval=interval_modelisation)
timer.add_callback(onclick,'a')
timer.start()
coom_id=fig.canvas.mpl_connect('key_press_event',graph)

load_data(input_file, data)
len_data=len(data)

def update(frame):
    return []
def init():
    return []
ani = animation.FuncAnimation(fig, update, interval=interval, blit=True, init_func=init) #cette fonction permet d'appeler la fonction "update" tous les interval ms

plt.show() #affiche la fenetre

