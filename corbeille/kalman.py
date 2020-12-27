input_file='ais_data.csv'

time_scale=50 #time_scale=x => le temps s'écoule x fois plus vite qu'en vrai
rayon_terrestre=6371000

interval=int(1000/time_scale) #ne pas toucher, modifier time_scale au-dessus

import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import cartopy.crs as ccrs #à installer avec "pip install Cartopy-0.18.0-cp38-cp38-win_amd64.whl" (problèmes de dépendances si installé depuis les dépots de Python)

#Bases
data=[] #contient les données du fichier d'entrée
len_data=0
boats={} #se remplit des bateaux détectés au fur et à mesure

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

        #pour le filtre
        self.__X = z #état du bateau donné par le filtre (au début, on prend l'état donné par les mesures)
        self.__P = np.identity(4) #matrice de covariance de l'état estimé, arbitrairement grande au départ

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
        plt.xlim(donnes_instant[1]-5, donnes_instant[1]+5)
        plt.ylim(donnes_instant[0]-5,donnes_instant[0]+5)
    def next(self):
        self.__compteur+=1
        self.plot_mesures(self.__compteur)
        self.kalman(self.__compteur)
        return self.__temps[self.__compteur]
    def description(self):
        print("mmsi : " + self.__mmsi)
        print("temps : " + str(self.__temps))
        print("Z : " + str(self.__Z))
        print("X : " + str(self.__X))
        print("P : " + str(self.__P))
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

    def kalman(self, compteur):
        global d
        Q = np.array([[0.1,0,0,0],[0,0.1,0,0],[0,0,0.00001,0],[0,0,0,0.00001]]) #matrice de covariance du bruit du modèle physique
        H = np.identity(4) #matrice de transition entre le repère du capteur et le notre
        R = Q#np.identity(4) #matrice de covariance liée aux bruits des capteurs (donné par le constructeur du capteur)

        #prediction
        delta_t = (self.__temps[compteur]-self.__temps[compteur-1])*0.001
        F = np.array([[1,0,delta_t,0],[0,1,0,delta_t],[0,0,1,0],[0,0,0,1]]) #matrice représentant le modèle physique
        Xprime = np.dot(F, self.__X)
        Pprime = np.dot(np.dot(F, self.__P), F.transpose()) + Q
        #mise a jour
        Y=self.__Z[compteur]-np.dot(H, Xprime)
        S=np.dot(np.dot(H, Pprime), H.transpose()) + R
        K=np.dot(np.dot(Pprime, H.transpose()), np.linalg.inv(S))#gain de Kalman
        self.__X = Xprime + np.dot(K, Y)
        self.__P = np.dot((np.identity(4)-np.dot(K, H)), Pprime)

        self.__dot_kalman.set_data(self.__X[1], self.__X[0])
        self.__traj_kalman[1].append(self.__X[1])
        self.__traj_kalman[0].append(self.__X[0])
        plt.plot(self.__traj_kalman[1],self.__traj_kalman[0], color='green')

        d=np.sqrt(self.__X[1]**2+ self.__X[0]**2)
    def get_pos(self):
        donnes_instant=self.__Z[self.__compteur]
        return [donnes_instant[1],donnes_instant[0]]

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
    #les données contenues dans le fichier étant désordonnées, il faut remettre les infos dans l'ordre chronologique
    for keys in boats:
        boats[keys].organiser()

def ecart():
    ec=d[0]-dreel[0]
    M.append(float(ec))
def moyenne(M):
    som=0
    for i in M:
        som+=i
    return som/len(M)

def centrer(liste_bateaux):
    e=5
    latitudes=[]
    longitudes=[]
    for bateau in liste_bateaux:
        latitudes.append(bateau.get_pos()[0])
        longitudes.append(bateau.get_pos()[1])
    moyenne_lon, = moyenne(longitudes)
    moyenne_lat, = moyenne(latitudes)
    plt.xlim(moyenne_lon-e, moyenne_lon+e)
    plt.ylim(moyenne_lat-e, moyenne_lat+e)

    print("----------------")
    print(longitudes)
    print(moyenne_lon)

def update(frame):
    try:
        bateau1=boats["369701000"]
        bateau2=boats["351925000"]
        bateau3=boats["338205000"]
        bateau4=boats["367010000"]
        try :
            temps = bateau1.next()
            bateau2.next()
            bateau3.next()
            bateau4.next()
            bateau1.centrer()
            #centrer([bateau1, bateau2])
            ecart()
            MM.append(float(moyenne(M)))
            plt.title(temps)
        except IndexError:
            print("Fin de liste atteinte")
            anim.event_source.stop()
    except IndexError:
        print("Bateau introuvable")
    return []

def graph(event):
    plt.clf()
    plt.cla()
    plt.close()
    plt.plot(np.linspace(0,len(M),len(M)),np.array(M))
    plt.title("Les écarts à chaque instant de mesure")
    ax.set_ylabel('Écarts (en °)')
    ax.set_xlabel("Nombre d'instants de mesure")
    plt.show()
    print(moyenne(M))

fig = plt.figure() #il faut créer une figure pour l'animation
ax = plt.axes(projection=ccrs.PlateCarree(), autoscale_on=False)
ax.coastlines() #dessine la carte
ax.set_aspect('equal') #évite la déformation de la carte

coom_id=fig.canvas.mpl_connect('key_press_event',graph)

anim = animation.FuncAnimation(fig, update, interval=interval) #cette fonction permet d'appeler la fonction "update" tous les interval ms


load_data(input_file, data)
len_data=len(data)


plt.show() #affiche la fenetre
