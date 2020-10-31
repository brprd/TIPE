input_file='bdd.csv'
RT = 6311000 #rayon de la Terre (en m)
pi=3.141592

time_scale=1000 #time_scale=x => le temps s'écoule x fois plus vite qu'en vrai

#changer la localisation de l'étude :
latitude_min=55
latitude_max=65
longitude_min=-165
longitude_max=-145
# latitude_min=57
# latitude_max=59
# longitude_min=-152
# longitude_max=-150

#DEBUG
frame_offset=0 #temps en secondes


interval=int(1000/time_scale) #ne pas toucher, modifier time_scale au-dessus

import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.animation as animation
import matplotlib.widgets as widget
import cartopy.crs as ccrs #à installer avec "pip install Cartopy-0.18.0-cp38-cp38-win_amd64.whl" (problèmes de dépendances si installé depuis les dépots de Python)
import time

data=[] #contient les données du fichier d'entrée
boats={} #se remplit des bateaux détectés au fur et à mesure
bateau_commande=False

#charger données
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
        self.dot, = ax.plot([],[], marker='o',color='blue',markersize=5) #le point représentant le bateau sur la carte
        self.kal_dot, = ax.plot([],[], marker='x',color='green',markersize=5) #le point représentant le bateau sur la carte

        #pour le filtre de Kalman
        self.kal_vecteur=vecteur[1:] #[latitude, longitude, vitesse_latitudinale, vitesse_longitudinale]
        self.kal_Q = np.array([[0.00001,0,0,0],[0,0.00001,0,0],[0,0,1.0,0],[0,0,0,1.0]]) #matrice de covariance du bruit du modèle physique
        self.kal_H = np.identity(4) #matrice de transition entre le repère du capteur et le notre
        self.kal_R = np.array([[0.00001,0,0,0],[0,0.00001,0,0],[0,0,1.0,0],[0,0,0,1.0]]) #matrice de covariance liée aux bruits des capteurs (donné par le constructeur du capteur)
        self.kal_P = np.identity(4) #matrice de covariance de l'état estimé, arbitrairement grande au départ
        self.kal_line, = ax.plot([], [], color='green', markersize=5)
        self.kal_ellipse = Ellipse((self.kal_vecteur[1], self.kal_vecteur[0]), height=0, width=0, color='green', fill=False)
        ax.add_patch(self.kal_ellipse)

        self.append(vecteur)
    def append(self, vecteur):
        self.liste_vecteurs.append(vecteur)
        self.tracer()
        if len(self.liste_vecteurs) >= 2:
            self.kalman()
            self.prediction()
    def tracer(self):
        self.dot.set_data(self.liste_vecteurs[-1][2], self.liste_vecteurs[-1][1])
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

        self.kal_dot.set_data(self.kal_vecteur[1], self.kal_vecteur[0])
        self.kal_ellipse.height=self.kal_P[0][0]
        self.kal_ellipse.width=self.kal_P[1][1]
        self.kal_ellipse.center=(self.kal_vecteur[1], self.kal_vecteur[0])
    def prediction(self):
        delta_t = 3600
        F = np.array([[1,0,delta_t,0],[0,1,0,delta_t],[0,0,1,0],[0,0,0,1]]) #matrice représentant le modèle physique
        kal_vecteur_prime = np.dot(F, self.kal_vecteur)
        #kal_P_prime = np.dot(np.dot(F, self.kal_P), F.transpose()) + self.kal_Q
        self.kal_line.set_data([self.liste_vecteurs[-1][2], kal_vecteur_prime[1]], [self.liste_vecteurs[-1][1], kal_vecteur_prime[0]])



class Bateau_commande:
    def __init__(self, mmsi, latitude_initiale, longitude_initiale):
        self.mmsi=mmsi
        self.dot, = ax.plot([],[], marker='o',color='red',markersize=5) #le point représentant le bateau sur la carte
        instant=time.time()
        self.liste_vecteurs = [[instant, latitude_initiale, longitude_initiale, 0.0, 0.0]]
        self.calculer_position()
    def calculer_position(self):
        vecteur = self.liste_vecteurs[-1]
        nouvel_instant = time.time()
        delta_t = nouvel_instant - vecteur[0]
        self.liste_vecteurs.append([nouvel_instant, vecteur[1]+delta_t*vecteur[3], vecteur[2]+delta_t*vecteur[4], vecteur[3], vecteur[4]])
        self.dot.set_data(self.liste_vecteurs[-1][2], self.liste_vecteurs[-1][1])
        plt.plot([self.liste_vecteurs[-2][2],self.liste_vecteurs[-1][2]], [self.liste_vecteurs[-2][1],self.liste_vecteurs[-1][1]], color='red')
    def accelerer(self, accelerations):
        acc_lat = 0.001#"accélération" latitudinale : ce n'est pas une vraie accélération au sens où elle n'a pas de dimension temporelle
        acc_lon = 0.001
        self.calculer_position()
        vecteur = self.liste_vecteurs.pop()
        self.liste_vecteurs.append([vecteur[0], vecteur[1], vecteur[2], vecteur[3]+acc_lat*accelerations[0], vecteur[4]+acc_lon*accelerations[1]])
        self.calculer_position()


def update(frame):
    global data
    global boats
    temps=(frame+frame_offset) % len_data #permet de recommencer l'animation au début plutôt que de planter #frame_offset : DEBUG
    plt.title((str(temps//3600)+" h "+str((temps//60)%60)+" min "+str(temps%60)) + " ("+str(temps)+" secondes)")
    if not data[temps] == []: #si il y a des données "reçues" durant la seconde représentée par la frame
        for infos in data[temps]:
            mmsi=infos[0] #le bateau est identifié par son mmsi dans le programme
            if mmsi in boats: #Si le bateau est déjà enregistré,
                boats[mmsi].append(infos[1]) #on met a jour sa position sa vitesse et son angle,
            else: #sinon,
                boats[mmsi]=Boat(mmsi,infos[1]) #on en crée un nouveau.
    if bateau_commande:
        bateau_commande.calculer_position()


fig = plt.figure()#pour créer l'animation
ax = plt.axes(projection=ccrs.PlateCarree(), autoscale_on=False, xlim=(longitude_min, longitude_max), ylim=(latitude_min, latitude_max))
ax.coastlines() #dessine la carte
ax.set_aspect('equal') #évite la déformation de la carte


def on_key(event):
    global bateau_commande
    if not bateau_commande and event.key=="+":
        bateau_commande = Bateau_commande("keke des plages", event.ydata, event.xdata)
    elif event.key=="up":
        bateau_commande.accelerer([1,0])
    elif event.key=="down":
        bateau_commande.accelerer([-1,0])
    elif event.key=="right":
        bateau_commande.accelerer([0,1])
    elif event.key=="left":
        bateau_commande.accelerer([0,-1])


cid = fig.canvas.mpl_connect('key_press_event', on_key)


ani = animation.FuncAnimation(fig, update, interval=interval) #cette fonction permet d'appeler la fonction "update" tous les interval ms
plt.show() #affiche la fenetre