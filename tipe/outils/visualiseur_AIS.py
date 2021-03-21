input_file='../bdd_aleatoire.csv'
RT = 6311000 #rayon de la Terre (en m)

time_scale=1000 #time_scale=x => le temps s'écoule x fois plus vite qu'en vrai

#changer la localisation de l'étude :
# latitude_min=55
# latitude_max=65
# longitude_min=-150
# longitude_max=-145
latitude_min=-90
latitude_max=90
longitude_min=-180
longitude_max=180

#DEBUG
frame_offset=0 #temps en secondes


interval=int(1000/time_scale) #ne pas toucher, modifier time_scale au-dessus

import csv
import numpy as np
import matplotlib.pyplot as plt
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
        #course over ground (COG) : direction de déplacement du bateau (pas son orientation)
        COG=float(row[5])

        vitesse_lat=SOG*np.cos(COG)/RT
        vitesse_lon=(SOG*np.sin(COG))/(RT*np.cos(latitude)) #on considèrera que cos(lat) ne varie pas entre 2 mesures consécutives

        while temps > len(data)-1:
            data.append([]) #secondes pendant lesquelles rien est reçu
        data[temps].append([mmsi,[temps, latitude, longitude, vitesse_lat, vitesse_lon]])


len_data=len(data)

class Boat:
    def __init__(self, mmsi, vecteur): #vecteur = [instant, latitude, longitude, vitesse_latitudinale, vitesse_longitudinale]
        self.__mmsi = mmsi
        self.__liste_vecteurs = []
        self.__dot, = ax.plot([],[], marker='o',color='blue',markersize=5) #le point représentant le bateau sur la carte
        self.append(vecteur)
    def append(self, vecteur):
        self.__liste_vecteurs.append(vecteur)
        self.tracer()
    def tracer(self):
        self.__dot.set_data(self.__liste_vecteurs[-1][2], self.__liste_vecteurs[-1][1])


class Bateau_commande:
    def __init__(self, mmsi, latitude_initiale, longitude_initiale):
        self.__mmsi=mmsi
        self.__dot, = ax.plot([],[], marker='o',color='red',markersize=5) #le point représentant le bateau sur la carte
        instant=time.time()
        self.__liste_vecteurs = [[instant, latitude_initiale, longitude_initiale, 0.0, 0.0]]
        self.calculer_position()
    def calculer_position(self):
        vecteur = self.__liste_vecteurs[-1]
        nouvel_instant = time.time()
        delta_t = nouvel_instant - vecteur[0]
        self.__liste_vecteurs.append([nouvel_instant, vecteur[1]+delta_t*vecteur[3], vecteur[2]+delta_t*vecteur[4], vecteur[3], vecteur[4]])
        self.__dot.set_data(self.__liste_vecteurs[-1][2], self.__liste_vecteurs[-1][1])
        plt.plot([self.__liste_vecteurs[-2][2],self.__liste_vecteurs[-1][2]], [self.__liste_vecteurs[-2][1],self.__liste_vecteurs[-1][1]], color='red')
    def accelerer(self, accelerations):
        acc_lat = 0.001#"accélération" latitudinale : ce n'est pas une vraie accélération au sens où elle n'a pas de dimension temporelle
        acc_lon = 0.001
        self.calculer_position()
        vecteur = self.__liste_vecteurs.pop()
        self.__liste_vecteurs.append([vecteur[0], vecteur[1], vecteur[2], vecteur[3]+acc_lat*accelerations[0], vecteur[4]+acc_lon*accelerations[1]])
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
    #return [] #on est censé renvoyer les objets à tracer mais ça marche comme ça


fig = plt.figure() #il faut créer une figure pour l'animation (j'ai pas tout compris au fonctionnement de mathplot.pyplot. J'ai l'impression qu'il y a beaucoup d'objets qui font sensiblement la même chose
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