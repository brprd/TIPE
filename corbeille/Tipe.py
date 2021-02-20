input_file='ais_data.csv'

time_scale=1000 #time_scale=x => le temps s'écoule x fois plus vite qu'en vrai
echelle_vitesse=0.5 #echelle d'affichage du vecteur vitesse

#changer la localisation de l'étude :
latitude_min=55
latitude_max=65
longitude_min=-155
longitude_max=-145

#DEBUG
frame_offset=0 #temps en secondes

interval=int(1000/time_scale) #ne pas toucher, modifier time_scale au-dessus

import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.widgets as widget
import cartopy.crs as ccrs #à installer avec "pip install Cartopy-0.18.0-cp38-cp38-win_amd64.whl" (problèmes de dépendances si installé depuis les dépots de Python)


data=[] #contient les données du fichier d'entrée
len_data=0
boats={} #se remplit des bateaux détectés au fur et à mesure

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
            vitesse=float(row[4])*0.514 #1 noeud ~= 0.514 m/s
            #course over ground (COG) : direction de déplacement du bateau (pas son orientation)
            angle=float(row[5]) # /!\ vitesse exprimée en noeuds : 1 noeud ~= 0.514 m/s

            time=int(hours*3600+minuts*60+seconds)
            while time > len(data)-1:
                data.append([]) #secondes pendant lesquelles rien est reçu
            data[time].append([mmsi,lat,lon,vitesse,angle])

load_data(input_file, data)
len_data=len(data)

class Boat:
    def __init__(self, mmsi):
        global ax
        self.__mmsi = mmsi
        self.__dot, = ax.plot([],[], marker='o',color='blue',markersize=5) #le point représentant le bateau sur la carte
        self.__line, = ax.plot([],[],color='red',linewidth=1) #le point représentant le bateau sur la carte
    def maj(self, latitude, longitude, vitesse, angle):
        self.__lat=latitude
        self.__lon=longitude
        self.__vitesse=vitesse
        self.__angle=angle
        self.__dot.set_data(self.__lon, self.__lat)
        x = self.__vitesse*echelle_vitesse*np.sin(self.__angle)
        y = self.__vitesse*echelle_vitesse*np.cos(self.__angle)
        self.__line.set_data([self.__lon, self.__lon+x], [self.__lat, self.__lat+y])
        if self.__vitesse==0:
            self.__dot.set_color('gray')
            self.__dot.set_alpha(.6) #transparence
        else:
            self.__dot.set_color('blue')
            self.__dot.set_alpha(1) #transparence
        return [self.__dot,self.__line]



def update(frame):
    global data
    global boats
    time=(frame+frame_offset) % len_data #permet de recommencer l'animation au début plutôt que de planter #frame_offset : DEBUG
    plt.title((str(time//3600)+" h "+str((time//60)%60)+" min "+str(time%60)) + " ("+str(time)+" secondes)")
    if not data[time] == []: #si il y a des données "reçues" durant la seconde représentée par la frame
        for infos in data[time]:
            mmsi=infos[0] #le bateau est identifié par son mmsi dans le programme
            if mmsi in boats: #Si le bateau est déjà enregistré,
                boats[mmsi].maj(infos[1],infos[2],infos[3],infos[4]) #on met a jour sa position sa vitesse et son angle,
            else: #sinon,
                boats[mmsi]=Boat(mmsi) #on en crée un nouveau.
                boats[mmsi].maj(infos[1],infos[2],infos[3],infos[4]) #puis on lui attribue sa position sa vitesse et son angle
    return [] #on est censé renvoyer les objets à tracer mais ça marche comme ça

def init(): #rien de spécial à faire à l'initialisation de l'animation
    return []

fig = plt.figure() #il faut créer une figure pour l'animation (j'ai pas tout compris au fonctionnement de mathplot.pyplot. J'ai l'impression qu'il y a beaucoup d'objets qui font sensiblement la même chose
ax = plt.axes(projection=ccrs.PlateCarree(), autoscale_on=False, xlim=(longitude_min, longitude_max), ylim=(latitude_min, latitude_max))
ax.coastlines() #dessine la carte
ax.set_aspect('equal') #évite la déformation de la carte




ani = animation.FuncAnimation(fig, update, interval=interval, blit=True, init_func=init) #cette fonction permet d'appeler la fonction "update" tous les interval ms
plt.show() #affiche la fenetre