input_file='bdd.csv'

#changer la localisation de l'étude :
latitude_min=55
latitude_max=65
longitude_min=-165
longitude_max=-145
# latitude_min=-90
# latitude_max=90
# longitude_min=-180
# longitude_max=180


time_scale=1000#time_scale=x => le temps s'écoule x fois plus vite qu'en vrai


frame_offset=0
interval=int(1000/time_scale) #ne pas toucher, modifier time_scale au-dessus


RT = 6311000 #rayon de la Terre (en m)
pi=3.141592


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cartopy.crs as ccrs #à installer avec "pip install Cartopy-0.18.0-cp38-cp38-win_amd64.whl" (problèmes de dépendances si installé depuis les dépots de Python)

import bib_ais as ais
import bib_bateau as bat

data=[] #contient les données du fichier d'entrée
boats={} #se remplit des bateaux détectés au fur et à mesure
bateau_commande=False


#charger données
data = ais.charger_ais(input_file)
len_data=len(data)


figure = plt.figure()#pour créer l'animation
axes = plt.axes(projection=ccrs.PlateCarree(), autoscale_on=False, xlim=(longitude_min, longitude_max), ylim=(latitude_min, latitude_max))
axes.coastlines() #dessine la carte
axes.set_aspect('equal') #évite la déformation de la carte

def on_key(event):
    global bateau_commande
    if not bateau_commande and event.key=="+":
        bateau_commande = bat.Bateau_commande("keke des plages", event.ydata, event.xdata, axes)
    if bateau_commande:
        if event.key=="up":
            bateau_commande.accelerer([1,0], axes)
        elif event.key=="down":
            bateau_commande.accelerer([-1,0], axes)
        elif event.key=="right":
            bateau_commande.accelerer([0,1], axes)
        elif event.key=="left":
            bateau_commande.accelerer([0,-1], axes)

figure.canvas.mpl_connect('key_press_event', on_key)


def update(frame):
    global boats
    temps=(frame+frame_offset) % len_data #permet de recommencer l'animation au début plutôt que de planter #frame_offset : DEBUG
    plt.title((str(temps//3600)+" h "+str((temps//60)%60)+" min "+str(temps%60)) + " ("+str(temps)+" secondes)")
    if not data[temps] == []: #si il y a des données "reçues" durant la seconde représentée par la frame
        for infos in data[temps]:
            mmsi=infos[0] #le bateau est identifié par son mmsi dans le programme
            if mmsi in boats: #Si le bateau est déjà enregistré,
                boats[mmsi].append(infos[1]) #on met a jour sa position, sa vitesse et son angle,
            else: #sinon,
                boats[mmsi]=bat.Boat_graph(mmsi, infos[1], axes, figure) #on en crée un nouveau
                boats[mmsi].append(infos[1])#on met a jour sa position, sa vitesse et son angle.
    if bateau_commande:
        bateau_commande.calculer_position(axes)


anim = animation.FuncAnimation(figure, update, interval=interval) #cette fonction permet d'appeler la fonction "update" tous les interval ms
plt.show() #affiche la fenetre