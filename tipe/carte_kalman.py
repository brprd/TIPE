input_file='bdd.csv'
RT = 6311000 #rayon de la Terre (en m)
pi=3.141592

time_scale=1000#time_scale=x => le temps s'écoule x fois plus vite qu'en vrai

#changer la localisation de l'étude :
latitude_min=55
latitude_max=65
longitude_min=-165
longitude_max=-145


interval=int(1000/time_scale) #ne pas toucher, modifier time_scale au-dessus
frame_offset=0

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

#produit matriciel
def produit(ARRAY):
    P = np.identity(4)
    for M in ARRAY:
        P = np.dot(P,M)
    return P

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
    kal_Q = 1e-9*np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]) #matrice de covariance du bruit du modèle physique
    kal_R = 1e-3*np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]) #matricee de covariance liée aux bruits des capteurs (donné par le constructeur du capteur)
    def __init__(self, mmsi, vecteur):#vecteur = [instant, latitude, longitude, vitesse_latitudinale, vitesse_longitudinale]
        self.__mmsi = mmsi
        self.__liste_vecteurs = []#avec cette liste, on garde l'historique des positions de chaque bateau
        self.__dot, = ax.plot([],[], marker='o',color='blue',markersize=5, picker=True) #le point représentant la position fournie pas l'AIS sur la carte
        self.__kal_dot, = ax.plot([],[], marker='x',color='green',markersize=5) #le point représentant l'estimation du filtre de Kalman

        #pour le filtre de Kalman
        self.__kal_vecteur=vecteur[1:] #[latitude, longitude, vitesse_latitudinale, vitesse_longitudinale]

        self.__kal_P = np.identity(4) #matrice de covariance de l'état estimé, arbitrairement grande au départ
        self.__kal_line, = ax.plot([], [], color='green', markersize=5)
        self.__kal_ellipse = Ellipse((self.__kal_vecteur[1], self.__kal_vecteur[0]), height=0, width=0, color='green', fill=False)#incertitude sur
        self.__ellipse_p = Ellipse((self.__kal_vecteur[1], self.__kal_vecteur[0]), height=0, width=0, color='red', fill=False)#ellipse de prédiction

        #trace les deux ellipses
        ax.add_patch(self.__kal_ellipse)
        ax.add_patch(self.__ellipse_p)

        #pour pouvoir cliquer sur le bateau et en afficher les infos
        self.__dot.set_pickradius(2.5)
        fig.canvas.mpl_connect('pick_event', self.__onpick)

        self.append(vecteur)
    def append(self, vecteur):#met à jour le bateau avec le vecteur passé en paramètre
        self.__liste_vecteurs.append(vecteur)
        self.__dot.set_data(self.__liste_vecteurs[-1][2], self.__liste_vecteurs[-1][1])#modifie la position AIS affichée
        if len(self.__liste_vecteurs) >= 2:
            self.__kalman()#calcule et affiche la position estimée du bateau à l'instant même
            self.__predire(900)#prédit et affiche la position du bateau dans 1/4 d'heure (900 secondes)
    def __prediction(self, delta_t):#la phase prédiction du filtre de Kalman
        F = np.array([[1,0,delta_t,0],[0,1,0,delta_t],[0,0,1,0],[0,0,0,1]]) #matrice représentant le modèle physique
        kal_vecteur_prime = produit((F, self.__kal_vecteur))
        kal_P_prime = produit((F, self.__kal_P, F.T)) + Boat.kal_Q
        return kal_vecteur_prime, kal_P_prime
    def __kalman(self):#le filtre
        #"__prediction"
        vecteur1 = self.__liste_vecteurs[-2]
        vecteur2 = self.__liste_vecteurs[-1]
        delta_t = vecteur2[0]-vecteur1[0]

        kal_vecteur_prime, kal_P_prime = self.__prediction(delta_t)

        #"mise a jour"
        K=np.dot(kal_P_prime, np.linalg.inv(kal_P_prime + Boat.kal_R))#gain de Kalman optimal
        self.__kal_vecteur = kal_vecteur_prime + produit((K, vecteur2[1:]-kal_vecteur_prime))
        self.__kal_P = produit((np.identity(4)-K, kal_P_prime, np.identity(4)-K.T)) + produit((K, Boat.kal_R, np.transpose(K)))

        self.__kal_dot.set_data(self.__kal_vecteur[1], self.__kal_vecteur[0])
        self.__kal_ellipse.height=4*np.sqrt(self.__kal_P[0][0])
        self.__kal_ellipse.width=4*np.sqrt(self.__kal_P[1][1])
        self.__kal_ellipse.center=(self.__kal_vecteur[1], self.__kal_vecteur[0])
    def __predire(self, t):#prédit et affiche la position du bateau au bout de t secondes
        kal_vecteur_prime, kal_P_prime = self.__prediction(t)

        self.__kal_line.set_data([self.__liste_vecteurs[-1][2], kal_vecteur_prime[1]], [self.__liste_vecteurs[-1][1], kal_vecteur_prime[0]])
        self.__ellipse_p.height=4*np.sqrt(kal_P_prime[0][0])
        self.__ellipse_p.width=4*np.sqrt(kal_P_prime[1][1])
        self.__ellipse_p.center=(kal_vecteur_prime[1], kal_vecteur_prime[0])
    def __onpick(self, event):#détecte le clic sur le bateau et en affiche les informations principales
        if event.artist == self.__dot:
            print(20*"=")
            print("MMSI : " + str(self.__mmsi))
            print("latitude : " + str(round(self.__kal_vecteur[0], 4)) + " ± " + str(round(2*self.__kal_P[0][0], 4)) + "°")
            print("longitude : " + str(round(self.__kal_vecteur[1], 4)) + " ± " + str(round(2*self.__kal_P[1][1], 4)) + "°")
            print("nombres de mesures reçues : " + str(len(self.__liste_vecteurs)))



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




class Bateau_commande:
    def __init__(self, mmsi, latitude_initiale, longitude_initiale):
        self.__mmsi=mmsi
        self.__dot, = ax.plot([],[], marker='o',color='purple',markersize=5) #le point représentant le bateau sur la carte
        instant=time.time()
        self.__liste_vecteurs = [[instant, latitude_initiale, longitude_initiale, 0.0, 0.0]]
        self.__calculer_position()
    def __calculer_position(self):
        vecteur = self.__liste_vecteurs[-1]
        nouvel_instant = time.time()
        delta_t = nouvel_instant - vecteur[0]
        self.__liste_vecteurs.append([nouvel_instant, vecteur[1]+delta_t*vecteur[3], vecteur[2]+delta_t*vecteur[4], vecteur[3], vecteur[4]])
        self.__dot.set_data(self.__liste_vecteurs[-1][2], self.__liste_vecteurs[-1][1])
        #afficher la trajectoire :
        #plt.plot([self.__liste_vecteurs[-2][2],self.__liste_vecteurs[-1][2]], [self.__liste_vecteurs[-2][1],self.__liste_vecteurs[-1][1]], color='purple')
    def accelerer(self, accelerations):
        acc_lat = 0.001#"accélération" latitudinale : ce n'est pas une vraie accélération au sens où elle n'a pas de dimension temporelle
        acc_lon = 0.001
        self.__calculer_position()
        vecteur = self.__liste_vecteurs.pop()
        self.__liste_vecteurs.append([vecteur[0], vecteur[1], vecteur[2], vecteur[3]+acc_lat*accelerations[0], vecteur[4]+acc_lon*accelerations[1]])
        self.__calculer_position()


def on_key(event):
    global bateau_commande
    if not bateau_commande and event.key=="+":
        bateau_commande = Bateau_commande("keke des plages", event.ydata, event.xdata)
    if bateau_commande:
        if event.key=="up":
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