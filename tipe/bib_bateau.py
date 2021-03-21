import numpy as np

#pour Boat_graph
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

#pour Bateau_commande
import time


def produit(ARRAY):#produit matriciel
    P = np.identity(4)
    for M in ARRAY:
        P = np.dot(P,M)
    return P

def gaussienne(x, y, A, varx, vary):
    return A*np.exp(-1/(2*varx)*x**2-1/(2*vary)*y**2)

class Boat:#bateau de base
    kal_Q = 1e-6*np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]) #matrice de covariance du bruit du modèle physique
    kal_R = 1e-7*np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]) #matricee de covariance liée aux bruits des capteurs (donné par le constructeur du capteur)
    delta_t_prediction=300#temps pour lequel doit s'effecteur la prédiction

    def __init__(self, mmsi, vecteur):#vecteur = [instant, latitude, longitude, vitesse_latitudinale, vitesse_longitudinale]
        self._mmsi = mmsi
        self._liste_vecteurs = []#avec cette liste, on garde l'historique des positions de chaque bateau
        self._liste_vecteurs_kal = []#liste des positions successives enregistrées par le filtre
        self._liste_vecteurs_kal_pred = []#liste des prédictions successives effectuées par le filtre
        self._liste_kal_cov = []#liste des covariances successives du filtre
        self._liste_kal_cov_pred = []#liste des covariances successives des predictions

        #pour le filtre de Kalman
        self._kal_vecteur=vecteur[1:] #[latitude, longitude, vitesse_latitudinale, vitesse_longitudinale]
        self._kal_P = np.identity(4) #matrice de covariance de l'état estimé, arbitrairement grande au départ
    def _prediction(self, delta_t):#la phase prédiction du filtre de Kalman
        F = np.array([[1,0,delta_t,0],[0,1,0,delta_t],[0,0,1,0],[0,0,0,1]]) #matrice représentant le modèle physique
        kal_vecteur_prime = produit((F, self._kal_vecteur))
        kal_P_prime = produit((F, self._kal_P, F.T)) + Boat.kal_Q
        return kal_vecteur_prime, kal_P_prime
    def _kalman(self):#le filtre
        #"__prediction"
        t1 = self._liste_vecteurs[-2][0]
        t2 = self._liste_vecteurs[-1][0]
        delta_t = t2-t1

        kal_vecteur_prime, kal_P_prime = self._prediction(delta_t)

        #"mise a jour"
        K=np.dot(kal_P_prime, np.linalg.inv(kal_P_prime + Boat.kal_R))#gain de Kalman optimal

        self._kal_vecteur = kal_vecteur_prime + produit((K, self._liste_vecteurs[-1][1:]-kal_vecteur_prime))
        self._kal_P = produit((np.identity(4)-K, kal_P_prime, np.identity(4)-K.T)) + produit((K, Boat.kal_R, np.transpose(K)))

        self._liste_vecteurs_kal.append(self._kal_vecteur)
        self._liste_kal_cov.append(self._kal_P)
    def _predire(self, t):#prédit et affiche la position du bateau au bout de t secondes
        kal_vecteur_prime, kal_P_prime = self._prediction(t)
        self._liste_vecteurs_kal_pred.append(kal_vecteur_prime)
        self._liste_kal_cov_pred.append(kal_P_prime)
        return kal_vecteur_prime, kal_P_prime
    def append(self, vecteur):#met à jour le bateau avec le vecteur passé en paramètre
        self._liste_vecteurs.append(vecteur)
        if len(self._liste_vecteurs) >= 2:#kalman utilise l'instant t+1
            self._kalman()#calcule et affiche la position estimée du bateau à l'instant même
            self._predire(Boat.delta_t_prediction)#prédit la position du bateau
    def get_data(self):
        return self._liste_vecteurs
    def get_kal_data(self):
        return self._liste_vecteurs_kal
    def get_kal_data_pred(self):
        return self._liste_vecteurs_kal
    def get_kal_cov(self):
        return self._liste_kal_cov
    def get_kal_cov_pred(self):
        return self._liste_kal_cov_pred


#redondances de code avec la classe Boat_graph, propreté à améliorer
class Boat_risque(Boat):#bateau qui met a jour la carte des risques.
    R=100
    delta_t_prediction_L = [x*50 for x in range(0,10)]
    def __init__(self, mmsi, vecteur, axes):#vecteur = [instant, latitude, longitude, vitesse_latitudinale, vitesse_longitudinale]

        Boat.__init__(self, mmsi, vecteur)

        self._dot, = axes.plot([],[], marker='o',color='red',markersize=0) #le point représentant la position fournie pas l'AIS sur la carte
        self._kal_dot, = axes.plot([],[], marker='x',color='green',markersize=0) #le point représentant l'estimation du filtre de Kalman


    def _prediction(self, delta_t):#la phase prédiction du filtre de Kalman
        F = np.array([[1,0,delta_t,0],[0,1,0,delta_t],[0,0,1,0],[0,0,0,1]]) #matrice représentant le modèle physique
        kal_vecteur_prime = produit((F, self._kal_vecteur))
        kal_P_prime = produit((F, self._kal_P, F.T)) + Boat.kal_Q
        return kal_vecteur_prime, kal_P_prime
    def _kalman(self):#le filtre
        Boat._kalman(self)
        self._kal_dot.set_data(self._kal_vecteur[1], self._kal_vecteur[0])
    def append(self, vecteur, M, scale, lat_min, lon_min):#met à jour le bateau avec le vecteur passé en paramètre
        Boat.append(self, vecteur)#self.__liste_vecteurs.append(vecteur)
        self._dot.set_data(vecteur[2], vecteur[1])#modifie la position AIS affichée
        if len(self._liste_vecteurs) >= 2:#kalman utilise l'instant t+1
            self._kalman()#calcule et affiche la position estimée du bateau à l'instant même
            predictions=[]
            for delta_t in Boat_risque.delta_t_prediction_L:
                predictions.append(self._predire(delta_t))#prédit la position du bateau
            self._risque_update(M, scale, predictions, lat_min, lon_min)
    def _risque_update(self, M, scale, predictions, lat_min, lon_min):
        R=Boat_risque.R

        for pred in predictions:

            vect = pred[0]
            COV = pred[1]

            varx, vary = COV[0,0], COV[1,1]

            x=np.linspace(-R,R, num=2*R)
            y=x
            x, y = np.meshgrid(x, y)

            posx, posy = round((vect[1] - lon_min)*scale), round((vect[0] - lat_min)*scale)
            MAT = gaussienne(x, y, 100, 5*10e1*varx, 5*10e1*vary)
            M[posy-R:posy+R, posx-R:posx+R] = M[posy-R:posy+R, posx-R:posx+R] + MAT


class Boat_graph(Boat):#bateau avec fonctionnalités graphiques
    def __init__(self, mmsi, vecteur, axes, figure):#vecteur = [instant, latitude, longitude, vitesse_latitudinale, vitesse_longitudinale]

        Boat.__init__(self, mmsi, vecteur)

        self._dot, = axes.plot([],[], marker='o',color='blue',markersize=5, picker=True) #le point représentant la position fournie pas l'AIS sur la carte
        self._kal_dot, = axes.plot([],[], marker='x',color='green',markersize=5) #le point représentant l'estimation du filtre de Kalman

        self._kal_line, = axes.plot([], [], color='green', markersize=5)
        self._kal_ellipse = Ellipse((0, 0), height=0, width=0, color='green', fill=False)#incertitude de position
        self._ellipse_p = Ellipse((0, 0), height=0, width=0, color='red', fill=False)#incertitude de prédiction
        #trace les deux ellipses
        axes.add_patch(self._kal_ellipse)
        axes.add_patch(self._ellipse_p)

        #pour pouvoir cliquer sur le bateau et en afficher les infos
        self._dot.set_pickradius(2.5)
        figure.canvas.mpl_connect('pick_event', self._onpick)

    def _prediction(self, delta_t):#la phase prédiction du filtre de Kalman
        F = np.array([[1,0,delta_t,0],[0,1,0,delta_t],[0,0,1,0],[0,0,0,1]]) #matrice représentant le modèle physique
        kal_vecteur_prime = produit((F, self._kal_vecteur))
        kal_P_prime = produit((F, self._kal_P, F.T)) + Boat.kal_Q
        return kal_vecteur_prime, kal_P_prime
    def _kalman(self):#le filtre
        Boat._kalman(self)
        self._kal_dot.set_data(self._kal_vecteur[1], self._kal_vecteur[0])
        self._kal_ellipse.height=4*np.sqrt(self._kal_P[0][0])
        self._kal_ellipse.width=4*np.sqrt(self._kal_P[1][1])
        self._kal_ellipse.center=(self._kal_vecteur[1], self._kal_vecteur[0])
    def append(self, vecteur):#met à jour le bateau avec le vecteur passé en paramètre
        Boat.append(self, vecteur)#self.__liste_vecteurs.append(vecteur)
        self._dot.set_data(vecteur[2], vecteur[1])#modifie la position AIS affichée
        if len(self._liste_vecteurs) >= 2:#kalman utilise l'instant t+1
            self._kalman()#calcule et affiche la position estimée du bateau à l'instant même
            kal_vecteur_prime, kal_P_prime = self._predire(Boat.delta_t_prediction)#prédit la position du bateau
            self._kal_line.set_data([self._liste_vecteurs[-1][2], kal_vecteur_prime[1]], [self._liste_vecteurs[-1][1], kal_vecteur_prime[0]])
            self._ellipse_p.height=4*np.sqrt(kal_P_prime[0][0])#2*(2 sigmas) fait la hauteur de l'ellipse
            self._ellipse_p.width=4*np.sqrt(kal_P_prime[1][1])
            self._ellipse_p.center=(kal_vecteur_prime[1], kal_vecteur_prime[0])

    def _onpick(self, event):#détecte le clic sur le bateau et en affiche les informations principales
        if event.artist == self._dot:
            print(20*"=")
            print("MMSI : " + str(self._mmsi))
            print("latitude : " + str(round(self._kal_vecteur[0], 4)) + " ± " + str(round(2*self._kal_P[0][0], 4)) + "°")
            print("longitude : " + str(round(self._kal_vecteur[1], 4)) + " ± " + str(round(2*self._kal_P[1][1], 4)) + "°")
            print("nombres de mesures reçues : " + str(len(self._liste_vecteurs)))


class Bateau_commande:
    def __init__(self, mmsi, latitude_initiale, longitude_initiale, axes):
        self._mmsi=mmsi
        self._dot, = axes.plot([],[], marker='o',color='purple',markersize=5) #le point représentant le bateau sur la carte
        instant=time.time()
        self._liste_vecteurs = [[instant, latitude_initiale, longitude_initiale, 0.0, 0.0]]
        self.calculer_position(axes)
    def calculer_position(self, axes):
        vecteur = self._liste_vecteurs[-1]
        nouvel_instant = time.time()
        delta_t = nouvel_instant - vecteur[0]
        self._liste_vecteurs.append([nouvel_instant, vecteur[1]+delta_t*vecteur[3], vecteur[2]+delta_t*vecteur[4], vecteur[3], vecteur[4]])
        self._dot.set_data(self._liste_vecteurs[-1][2], self._liste_vecteurs[-1][1])
        #afficher la trajectoire :
        axes.plot([self._liste_vecteurs[-2][2],self._liste_vecteurs[-1][2]], [self._liste_vecteurs[-2][1],self._liste_vecteurs[-1][1]], color='purple')
    def accelerer(self, accelerations, axes):
        acc_lat = 0.001#"accélération" latitudinale
        acc_lon = 0.001#"accélération" longitudinale
        self.calculer_position(axes)
        vecteur = self._liste_vecteurs.pop()
        self._liste_vecteurs.append([vecteur[0], vecteur[1], vecteur[2], vecteur[3]+acc_lat*accelerations[0], vecteur[4]+acc_lon*accelerations[1]])
        self.calculer_position(axes)