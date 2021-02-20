import numpy as np


def produit(ARRAY):#produit matriciel
    P = np.identity(4)
    for M in ARRAY:
        P = np.dot(P,M)
    return P

class Boat:
    kal_Q = 1e-9*np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]) #matrice de covariance du bruit du modèle physique
    kal_R = 1e-3*np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]) #matricee de covariance liée aux bruits des capteurs (donné par le constructeur du capteur)

    def __init__(self, mmsi, vecteur):#vecteur = [instant, latitude, longitude, vitesse_latitudinale, vitesse_longitudinale]
        self.__mmsi = mmsi
        self.__liste_vecteurs = []#avec cette liste, on garde l'historique des positions de chaque bateau
        self.__liste_vecteurs_kal = []#liste des positions successives enregistrées par le filtre
        self.__liste_kal_cov = []#liste des covariances successives du filtre

        #pour le filtre de Kalman
        self.__kal_vecteur=vecteur[1:] #[latitude, longitude, vitesse_latitudinale, vitesse_longitudinale]
        self.__kal_P = np.identity(4) #matrice de covariance de l'état estimé, arbitrairement grande au départ

        self.append(vecteur)
    def __prediction(self, delta_t):#la phase prédiction du filtre de Kalman
        F = np.array([[1,0,delta_t,0],[0,1,0,delta_t],[0,0,1,0],[0,0,0,1]]) #matrice représentant le modèle physique
        kal_vecteur_prime = produit((F, self.__kal_vecteur))
        kal_P_prime = produit((F, self.__kal_P, F.T)) + Boat.kal_Q
        return kal_vecteur_prime, kal_P_prime
    def __kalman(self):#le filtre
        #"__prediction"
        t1 = self.__liste_vecteurs[-2][0]
        t2 = self.__liste_vecteurs[-1][0]
        delta_t = t2-t1

        kal_vecteur_prime, kal_P_prime = self.__prediction(delta_t)

        #"mise a jour"
        K=np.dot(kal_P_prime, np.linalg.inv(kal_P_prime + Boat.kal_R))#gain de Kalman optimal
        self.__kal_vecteur = kal_vecteur_prime + produit((K, self.__liste_vecteurs[-1][1:]-kal_vecteur_prime))
        self.__kal_P = produit((np.identity(4)-K, kal_P_prime, np.identity(4)-K.T)) + produit((K, Boat.kal_R, np.transpose(K)))

        self.__liste_vecteurs_kal.append(self.__kal_vecteur)
        self.__liste_kal_cov.append(self.__kal_P)

    def append(self, vecteur):#met à jour le bateau avec le vecteur passé en paramètre
        self.__liste_vecteurs.append(vecteur)
        if len(self.__liste_vecteurs) >= 2:
            self.__kalman()#calcule et affiche la position estimée du bateau à l'instant même
    def get_data(self):
        return self.__liste_vecteurs
    def get_kal_data(self):
        return self.__liste_vecteurs_kal
    def get_kal_cov(self):
        return self.__liste_kal_cov
