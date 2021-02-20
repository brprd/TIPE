input_file="D:/brieu/Documents/MP/TIPE/tipe/bdd_aleatoire.csv"
mmsi="3337641"
delta_t =120


import matplotlib.pyplot as plt
import numpy as np
import bib_ais as ais



def norme2(x):
    return np.sqrt(x[0]**2+x[1]**2)
def distance(x,y):
    return norme2(x-y)


boats=ais.calculs_bateaux(input_file)

g=delta_t/60
boats_out, boats_kal_out = boats[mmsi].get_data(), boats[mmsi].get_kal_data()
len_boats_out=len(boats_out)
len_boats_kal_out=len(boats_kal_out)
TrX=[boats_out[t][1] for t in range(len_boats_out)]
TrY=[boats_out[t][2] for t in range(len_boats_out)]
KX=[boats_kal_out[t][0] for t in range(1,len(boats_kal_out)-int(g)+1)]
KY=[boats_kal_out[t][1] for t in range(1,len(boats_kal_out)-int(g)+1)]
plt.subplot(311)
plt.plot(TrX,TrY,'ob',label='trajectoire')
plt.plot(KX,KY,'.r',label='prédiction Kalman')
plt.legend()

D=[distance(np.array([boats_kal_out[t][0],boats_kal_out[t][1]]),np.array([boats_out[int(t+g)][1],boats_out[int(t+g)][2]])) for t in range(min(len(boats_kal_out),len(boats_out))-int(g)+1)]
plt.subplot(312)
plt.grid()
plt.plot(D,label='distance Kalman/trajectoire')
plt.legend()

Dx=[(np.array([boats_kal_out[t][0],boats_kal_out[t][1]])
- np.array([boats_out[t+int(g)][1],boats_out[t+int(g)][2]]))[0] for t in range(min(len_boats_kal_out,len_boats_out)-int(g)+1)]

Dy=[(np.array([boats_kal_out[t][0],boats_kal_out[t][1]])
- np.array([boats_out[t+int(g)][1],boats_out[t+int(g)][2]]))[1] for t in range(min(len_boats_kal_out,len_boats_out)-int(g)+1)]

moyenne=[(Dx[0]+Dy[0])/2]
for t in range(1,min(len_boats_kal_out,len_boats_out)-int(g)+1):
    moyenne.append((moyenne[t-1]*2*t+Dx[t]+Dy[t])/(2*(t+1)))

plt.subplot(313)
plt.grid()
plt.plot(Dx,'--g',label='écart en x')
plt.plot(Dy,':r',label='écart en y')
plt.plot(moyenne,'-b',label="moyenne")
plt.legend()

plt.show()