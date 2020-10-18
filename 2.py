import numpy as np
import matplotlib.pyplot as plt

def norme2(x):
    return np.sqrt(x[0]**2+x[1]**2)

def distance(x,y):
    return norme2(x-y)

def p(t,t0,t1,x0,x1,vx0,vx1):
    a,b,c,d=polynome3((t-t0)/(t1-t0))
    p=a*x0+b*vx0*(t1-t0)+c*x1+d*vx1*(t1-t0)
    return p
def conv_temps(T):
    L=[]
    for i in T:
            hours=int(i[11:13])
            minuts=int(i[14:16])
            seconds=int(i[17:19])
            ins=int(hours*3600+minuts*60+seconds)
            L.append(ins)
    return L


def polynome3(x):
    return (2*x**3-3*x**2+1,x**3-2*x**2+x,-2*x**3+3*x**2,x**3-x**2)
V=[0,3,4,6,7]
X=[59.44068,59.44105,59.44207,59.44368,59.44554]
Y=[-151.72107,-151.72191,-151.72321,-151.72344,-151.72309]
T=["2018-12-31T01:36:19","2018-12-31T01:37:28","2018-12-31T01:38:32","2018-12-31T01:39:35","2018-12-31T01:40:36"]
t=conv_temps(T)

def trajectoire(X,Vx,Y,Vy,t):
    Dx=[]
    Dy=[]
    temps=[]
    for i in range(1,len(t)):
        T=np.linspace(t[i-1],t[i],15,endpoint=True)
        for time in T:
            Dx.append(p(time,t[i-1],t[i],X[i-1],X[i],Vx[i-1],Vx[i]))
            temps.append(time)
            Dy.append(p(time,t[i-1],t[i],Y[i-1],Y[i],Vy[i-1],Vy[i]))
    return Dx,Dy,temps
    
X,Y,T= trajectoire(X,V,Y,V,t)
print(Y)
print(Y)
plt.plot(X,Y)
plt.show()