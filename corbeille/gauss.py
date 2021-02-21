import numpy as np
p1=1
q1=1
p2=3/2
q2=1/2
def exp(t,p,q):
    return p*np.exp(-q*t**2)
    
T=np.linspace(-5,5,6000)
Y=[]

Ye=[]
for i in range(len(T)):
    Y.append(exp(T[i],1,1))
plt.plot(T,Y,label='q='+str(q1)+', p='+str(p1))

for i in range(len(T)):
    Ye.append(exp(T[i],np.pi/2,np.log(np.pi/2)))
plt.plot(T,Ye,label='q='+str(q2)+', p='+str(p2))
plt.legend()
plt.show()

x, t = np.meshgrid(x, t, indexing = 'ij')
ax.plot_surface(x, t, T, cmap='plasma')
