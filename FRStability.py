from math import inf
import numpy as np
from numpy.core.fromnumeric import transpose
from numpy.core.function_base import linspace
import numpy.polynomial.polynomial as poly
from numpy.polynomial import Legendre
from scipy.interpolate import lagrange
from scipy.special import eval_legendre
import matplotlib.pyplot as plt

k = 4
n = 6
tau = 0.105
dx = 10e-11

xi = np.linspace(-1,1,k+1)
thetaRange = np.linspace(-np.pi+dx,np.pi-dx,500)

def gL(x, c=0):
    ak = np.math.factorial(2*k)/(2**k*np.math.factorial(k)**2)
    etak = (c*(2*k+1)*(ak*np.math.factorial(k))**2)/2
    return ((-1)**k/2)*(eval_legendre(k,x)-(etak*eval_legendre(k-1,x)+eval_legendre(k+1,x))/(1+etak))

def centDiff(x1, x2, dx):
    return (x2-x1)/(2*dx)

D = np.zeros((k+1,k+1))
for i in range(k+1):
    temp = np.zeros(k+1)
    temp[i] = 1
    phi = np.flip(lagrange(xi,temp))
    phider = poly.polyder(phi)
    for j in range(k+1):
        D[j,i] = poly.polyval(xi[j],phider)

g = np.zeros((k+1,1))
for i in range(k+1):
    g[i] = centDiff(gL(xi[i]-dx),gL(xi[i]+dx),dx)

l = np.zeros((k+1,1))
r = np.zeros((k+1,1))
for i in range(k+1):
    temp = np.zeros(k+1)
    temp[i] = 1
    phi = np.flip(lagrange(xi,temp))
    l[i] = poly.polyval(-1,phi)
    r[i] = poly.polyval(1,phi)

Cneg1 = -2*g*np.transpose(r)
C0 = -2*(D-(g*np.transpose(l)))

L = C0
for i in range(n-2):
    L = np.concatenate((L,np.zeros((k+1,k+1))),1)
L = np.concatenate((L,Cneg1),1)
LRow = L
for i in range(n-1):
    LRow = np.roll(LRow,k+1,1)
    L = np.concatenate((L,LRow),0)
L = np.matrix(L)

#RK4
#M = np.eye(n*(k+1))+tau*L+(1/2)*(tau*L)**2+(1/6)*(tau*L)**3+(1/24)*(tau*L)**4

#RK45 Formula 1
M = np.eye(n*(k+1))+1*(tau*L)+0.5*(tau*L)**2+0.166666666666667*(tau*L)**3+0.0416666666666667*(tau*L)**4+0.0104166666666667*(tau*L)**5

#RK45 Formula 2
#M = np.eye(n*(k+1))+1*(tau*L)+0.5*(tau*L)**2+0.166666666666667*(tau*L)**3+0.0416666666666667*(tau*L)**4+0.00961538461538462*(tau*L)**5

#RK45 Sarafyan
#M = np.eye(n*(k+1))+1*(tau*L)+0.5*(tau*L)**2+0.166666666666667*(tau*L)**3+0.0416666666666667*(tau*L)**4


real = []
imag = []
vg = []
ET = []
x = []
for theta in thetaRange:
    N = np.zeros((k+1,k+1),dtype=complex)
    for i in range(n):
        N += M[((k+1)*(n-1)):((k+1)*n),((k+1)*i):((k+1)*(i+1))]*np.exp((i-n+1)*theta*1j)
    
    omega,v = np.linalg.eig(N)
    omegaBar = -np.log(omega)/(1j*tau)
    omegaBar = omegaBar.reshape((len(omegaBar),1))

    vPoly = np.zeros((k+1,k+1),dtype=complex)
    vLegendre = np.zeros((k+1,k+1),dtype=complex)
    for i in range(k+1):
        vPoly[:,i] = np.flip(lagrange(xi,v[:,i]))
        vLegendre[:,i] = Legendre.fit(linspace(-1,1),poly.polyval(linspace(-1,1),vPoly[:,i]),k).coef

    count = 0
    thetaTrue = np.zeros((k+1,1))+np.inf
    
    m = np.linspace(-k,k,2*k+1)
    thetaTrueMin = theta+m*2*np.pi
    for i in range(len(thetaTrueMin)):
            if thetaTrueMin[i]>(k+1)*np.pi:
                thetaTrueMin[i] = np.inf
            elif thetaTrueMin[i]<-(k+1)*np.pi:
                thetaTrueMin[i] = np.inf
        
    while count <= k:
        index = np.argmax(abs(vLegendre[count,:]))

        index2 = np.argmin(abs(thetaTrueMin))

        thetaTrue[index] = thetaTrueMin[index2]
        thetaTrueMin[index2] = inf

        vLegendre[:,index] = 0

        count += 1

    real.extend(omegaBar.real)
    imag.extend(omegaBar.imag)
    x.extend(thetaTrue)
    vg.extend(omegaBar.real)
    ET.extend(abs(omegaBar-thetaTrue))

results = np.concatenate((x,real,imag,vg,ET),1)
results = results[results[:,0].argsort()]
results[:,3] = np.gradient(results[:,3],results[:,0])
np.savetxt('results.txt',results)
print("Maximum Diffusion Error: "+str(max(results[:,2])))
plt.figure()
plt.plot(results[:,0],results[:,1],'o')
plt.figure()
plt.plot(results[:,0],results[:,2])
plt.figure()
plt.plot(results[:,0],results[:,3])
plt.figure()
plt.plot(results[:,0],results[:,4])
plt.yscale('log')
plt.xscale('log')
plt.show()