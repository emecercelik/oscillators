## Matsuoka Oscillator 

## Some parameters makes this oscillator stable
## This can be useful for some joints
## Relationship between parameters for oscillation
## 1+tau/T<a<1+b
## 'Analysis of neural oscillator', Matsuoka
## We can restrict the output inside certain values
## by using g function and c parameter

import numpy as np
import matplotlib.pyplot as plt

h=1e-3
tau=0.25
T=0.5
a=2.5
b=3.5
c=1.5

x=[0.1,0.1]
v=[0.1,0.1]
y=[0.1,0.1]



g=lambda x:max(0,x)

yRec0=[]
yRec1=[]
stopTime=10000
for t in range(stopTime):
    x[0]=x[0]+h*(-x[0]+c-a*y[1]-b*v[0])/tau
    v[0]=v[0]+h*(-v[0]+y[0])/T
    y[0]=g(x[0])

    x[1]=x[1]+h*(-x[1]+c-a*y[0]-b*v[1])/tau
    v[1]=v[1]+h*(-v[1]+y[1])/T
    y[1]=g(x[1])

    yRec0.append(y[0])
    yRec1.append(y[1])
    xx=x[0]-x[1]
    vv=v[0]-v[1]
    XX=x[0]+x[1]
    VV=v[0]+v[1]

time=np.arange(0,stopTime*h,h)
plt.plot(time,yRec0,label='1st output')
plt.plot(time,yRec1,label='2nd output')
plt.title('Outputs of 2 neurons of Matsuoka Oscillator\n'+\
          'tau={0:.2f}, T={1:.2f}, a={2:.2f}, b={3:.2f}, c={4:.2f}'\
          .format(tau,T,a,b,c))
plt.xlabel('Time [sec]')
plt.ylabel('Value of outputs')
plt.legend()
plt.show()
