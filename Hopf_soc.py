# "Pattern generators with sensory feedback for the control of
# quadruped locomotion", Ludovic Righetti and Auke Jan Ijspeert

import numpy as np
import matplotlib.pyplot as plt

h=1e-3
numOsc=4


alpha=5.
beta=50.
mu=1.
w_stance=10. # Changes the stance phase
w_swing=1.
b=10. # Changes the shape of periodic signal
F=300
feedback=1

# Coupling matrices for different gaits
gait=3
gaitType=['Trot','Pace','Bound','Walk']
if gait==0:
    K=np.array([[0,-1,-1,1],[-1,0,1,-1],[-1,1,0,-1],[1,-1,-1,0]]) #Trot
elif gait==1:
    K=np.array([[0,-1,1,-1],[-1,0,-1,1],[1,-1,0,-1],[-1,1,-1,0]]) #Pace
elif gait==2:
    K=np.array([[0,1,-1,-1],[1,0,-1,-1],[-1,-1,0,1],[-1,-1,1,0]]) #Bound
else:
    K=np.array([[0,-1,1,-1],[-1,0,-1,1],[-1,1,0,-1],[1,-1,-1,0]]) #Walk

x=np.zeros((numOsc,1))+np.random.rand(numOsc,1)
y=np.zeros((numOsc,1))+np.random.rand(numOsc,1)
u=np.zeros((numOsc,1))
r=np.sqrt(x**2+y**2)
w=w_stance/(np.exp(-b*y)+1)+w_swing/(np.exp(b*y)+1)

timeStop=1000
xRec=np.array(x)
yRec=np.array(y)
rRec=np.array(r)


for t in range(timeStop):
    if feedback==1:
        for ii in range(numOsc):
            if y[ii]<0.25*r[ii] and y[ii]>-0.25*r[ii]:
                u[ii]=-w[ii]*x[ii]-K.dot(y)[ii]
            elif (y[ii]>0. and x[ii]<0.) or (y[ii]<0. and x[ii]>0.):
                u[ii]=-np.sign(y[ii])*F
            else:
                u[ii]=0.
    x=x+h*(alpha*(mu-r**2)*x-w*y)
    y=y+h*(beta*(mu-r**2)*y+w*x+K.dot(y)+u)
    w=w_stance/(np.exp(-b*y)+1)+w_swing/(np.exp(b*y)+1)
    r=np.sqrt(x**2+y**2)
    xRec=np.hstack((xRec,x))
    yRec=np.hstack((yRec,y))
    rRec=np.hstack((rRec,r))



##############Plot##################
time=np.arange(0,timeStop+1)
pltxAxisStep=timeStop/10

####Plot1##############################
plt.figure(1,figsize=(15,10))
plt.suptitle('Hopf Oscillator\n'+\
          'h={0:.3f}, alpha={1:.2f}, beta={2:.2f},'.format(h,alpha,beta)+\
          ' mu={0:.2f}, w_stance={1:.2f}, w_swing={2:.2f}, b={3:.2f}'\
          .format(mu,w_stance,w_swing,b)+\
             '\nGait Type={0:s}'.format(gaitType[gait]))

plt.subplot(421)
plt.plot(time,xRec[0,:],'b',label='x1')
plt.xticks(np.arange(0,timeStop,pltxAxisStep))
plt.xlabel('Time step')
plt.ylabel('Value of state variables')
plt.legend()
plt.subplot(423)
plt.plot(time,xRec[1,:],'k',label='x2')
plt.xticks(np.arange(0,timeStop,pltxAxisStep))
plt.legend()
plt.subplot(425)
plt.plot(time,xRec[2,:],'g',label='x3')
plt.xticks(np.arange(0,timeStop,pltxAxisStep))
plt.legend()
plt.subplot(427)
plt.plot(time,xRec[3,:],'m',label='x4')
plt.xticks(np.arange(0,timeStop,pltxAxisStep))
plt.legend()

plt.xlabel('Time step')
plt.ylabel('Value of state variables')

plt.subplot(422)
plt.plot(time,yRec[0,:],'b',label='y1')
plt.xticks(np.arange(0,timeStop,pltxAxisStep))
plt.legend()
plt.subplot(424)
plt.plot(time,yRec[1,:],'k',label='y2')
plt.xticks(np.arange(0,timeStop,pltxAxisStep))
plt.legend()
plt.subplot(426)
plt.plot(time,yRec[2,:],'g',label='y3')
plt.xticks(np.arange(0,timeStop,pltxAxisStep))
plt.legend()
plt.subplot(428)
plt.plot(time,yRec[3,:],'m',label='y4')
plt.xticks(np.arange(0,timeStop,pltxAxisStep))
plt.legend()

####Plot2##############################
plt.figure(2,figsize=(15,10))
plt.suptitle('Hopf Oscillator\n'+\
          'h={0:.3f}, alpha={1:.2f}, beta={2:.2f},'.format(h,alpha,beta)+\
          ' mu={0:.2f}, w_stance={1:.2f}, w_swing={2:.2f}, b={3:.2f}'\
          .format(mu,w_stance,w_swing,b)+\
             '\nGait Type={0:s}'.format(gaitType[gait]))

plt.subplot(211)
plt.plot(time,xRec[0,:],'b',label='x1')
plt.plot(time,xRec[1,:],'k',label='x2')
plt.plot(time,xRec[2,:],'g',label='x3')
plt.plot(time,xRec[3,:],'m',label='x4')
plt.xticks(np.arange(0,timeStop,pltxAxisStep))
plt.xlabel('Time step')
plt.ylabel('Value of state variables')
plt.legend()

plt.subplot(212)
plt.plot(time,yRec[0,:],'b',label='y1')
plt.plot(time,yRec[1,:],'k',label='y2')
plt.plot(time,yRec[2,:],'g',label='y3')
plt.plot(time,yRec[3,:],'m',label='y4')
plt.xticks(np.arange(0,timeStop,pltxAxisStep))
plt.xlabel('Time step')
plt.ylabel('Value of state variables')
plt.legend()

####Plot3##############################
plt.figure(3)
plt.plot(rRec[0,:])
plt.plot(rRec[1,:])
plt.plot(rRec[2,:])
plt.plot(rRec[3,:])

plt.show()
