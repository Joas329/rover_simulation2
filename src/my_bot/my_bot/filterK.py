import csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
from matplotlib import animation


#kalman filter step by step implementation

#Defining an Identity Matrix

I = I = np.eye(6)

#state vector contains [x,y,heading,velocity,yaw rate, longitudinal acceleration]^T (transposed)

#define final matrices
filtered_x =[]
filtered_y = []
filtered_heading = []
filtered_vel = []
filtered_yawrate = []
filtered_longacc =[]
Zx = []
Zy = []
Px = []
Py = []
Pdx= []
Pdy= []
Pddx=[]
Pddy=[]
Pdv =[]
Kx = []
Ky = []
Kdx= []
Kdy= []
Kddx=[]
Kdv= []
dstate=[]

#sample rates for IMU and GPS
dt = 1.0/50.0
dtGPS = 1.0/10.0

#inital state

#x = np.matrix([[0.0,0.0,40.36,0.0,-18.21,0.23]]).T
x = np.matrix([[0.0, 0.0, (-252.2/180.0*np.pi+np.pi)%(2.0*np.pi) - np.pi, 2.42/3.6+0.001,-18.713/180.0*np.pi, 0.2647]]).T

U=float(np.cos(x[2])*x[3])
V=float(np.sin(x[2])*x[3])

#Initial Uncertainty (Initialize it with 0 when you are certain of the initial position)
P = np.diag([1000.0,1000.0,1000.0,1000.0,1000.0,1000.0])

#Process Noise Covariance Matrix Q

sGPS     = 0.5*8.8*dt**2  # assume 8.8m/s2 as maximum acceleration, forcing the vehicle
sCourse  = 0.1*dt # assume 0.1rad/s as maximum turn rate for the vehicle
sVelocity= 8.8*dt # assume 8.8m/s2 as maximum acceleration, forcing the vehicle
sYaw     = 1.0*dt # assume 1.0rad/s2 as the maximum turn rate acceleration for the vehicle
sAccel   = 0.5

Q = np.diag([sGPS**2, sGPS**2, sCourse**2, sVelocity**2, sYaw**2, sAccel**2])

#Measurement Noise Covaraince R
varGPS = 5.0 # Standard Deviation of GPS Measurement
varspeed = 3.0 # Variance of the speed measurement
varyaw = 0.1 # Variance of the yawrate measurement
varacc = 1.0 # Variance of the longitudinal Acceleration
R = np.diag([varGPS**2, varGPS**2, varspeed**2, varyaw**2, varacc**2])


#define the kalman function

    
def ekf(counter,GPS,yawrate,Z):
    #Kalman filter
     # Time Update (Prediction)
    # ========================
    # Project the state ahead
    # see "Dynamic Matrix"
    global x,P,Q,R
    if np.abs(yawrate[counter])<0.0001: # Driving straight
        x[4] = 0.0001
    x[0] = x[0] + (1 / x[4]**2) * ((x[3]*x[4] + x[5] * x[4] * dt) * \
        np.sin(x[2] + x[4]* dt) + x[5] * np.cos(x[2] + x[4] * dt) - x[3] *  \
        x[4] * np.sin(x[2]) - x[5] * np.cos(x[2]))
    x[1] = x[1] + (1 / x[4]**2) * ((-x[3]*x[4] - x[5] * x[4] * dt) * \
        np.cos(x[2] + x[4]* dt) + x[5] * np.sin(x[2] + x[4] * dt) + x[3] * \
        x[4] * np.cos(x[2]) - x[5] * np.sin(x[2]))
    x[2] = (x[2] + x[4] * dt + np.pi) % (2.0 * np.pi) - np.pi
    x[3] = x[3] + x[5] * dt 
    x[4] = x[4]
    x[5] = x[5]
    
    
    # Calculate the Jacobian of the Dynamic Matrix A
    # see "Calculate the Jacobian of the Dynamic Matrix with respect to the state vector"
    a13 = ((-x[4]*x[3]*np.cos(x[2]) + x[5]*np.sin(x[2]) - x[5]*np.sin(dt*x[4] + x[2]) + \
        (dt*x[4]*x[5] + x[4]*x[3])*np.cos(dt*x[4] + x[2]))/x[4]**2).item(0)

    a14 = ((-x[4]*np.sin(x[2]) + x[4]*np.sin(dt*x[4] + x[2]))/x[4]**2).item(0)

    a15 = ((-dt*x[5]*np.sin(dt*x[4] + x[2]) + dt*(dt*x[4]*x[5] + x[4]*x[3])* \
        np.cos(dt*x[4] + x[2]) - x[3]*np.sin(x[2]) + (dt*x[5] + x[3])* \
        np.sin(dt*x[4] + x[2]))/x[4]**2 - 2*(-x[4]*x[3]*np.sin(x[2]) - x[5]* \
        np.cos(x[2]) + x[5]*np.cos(dt*x[4] + x[2]) + (dt*x[4]*x[5] + x[4]*x[3])* \
        np.sin(dt*x[4] + x[2]))/x[4]**3).item(0)

    a16 = ((dt*x[4]*np.sin(dt*x[4] + x[2]) - np.cos(x[2]) + np.cos(dt * x[4] + x[2]))/x[4]**2).item(0)

    a23 = ((-x[4] * x[3] * np.sin(x[2]) - x[5] * np.cos(x[2]) + x[5] * np.cos(dt * x[4] + x[2]) - \
        (-dt * x[4]*x[5] - x[4] * x[3]) * np.sin(dt * x[4] + x[2])) / x[4]**2).item(0)
    a24 = ((x[4] * np.cos(x[2]) - x[4]*np.cos(dt*x[4] + x[2]))/x[4]**2).item(0)
    a25 = ((dt * x[5]*np.cos(dt*x[4] + x[2]) - dt * (-dt*x[4]*x[5] - x[4] * x[3]) * \
        np.sin(dt * x[4] + x[2]) + x[3]*np.cos(x[2]) + (-dt*x[5] - x[3])*np.cos(dt*x[4] + x[2]))/ \
        x[4]**2 - 2*(x[4]*x[3]*np.cos(x[2]) - x[5] * np.sin(x[2]) + x[5] * np.sin(dt*x[4] + x[2]) + \
        (-dt * x[4] * x[5] - x[4] * x[3])*np.cos(dt*x[4] + x[2]))/x[4]**3).item(0)
    a26 =  ((-dt*x[4]*np.cos(dt*x[4] + x[2]) - np.sin(x[2]) + np.sin(dt*x[4] + x[2]))/x[4]**2).item(0)
        
    JA = np.matrix([[1.0, 0.0, a13, a14, a15, a16],
                    [0.0, 1.0, a23, a24, a25, a26],
                    [0.0, 0.0, 1.0, 0.0, dt, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0, dt],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    
    
    # Project the error covariance ahead
    P = JA*P*JA.T + Q
    
    # Measurement Update (Correction)
    # ===============================
    # Measurement Function
    hx = np.matrix([[float(x[0])],
                    [float(x[1])],
                    [float(x[3])],
                    [float(x[4])],
                    [float(x[5])]])

    if GPS[counter]: # with 10Hz, every 5th step
        JH = np.matrix([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    else: # every other step
        JH = np.matrix([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])        
    
    S = JH*P*JH.T + R
    K = (P*JH.T) * np.linalg.inv(S)

    # Update the estimate via
    #Z = measurements[:,counter].reshape(JH.shape[0],1)
    y = Z - (hx)                         # Innovation or Residual
    x = x + (K*y)

    # Update the error covariance
    P = (I - (K*JH))*P


    # Save states for Plotting
    filtered_x.append(float(x[0]))
    filtered_y.append(float(x[1]))
    filtered_heading.append(float(x[2]))
    filtered_vel.append(float(x[3]))
    filtered_yawrate.append(float(x[4]))
    filtered_longacc.append(float(x[5]))
    Zx.append(float(Z[0]))
    Zy.append(float(Z[1]))    
    Px.append(float(P[0,0]))
    Py.append(float(P[1,1]))
    Pdx.append(float(P[2,2]))
    Pdy.append(float(P[3,3]))
    Pddx.append(float(P[4,4]))
    Pdv.append(float(P[5,5]))
    Kx.append(float(K[0,0]))
    Ky.append(float(K[1,0]))
    Kdx.append(float(K[2,0]))
    Kdy.append(float(K[3,0]))
    Kddx.append(float(K[4,0]))
    Kdv.append(float(K[5,0]))
  
  
CM = 0
#define meassurement matrices
mx =[]
my = []
mheading = []
mvel = []
myawrate = []
mlongacc =[]
longitude =[]
latitude =[]
altitude = []    




def main():
    
    while True:
        data = pd.read_csv('2014-03-26-000-Data.csv')
        if(data.empty == False):
            global CM
            #append data to corresponding meassurement matrices
            longitude.append(data['longitude'][CM])
            latitude.append(data['latitude'][CM])
            altitude.append(data['altitude'][CM])
            #course
            temp = data['course'][CM]
            temp = ((-temp+90)/180.0*np.pi+np.pi)%(2.0*np.pi) - np.pi
            mheading.append(temp)
            #velocity
            temp2 = data['speed'][CM]
            temp2 = temp2 / 3.6 + 0.001
            mvel.append(temp2)
            #yaw rate
            temp3 = data['yawrate'][CM]
            temp3 = temp3/180.0 * np.pi
            myawrate.append(temp3)
            #longitudinal acceleration
            mlongacc.append(data['ax'][CM])
            
            #transform latitude and longitude into mx and my
            RadiusEarth = 6378388.0 # m
            #First iteration
            if(CM ==0):
                arc= 2.0*np.pi*(RadiusEarth+altitude[CM])/360.0 # m/°
                dx = arc * np.cos(latitude[CM]*np.pi/180.0) * np.hstack((0.0, np.diff(longitude))) # in m
                dy = arc * np.hstack((0.0, np.diff(latitude))) # in m
            else:
                
                arc= 2.0*np.pi*(RadiusEarth+altitude[CM])/360.0 # m/°
                dx = arc * np.cos(latitude[CM]*np.pi/180.0) * np.hstack((0.0, np.diff(longitude))) # in m
                dy = arc * np.hstack((0.0, np.diff(latitude))) # in m
            global mx,my
            mx = np.cumsum(dx)
            my = np.cumsum(dy)
            
            ds = np.sqrt(dx**2+dy**2)
            
            GPS=(ds!=0.0).astype('bool') # GPS Trigger for Kalman Filter
            
            #make Z measurement vector for ekf calculation
            Z = np.matrix([[mx[CM],my[CM],mvel[CM],myawrate[CM],mlongacc[CM]]]).T
            
            ekf(CM, GPS,myawrate,Z)
            CM = CM +1
            
    

         
            if(CM == 300):
                break
            
main()







#plt.plot(mx,my, label = "meassured position") 
#plt.plot(filtered_x, filtered_y, label = "filtered position") 
fig = plt.figure(figsize=(6,6))
plt.step(range(300),mx, label ="x measured")
plt.step(range(300),my, label = "y measured")
plt.legend()
fig = plt.figure(figsize=(6,6))
plt.step(range(300),filtered_x, label ="x filtered")
plt.step(range(300),filtered_y, label = "y filtered")
plt.legend()
fig = plt.figure(figsize=(6,6)) 
plt.step(range(300),mheading, label ="Course measured")
plt.step(range(300),filtered_heading, label = "Course filtered")
plt.legend()
fig = plt.figure(figsize=(6,6))
plt.step(range(300),mvel, label ="velocity measured")
plt.step(range(300),filtered_vel, label = "vel filtered")
plt.legend()
fig = plt.figure(figsize=(6,6))
plt.step(range(300),myawrate, label ="yaw rate measured")
plt.step(range(300),filtered_yawrate, label = "yaw rate filtered")
plt.legend()
fig = plt.figure(figsize=(6,6))
plt.step(range(300),mlongacc, label ="long acc measured")
plt.step(range(300),filtered_longacc, label = "long acc filtered")
plt.legend()




plt.tight_layout()
plt.show()

