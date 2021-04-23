#! /usr/bin/env python3

from tensorflow import keras
from tensorflow.keras.layers import Dense
import tensorflow as tf
import pandas as pd
import numpy as np
import psi4

global atoms
atoms = ['O','C','H1','H2']

def loadCoords(file='Structs.xyz'):


    #Empty df for holding training examples
    df = pd.DataFrame({})
    
    with open(file,'r') as file:
        line = 'True'
        struct = pd.DataFrame({})
        h_index = 1  #Index of the H atom, so as to distinguish both atoms
        while line:
            line = file.readline()
            if not line: #If end of line, end loop
                break 

            if line[-5:].strip() == '.log':
                name = line[:-5]
                #Restart struct df
                struct = pd.DataFrame({})

            elif line[0]!= '\n': #If line not a name, then read structures
                line = line.split(' ')
                atom = line[0] #atom name

                #Distinguish between the two H atoms: rename as H1 and H2
                if atom == 'H':
                    atom = atom + str(h_index)
                    if h_index == 1: h_index += 1
                    else: h_index = 1

                #Read coordinates and append to struct df
                coords = pd.Series(line[1:],name=atom).astype(float)
                struct=struct.append(coords)

                if struct.shape[0] == 4:  #If reading atoms is done, do computations
                    struct = struct.loc[atoms] #Reorder atoms
                    df = df.append(pd.Series(struct.values.ravel(),name=name))
        return df

def HFenergy(row):
    psi4.set_memory('500 MB')

    #charge = +1, spin multipicity = 2
    h2o = psi4.geometry("""
    1 2
    O {} {} {}
    C {} {} {} 
    H {} {} {}
    H {} {} {}
    """.format(*row))
    
    psi4.set_options({'reference': 'uhf'})
    
    #Energy calculated using UHF/cc-pVDZ
    try:
        return psi4.energy('scf/cc-pvdz') 
    except:  #In case there's a convergence error or alike
        return 0.0


def loadCoords_IRC(file='ts3.irc'):
    #Empty df for holding training examples
    df = pd.DataFrame({})
    energyDF = []
    with open(file,'r') as file:
        line = 'True'
        struct = pd.DataFrame({})
        h_index = 1  #Index of the H atom, so as to distinguish both atoms
        while line:
            line = file.readline()
            if not line: #If end of file, end loop
                break 

            if line[:5] == 'Point':
                energy = float(line.split('=')[1].strip()[:17])*1000
                energyDF.append(energy)
                #Restart struct df
                struct = pd.DataFrame({})

            elif line[0]!= '\n' and line[0]!= '4': #If line not a name, then read structures
                line = line.split(' ')
                atom = line[0] #atom name

                #Distinguish between the two H atoms: rename as H1 and H2
                if atom == 'H':
                    atom = atom + str(h_index)
                    if h_index == 1: h_index += 1
                    else: h_index = 1

                #Read coordinates and append to struct df
                coords = pd.Series(line[1:],name=atom).astype(float)
                struct=struct.append(coords)

                if struct.shape[0] == 4:  #If reading atoms is done, do computations
                    struct = struct.loc[atoms] #Reorder atoms
                    df = df.append(pd.Series(struct.values.ravel(),name=0))
        df['energy'] = energyDF
        return df.reset_index(drop=True)

def getInput(xyz,forGrad=False):
    """xyz: vector containing xyz coords of each atom [Ox, Oy, Oz, Cx, ...]
    pass xyz as tf.Variable(xyz)"""
    xyzM = tf.reshape(xyz,(4,1,3))
    xyzT = tf.transpose(xyzM,perm=[1,0,2])

    DistMatr = tf.sqrt(tf.reduce_sum(tf.square(xyzM-xyzT),axis=2))

    mask = np.zeros((4,4))
    mask[np.triu_indices(4,1)] = 1

    inps = 1./tf.boolean_mask(DistMatr,mask)
    if forGrad: return tf.reshape(inps,(-1,6))
    else: return pd.Series(tf.reshape(inps,(-1,6)).numpy()[0])

def Grad(model,xyz):
    """Compute the gradients of E with respect to xyz coords"""
    xyz = tf.Variable(xyz) #Convert xyz np.array to tf.Variable
    with tf.GradientTape() as tape:  #Start recording operations
        X = getInput(xyz,forGrad=True)   #Convert to inputs
        e = model(X)        #Calculate energies
    return tape.gradient(e,xyz).numpy().flatten()   #Return gradient of E wrt xyz


class Trajectory:
    def __init__(self,X0,V0,b=0,Nsteps=1000,timeDelta=1e-17,Xdim=12,saveMolden=False,gradient=False,model=False):
        self.b = b
        self.Nsteps = Nsteps
        self.timeDelta = timeDelta
        self.Xdim = Xdim #Dimensionality of the space
        self.saveMolden = saveMolden
        self.gradient = gradient
        self.model = model
        
        #Simulation initial conditions
        self.X0 = X0  #Initial coords (given in Ang)
        self.V0 = V0  #Initial velocities (given in Ang/s)
   
        #Initialize data array  (3D tensor: quantities x timestep x dimension )
        data=np.zeros((3,self.Nsteps,self.Xdim))
        self.data = data
    
    def getAccel(self,t):
        m = np.array([15.999,12.011,1.008,1.008])#Masses O,C,H1,H2
        m = np.repeat(m,3)*1.660540199e-27   #kg
        forces = -self.gradient(self.model,self.data[0,t])*431.75024  #kg*Ang/s**2
        accel = forces/m   #Ang/s^2
        return accel.reshape(1,-1)
        
    def run(self): #Using velocity verlet algorithm
        dt=self.timeDelta
        steps=self.Nsteps
        data=self.data
        
        #data: x,v,a. Start trajectory: fist point
        data[:2,0] = np.concatenate([self.X0, self.V0],axis=0) #Set initial position and velocities
        data[2,0] = self.getAccel(0)            #Compute initial acceleration
        
        for i in range(steps-1):
            data[0,i+1]=data[0,i]+data[1,i]*dt + 0.5*data[2,i]*dt*dt #update X
            data[2,i+1]=self.getAccel(i+1)                        #update acceleration
            tempV0_5=data[1,i]+0.5*data[2,i]*dt           #velocity at t+dt/2 (temporal)
            data[1,i+1]=tempV0_5 + 0.5*data[2,i+1]*dt     #velocity at t+dt
        self.data=data
        print("Trajectory completed!")
        
        if self.saveMolden:
            self.saveTrajectMOLDEN()
    
    def getData(self):
        return self.data
    
    def saveTrajectMOLDEN(self):
        """Pass as input a trayectory object"""
        coords = self.data[0]
        with open(self.saveMolden, "w") as f:
            for i,struct in enumerate(coords):
                print("4\nPoint {}\nO {:.8f} {:.8f} {:.8f}\nC {:.8f} {:.8f} {:.8f}\nH {:.8f} {:.8f} {:.8f}\nH {:.8f} {:.8f} {:.8f}".format(i,*struct),file=f)

