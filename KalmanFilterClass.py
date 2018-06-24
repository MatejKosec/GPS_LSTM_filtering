#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Matej Kosec
Standalone Kalman Filter implementation
"""
import scipy as sp
import functools

#Two helper function
#Chained dot product
def dot(listOfMatrices):
    return functools.reduce(sp.matmul,listOfMatrices)

#Return bKb^T
def quadratic_form(b,K):
    return functools.reduce(sp.matmul,[b,K,b.T])

from collections import namedtuple

Data = namedtuple('Data',['x','y','vx','vy','prx','pry'])
Data1D = namedtuple('Data1D',['x','vx','prx'])
Data3D = namedtuple('Data3D',['x','y','z','vx','vy','vz','prx','pry'])
DOPS = namedtuple('DOPS',['x','y','p'])

    
class LinearKalmanFilter1D(object):
    def __init__(self, F, H, P, Q, R, initialState):
        self.F = F
        self.H = H
        self.P = P
        self.Q = Q
        self.R = R
        self.hard_reset(initialState)
        
    
    def hard_reset(self, state):
        self.state = state
        self.data = Data1D([],[],[])
        self.append_data()
    
    def predict(self):
        #Use the system dynawmics to predict how well we are fittinig
        self.predictedState = dot([self.F, self.state])
        self.predictedP = quadratic_form(self.F, self.P)+ self.Q

    def update(self, measuredState):
        #Compte the residual between measurement and prediction
        self.prefitResidual = measuredState - dot([self.H, self.predictedState])
        
        #Compute the Klaman gain
        intermediate = sp.linalg.inv(self.R + quadratic_form(self.H, self.predictedP))
        self.Kt = dot([self.predictedP, self.H.T,intermediate])
        
        #Update the state
        self.state = self.predictedState + dot([self.Kt, self.prefitResidual])
        
        #Update covariance matrix
        self.P = quadratic_form(\
                sp.identity(self.Kt.shape[0]) - dot([self.Kt,self.H]), self.predictedP)\
                + quadratic_form(self.Kt, self.R)

        #Compute the postfit residual to see how well we are doing
        self.postfitResidual = measuredState - dot([self.H,self.state])
        
        #Store the results
        self.append_data()
        self.data.prx.append(self.prefitResidual[0])
        
    def append_data(self):
        self.data.x.append(self.state[0])
        self.data.vx.append(self.state[1])
        
    def process_data(self, data):
        for i in range(len(data.x)):
            m = sp.array([data.x[i],data.vx[i]]).T
            self.predict()
            self.update(m)
        
        return self.data



class LinearKalmanFilter2D(object):
    def __init__(self, F, H, P, Q, R, initialState):
        self.F = F
        self.H = H
        self.P = P
        self.Q = Q
        self.R = R
        self.hard_reset(initialState)
        
    
    def hard_reset(self, state):
        self.state = state
        self.data = Data([],[],[],[],[],[])
        self.append_data()
    
    def predict(self):
        #Use the system dynawmics to predict how well we are fittinig
        self.predictedState = dot([self.F, self.state])
        self.predictedP = quadratic_form(self.F, self.P)+ self.Q

    def update(self, measuredState):
        #Compte the residual between measurement and prediction
        self.prefitResidual = measuredState - dot([self.H, self.predictedState])
        
        #Compute the Klaman gain
        intermediate = sp.linalg.inv(self.R + quadratic_form(self.H, self.predictedP))
        self.Kt = dot([self.predictedP, self.H.T,intermediate])
        
        #Update the state
        self.state = self.predictedState + dot([self.Kt, self.prefitResidual])
        
        #Update covariance matrix
        self.P = quadratic_form(\
                sp.identity(self.Kt.shape[0]) - dot([self.Kt,self.H]), self.predictedP)\
                + quadratic_form(self.Kt, self.R)

        #Compute the postfit residual to see how well we are doing
        self.postfitResidual = measuredState - dot([self.H,self.state])
        
        #Store the results
        self.append_data()
        self.data.prx.append(self.prefitResidual[0])
        self.data.pry.append(self.prefitResidual[1])
        
    def append_data(self):
        self.data.x.append(self.state[0])
        self.data.y.append(self.state[1])
        self.data.vx.append(self.state[2])
        self.data.vy.append(self.state[3])
        
    def process_data(self, data):
        for i in range(len(data.x)):
            m = sp.array([data.x[i],data.y[i],data.vx[i],data.vy[i]]).T
            self.predict()
            self.update(m)
        
        return self.data
    
    
class LinearKalmanFilter3D(object):
    def __init__(self, F, H, P, Q, R, initialState):
        self.F = F
        self.H = H
        self.P = P
        self.Q = Q
        self.R = R
        self.hard_reset(initialState)
        
    
    def hard_reset(self, state):
        self.state = state
        self.data = Data3D([],[],[],[],[],[],[],[])
        self.append_data()
    
    def predict(self):
        #Use the system dynawmics to predict how well we are fittinig
        self.predictedState = dot([self.F, self.state])
        self.predictedP = quadratic_form(self.F, self.P)+ self.Q

    def update(self, measuredState):
        #Compte the residual between measurement and prediction
        self.prefitResidual = measuredState - dot([self.H, self.predictedState])
        
        #Compute the Klaman gain
        intermediate = sp.linalg.inv(self.R + quadratic_form(self.H, self.predictedP))
        self.Kt = dot([self.predictedP, self.H.T,intermediate])
        
        #Update the state
        self.state = self.predictedState + dot([self.Kt, self.prefitResidual])
        
        #Update covariance matrix
        self.P = quadratic_form(\
                sp.identity(self.Kt.shape[0]) - dot([self.Kt,self.H]), self.predictedP)\
                + quadratic_form(self.Kt, self.R)

        #Compute the postfit residual to see how well we are doing
        self.postfitResidual = measuredState - dot([self.H,self.state])
        
        #Store the results
        self.append_data()
        
    def append_data(self):
        self.data.x.append(self.state[0])
        self.data.y.append(self.state[1])
        self.data.z.append(self.state[2])
        self.data.vx.append(self.state[3])
        self.data.vy.append(self.state[4])
        self.data.vz.append(self.state[5])
        
    def process_data(self, data):
        for i in range(len(data.x)):
            m = sp.array([data.x[i],data.y[i],data.z[i],data.vx[i],data.vy[i],data.vz[i]]).T
            self.predict()
            self.update(m)
        
        return self.data
    
if __name__ == "__main__":
    deltaT = 0.1
    state0 = [0,0,0,0,0,0]
    state0[0] = 0
    state0[1] = 0
    P0     = sp.identity(6)*0.0001
    F0     = sp.array([[1, 0, 0, deltaT, 0, 0],\
                       [0, 1, 0, 0, deltaT, 0],\
                       [0, 0, 1, 0, 0, deltaT],\
                       [0, 0, 0, 1, 0, 0],\
                       [0, 0, 0, 0, 1, 0],
                       [0, 0, 0, 0, 0, 1]])
    H0     = sp.identity(6)
    Q0     = sp.diagflat([0.0001,0.0001,0.0001,0.1,0.1,0.1])
    R0     = sp.diagflat([6.0,6.0,6.0,0.5,0.5,0.5])
    filter3d = LinearKalmanFilter3D(F0, H0, P0, Q0, R0, state0)


#Linearized version 
class ExtendedKalmanFilter(object):
    def __init__(self,P, Q, R, initialState):
        self.P = P
        self.Q = Q
        self.R = R
        self.hard_reset(initialState)
            
    def hard_reset(self, state):
        self.state = state
        self.data = Data([],[],[],[],[],[])
        self.DOPS = DOPS([],[],[])
        self.append_data()
        
    def compute_f_and_F(self,previousState):
        #Predict the next state
        newState = sp.zeros(4)
        newState[0] = previousState[0] + previousState[2]*self.deltaT
        newState[1] = previousState[1] + previousState[3]*self.deltaT
        newState[2] = previousState[2]
        newState[3] = previousState[3]
        
        #Compute the jacobian
        F = sp.array([[1, 0, self.deltaT, 0],\
                   [0, 1, 0, self.deltaT],\
                   [0, 0, 1, 0],\
                   [0, 0, 0, 1]])
        return sp.reshape(newState,[1,4]),F
        
    def compute_h_and_H(self,state):
        numVal = self.beaconLocations.shape[0]
        tiledState = sp.tile(sp.reshape(state,[1,4]),[numVal,1])
        
        #Compute the distances and the range model
        distances = self.beaconLocations  - tiledState[:,:2]
        rangeModel = sp.linalg.norm(distances,axis=1,keepdims=True)
        
        #Compute the range rate model
        numerator = -distances[:,0]*tiledState[:,2] - distances[:,1]*tiledState[:,3]
        rangeRateModel = sp.reshape(numerator,rangeModel.shape)/rangeModel
        
        #Now to get the derivatives
        dRangeModel = sp.hstack([-distances,sp.zeros_like(distances)])
        dRangeModel = dRangeModel/rangeModel
        
        dRangeRateModel = (sp.hstack([tiledState[:,2:],-distances]) - rangeRateModel*dRangeModel)
        dRangeRateModel = dRangeRateModel/rangeModel
        
        return sp.vstack([rangeModel,rangeRateModel]),sp.vstack([dRangeModel,dRangeRateModel])
    
    def predict_and_update(self,measuredRange, measuredRangeRate):
        
        #First computed f and F
        f,F = self.compute_f_and_F(self.state) 
        
        #Use the system dynawmics to predict how well we are fittinig
        predictedState = f
        
        #We can now compute h and H
        h,H = self.compute_h_and_H(predictedState)
        
        #Update the distribution
        predictedP = quadratic_form(F, self.P)+ self.Q

        #Compte the residual between measurement and prediction
        measurements = sp.reshape(sp.hstack([measuredRange,measuredRangeRate]),[-1,1])
        prefitResidual = measurements - h
        
        #Compute the Klaman gain
        intermediate = sp.linalg.inv(self.R + quadratic_form(H, predictedP))
        Kt = dot([predictedP, H.T,intermediate])
        
        #Update the state
        self.state = sp.reshape(predictedState,[4,1]) + dot([Kt, prefitResidual])
        
        #Update covariance matrix
        self.P = quadratic_form(\
                sp.identity(Kt.shape[0]) - dot([Kt,H]), predictedP)\
                + quadratic_form(Kt, self.R)

        #Compute the postfit residual to see how well we are doing
        postfitResidual =  measurements - dot([H,self.state])
        
        #Store the results
        self.append_data()
        self.append_dops(H)#dot([H,self.P,H.T]))
        self.data.prx.append(prefitResidual[0])
        self.data.pry.append(prefitResidual[1])
    
    def append_dops(self,A):
        H = A[:A.shape[0]/2,:2]
        H = sp.linalg.inv(dot([H.T,H]))
        xdop2 = H[0,0]
        ydop2 = H[1,1]
        PDOP  = pow(xdop2 + ydop2,0.5)
        XDOP  = pow(xdop2,0.5)
        YDOP  = pow(ydop2,0.5)
        self.DOPS.x.append(XDOP)
        self.DOPS.y.append(YDOP)
        self.DOPS.p.append(PDOP)
    
    def get_dops(self):
        return self.DOPS
        
    def append_data(self):
        self.data.x.append(self.state[0])
        self.data.y.append(self.state[1])
        self.data.vx.append(self.state[2])
        self.data.vy.append(self.state[3])
    
    def process_data(self, data):
        self.deltaT  = sp.mean(data.t[1:]-data.t[:-1])
        self.Ntime = len(data.t)
        self.beaconLocations = data.beacon_locations
        #Resturcture the data
        self.beaconRanges = sp.array([[data.beacon_measurements[i].range[j] \
                                       for i in range(len(data.beacon_locations))\
                                       ] for j in range(self.Ntime)])
        self.beaconRangeRates = sp.array([[data.beacon_measurements[i].range_rates[j] \
                                       for i in range(len(data.beacon_locations))\
                                       ] for j in range(self.Ntime)])
        for i in range(len(data.t)):
            self.predict_and_update(self.beaconRanges[i],self.beaconRangeRates[i])
        
        return self.data