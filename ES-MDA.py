# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 12:35:46 2019

@author: iha
"""

"""
This is an attempt to make simulate ES-MDA based on a paper Analysis of the Performance of Ensemble-based Assimilation of Production and Seismic Data by Alexandre Anoze Emerick 
https://www.researchgate.net/profile/Alexandre_Emerick/publication/292213980_Analysis_of_the_Performance_of_Ensemble-based_Assimilation_of_Production_and_Seismic_Data/links/5a878fe3aca272017e5abf36/Analysis-of-the-Performance-of-Ensemble-based-Assimilation-of-Production-and-Seismic-Data.pdf)
"""

#------------------------------------------------INTRODUCTIONS----------------------------------------------------------#
#Import libraries

import numpy as np
import scipy.linalg as sp
import matplotlib.pyplot as plt

#Constants

mLength = 10 #the length of the parameter m
dLength = 5 #the length of the data d
nEnsemble = 100 #the number of ensembles

#stdevM = 1. #standard deviation of the parameter


alpha_max = 1000.
maxIter = 40

#Declaring variables

stdevD = np.ones(dLength)*1. #standard deviation of the data
stdevM = np.ones(mLength)*1.

mInit = np.zeros(mLength) #Initial ensemble
mAnswer = np.zeros(mLength) #True parameter values
mPrior = np.zeros([mLength, nEnsemble]) #Prior ensemble
mPred = np.zeros([mLength, nEnsemble]) #Predicted ensemble
mAverage = np.zeros(mLength)
dAverage = np.zeros(dLength)
dPrior = np.zeros([dLength, nEnsemble]) #Prior ensemble

d = np.zeros([dLength, nEnsemble]) #Forecasted data
obsData = np.zeros([dLength, nEnsemble]) #Observed data --> true data + measurement noise
dAnswer = np.zeros(dLength) #True data values
alpha = np.zeros(nEnsemble)
z = np.zeros([dLength, nEnsemble])

deltaM = np.zeros([mLength, nEnsemble])
deltaD = np.zeros([dLength, nEnsemble])
ddMD = np.zeros(nEnsemble)
ddDD = np.zeros(nEnsemble)

for p in range(maxIter):
    alpha[p] = (2**(maxIter-p))

#Generating the initial ensemble
for i in range(mLength):
#    mInit[i] = np.random.uniform(0,1)*1.
#    mInit[i] = np.random.uniform(0,1)*0.005
#    mInit[i] = i**2 + 1
    mInit[i] = 1/(i + 1)


#--------------------------------------------------FUNCTIONS-------------------------------------------------------------#
#Calculating the observation based on the parameter. This is function d = g(m)
#the non-linear functions obsData = gFunctions(parameters, dLength)
def gFunctions(parameters, dLength):
    predData = np.zeros(dLength)
    
    mSum = np.sum(parameters)
    for i in range(dLength):    
        predData[i] = mSum/(i + 1)
        
    return predData
   
#In the reservoir, the function is much more complex and needs a reservoir simulator to solve it (Eclipse, Intersect, etc.)


#----------------------------------------------THE 'ANSWERS'-------------------------------------------------------------#
    
for i in range(mLength):
#    mAnswer[i] = i**3 + i**2 + 5
#    mAnswer[i] = i**2 + 10*i + 5
#    mAnswer[i] = np.sin(i*np.pi/6)
#    mAnswer[i] = np.cos(i*np.pi/6)
#    mAnswer[i] = 1/(i**2 + 3)
    mAnswer[i] = np.exp(i/10)

dAnswer[:] = gFunctions(mAnswer[:], dLength)

stdevD = np.diag(dAnswer*0.1)
stdevM = np.multiply(mAnswer,0.05)

#------------------------------------------------POPULATING ENSEMBLE-----------------------------------------------------#
#Populate ensemble 

#Perturb the parameter
for i in range(mLength):
    mPrior[i,:] = np.random.normal(mInit[i], np.abs(stdevM[i]), nEnsemble)
    
m = mPrior #Initial ensemble

    
#Calculate prediction
for j in range(nEnsemble):
    dPrior[:,j] = gFunctions(mPrior[:,j], dLength)
    d[:,j] = gFunctions(mPrior[:,j], dLength)
    


##Perturb the data
#for i in range(dLength):
#    z[i,:] = np.random.normal(0, stdevD[i,i], nEnsemble)

#dInit = dPrior #Initial ensemble
#covarianceD = np.cov(d)

#------------------------------------------MAIN LOOP STARTS HERE---------------------------------------------------------#


for p in range(maxIter):
    #Get data
#    for j in range(nEnsemble):
#        obsData[:,j] = dAnswer[:] + (alpha[p]**(0.5))*(np.matmul(sp.fractional_matrix_power(stdevD,1/2),(z[:,j])))
    
    for i in range(dLength):
        obsData[i,:] = np.random.normal(dAnswer[i], np.abs(stdevD[i,i]), nEnsemble)

    
        
    #Calculate Average and Covariance MD and Covariance DD
    for i in range(mLength):
        summationM = np.sum(mPrior[i,:])
        mAverage[i] = (1/nEnsemble)*summationM
    
    for i in range(dLength):
        summationD = np.sum(dPrior[i,:])
        dAverage[i] = (1/nEnsemble)*summationD
    
    ddMD = 0.
    ddDD = 0.
    for j in range(nEnsemble):
        deltaM[:,j] = mPrior[:,j] - mAverage[:]
        deltaD[:,j] = dPrior[:,j] - dAverage[:]
        
        #This should be a matrix
        ddMD += np.outer(deltaM[:,j],deltaD[:,j])
        ddDD += np.outer(deltaD[:,j],deltaD[:,j])

    covarianceMD = ddMD / (nEnsemble - 1.)
    covarianceDD = ddDD / (nEnsemble - 1.)

    
#-----------------------------------------------UPDATE--------------------------------------------------------------------#

    for j in range(nEnsemble):
        dummyMat = np.matmul(covarianceMD,np.linalg.inv(covarianceDD + alpha[p]*stdevD)) 
        dummyVec = obsData[:,j] - dPrior[:,j]
        mPred[:,j] = mPrior[:,j] + np.matmul(dummyMat,dummyVec)

    mPrior = mPred
    
    #Calculate prediction
    for j in range(nEnsemble):
        dPrior[:,j] = gFunctions(mPrior[:,j], dLength)
        
    meanP = np.average(mPred, axis=1)
    plt.figure(3)
    plt.plot(meanP)
    plt.draw()
#    plt.pause(0.03)
#    plt.clf()


#-------------------------------------------------OUTPUT-----------------------------------------------------------------#

plt.figure(1)


plt.plot(m, 'g-')
plt.plot(mPred, 'b-')
plt.plot(mAnswer, 'r-')
plt.show()

plt.figure(2)
plt.plot(d, 'g-')
plt.plot(dPrior, 'b-')
plt.plot(dAnswer, 'r-')
plt.show()

print('green = initial ensemble')
print('blue = ensemble at last iteration')
print('red = the answer')