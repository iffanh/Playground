# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 12:35:46 2019

@author: iha
"""

"""
This is an attempt to simulate ES-MDA based on a paper Analysis of the Performance of Ensemble-based Assimilation of Production and Seismic Data by Alexandre Anoze Emerick 
https://www.researchgate.net/profile/Alexandre_Emerick/publication/292213980_Analysis_of_the_Performance_of_Ensemble-based_Assimilation_of_Production_and_Seismic_Data/links/5a878fe3aca272017e5abf36/Analysis-of-the-Performance-of-Ensemble-based-Assimilation-of-Production-and-Seismic-Data.pdf)

In this script, we will predict a vector of model parameters, mAnswer ∈ R^M, by data conditioning to the observed data, dAnswer ∈ R^D. 
The true parameters, mAnswer, as well as the true data, dAnswer, are hidden. The relationship between the model parameters and the 
data are defined in some function gFunction() that spans from R^M -> R^D. Here, we assume that we know the function perfectly (no systematic error)

We start the simulation by having an initial prediction of parameter, mInit, and the corresponding prediction data, dPrior. 
We populate the initial ensemble by distributing the model parameters, mPrior, by sampling it from Gaussian distribution. mPrior ~ N(mInit, stdevM)
The observed data also has some uncertainty due to measurement noise. Here, the noise is assumed to be Gaussian. obsData ~ N(dAnswer, stdevD)

(Control) parameters
    ----------
    mLength : int
        The number of model parameter
    dLength : int 
        The number of data
    nEnsemble : int
        The number of element in an ensemble
    maxIter : int
        The number of ES-MDA iteration
    stdevD : float
        Standard deviation of the observed data
    stdevM : float
        Standard deviation of the model parameter in the initial ensemble
    
    
Returns
    -------
    1. Plot of the average of the ensemble over time
    2. Plot of the true parameter, and the corresponding initial ensemble, and ensemble at last iteration
    3. Plot of the true observed data, and the corresponding initial ensemble, and ensemble at last iteration
    '''

"""

#------------------------------------------------INTRODUCTIONS----------------------------------------------------------#
#Import libraries

import numpy as np
import matplotlib.pyplot as plt

#Constants

mLength = 10 #the length of the parameter m
dLength = 5 #the length of the data d
nEnsemble = 100 #the number of ensembles

alpha_max = 1000.
maxIter = 4

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

#Generating the initial parameter (base case)
for i in range(mLength):
    ''' Here, you can initialize your own model parameter'''
    a = i/mLength
#Some functions that might be used
#    mInit[i] = np.random.uniform(0,1)*1.
#    mInit[i] = np.random.uniform(0,1)*0.005
#    mInit[i] = a**2 + 1
    mInit[i] = a/(a**2 + 1)


#--------------------------------------------------FUNCTIONS-------------------------------------------------------------#
#Calculating the observation based on the parameter. This is function d = g(m)
#the non-linear functions obsData = gFunctions(parameters, dLength)
#In the reservoir, the function is much more complex and needs a reservoir simulator to solve it (Eclipse, Intersect, etc.)


def gFunctions(parameters, dLength):
    ''' Here, you can make your own functions!'''

    predData = np.zeros(dLength)
    
    mSum = np.sum(parameters)
    for i in range(dLength):
        a = i/dLength
#       predData[i] = mSum/(a + 1)
        predData[i] = mSum*(a+1)**2/(100)
        
    return predData
   
#----------------------------------------------THE 'ANSWERS'-------------------------------------------------------------#

#This parts consists of parameters that are considered the truth value of the model. 
#Consequently, by plugging it into the gFunction() we would get the true observed data (data without noise) 

for i in range(mLength):
    ''' Here, you store the true value of the model parameter'''

    a = i/mLength
#    mAnswer[i] = a**3 + a**2 + 5
#    mAnswer[i] = a**2 + 10*a + 5
#    mAnswer[i] = np.sin(a*np.pi/6)
#    mAnswer[i] = np.cos(a*np.pi/6)
#    mAnswer[i] = 1/(a**2 + 3)
    mAnswer[i] = np.exp(a/10)

dAnswer[:] = gFunctions(mAnswer[:], dLength)

stdevD = np.diag(dAnswer*0.001)
stdevM = np.multiply(mAnswer,0.1)

#------------------------------------------------POPULATING ENSEMBLE-----------------------------------------------------#
#Populate ensemble based on mean and standard deviation (we assume normal distribution for the noise in measurement)

#Perturb the parameter
for i in range(mLength):
    mPrior[i,:] = np.random.normal(mInit[i], np.abs(stdevM[i]), nEnsemble)
    
m = mPrior #Initial ensemble

#Calculate prediction
for j in range(nEnsemble):
    dPrior[:,j] = gFunctions(mPrior[:,j], dLength)
    d[:,j] = gFunctions(mPrior[:,j], dLength)

#d = dPrior
    
#------------------------------------------MAIN LOOP STARTS HERE---------------------------------------------------------#


for p in range(maxIter):
    #Get data
    
    #Adding measurement noise to the true data    
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

    #Main update equation
    for j in range(nEnsemble):
        dummyMat = np.matmul(covarianceMD,np.linalg.inv(covarianceDD + alpha[p]*stdevD)) 
        dummyVec = obsData[:,j] - dPrior[:,j]
        mPred[:,j] = mPrior[:,j] + np.matmul(dummyMat,dummyVec)

    
    #Calculate new forecast based on the predicted parameters
    for j in range(nEnsemble):
        dPrior[:,j] = gFunctions(mPred[:,j], dLength)
    
    #Update the prior parameter for next iteration
    mPrior = mPred
        
    #Plotting for change of average of the parameters
    meanP = np.average(mPred, axis=1)
    plt.figure(3)
    plt.plot(meanP)

    plt.title('(averaged) model parameters')
    plt.xlabel('i-th parameter')
    plt.ylabel('m[i]')

    plt.draw()

#-------------------------------------------------OUTPUT-----------------------------------------------------------------#

#Plot of the ensemble of the parameters
plt.figure(1)
plt.plot(m, 'g-')
plt.plot(mPred, 'b-')
plt.plot(mAnswer, 'r-')

plt.title('model parameters')
plt.xlabel('i-th parameter')
plt.ylabel('m[i]')

plt.show()

#Plot of the ensemble of the data
plt.figure(2)
plt.plot(d, 'g-')
plt.plot(dPrior, 'b-')
plt.plot(dAnswer, 'r-')

plt.title('data')
plt.xlabel('i-th data')
plt.ylabel('d[i]')
plt.show()

print('green = initial ensemble')
print('blue = ensemble at last iteration')
print('red = the answer')