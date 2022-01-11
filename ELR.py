# -*- coding: utf-8 -*-
"""
File:   ELR.py
Author:   Tashfique Hasnine Choudhury  
Date:   10.11.2021
Desc:   Implementation of Elastic Net Regularization
    
"""


""" =======================  Import dependencies ========================== """
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

plt.close('all') #close any open plots


""" ======================  Function definitions ========================== """

def plotData(x1,t1,x2=None,t2=None,x3=None,t3=None, legend=[]):
    plt.figure(figsize=(8,6), dpi=100)
    p1 = plt.plot(x1, t1, 'bo') #plot training data
    if(x2 is not None):
        p2 = plt.plot(x2, t2, 'g') #plot true value
    if(x3 is not None):
        p3 = plt.plot(x3, t3, 'r') #plot test data 
    plt.ylabel('y') #label x and y axes
    plt.xlabel('x')
    if(x2 is None):
        plt.legend((p1[0]),legend)
    if(x3 is None):
        plt.legend((p1[0],p2[0]),legend)
    else:
        plt.legend((p1[0],p2[0],p3[0]),legend)
    plt.title(f's={s}, M={M}, iteration={itr}, l1={l1}, Alpha={Alpha}')
    #plt.savefig(f'Functions and estimates s={s}, M={M}, iteration={itr}, l1={l1}, Alpha={Alpha}.png')
    plt.show()                
 
def ErrorPlot3D(array, Data):
    p = np.arange(0, 1.01, 0.01)
    q = np.arange(0,11,1)
    grid_x, grid_y = np.mgrid[min(q):max(q)+1:1, min(p):max(p)+0.01:101j]
    fig = plt.figure(figsize=(8,6), dpi=100)
    ax = fig.gca(projection='3d')
    sp = ax.plot_surface(grid_x, grid_y, array, cmap=cm.cividis)
    ax.set_xlim(min(q)-1, max(q)+1)
    ax.set_ylim(min(p)-0.01, max(p)+0.01)
    ax.set_ylabel('Lambda1')
    ax.set_xlabel('Alpha')
    ax.set_zlabel('Error')
    fig.colorbar(sp, fraction=0.12, shrink=0.8, aspect=15, pad=0.2)
    if Data==True:
        ax.set_title(f's = {s}, Learning rate = {lr}, Iteration = {itr}, Data = Train')
        #plt.savefig(f's = {s}, Learning rate = {lr}, Iteration = {itr}, Data = Train.png')
    elif Data==False:
        ax.set_title(f's = {s}, Learning rate = {lr}, Iteration = {itr}, Data = Validation')
        #plt.savefig(f's = {s}, Learning rate = {lr}, Iteration = {itr}, Data = Validation.png')
    else:
        ax.set_title(f's = {s}, Learning rate = {lr}, Iteration = {itr}, Data = Test')
        #plt.savefig(f's = {s}, Learning rate = {lr}, Iteration = {itr}, Data = Test.png')
    plt.show()      


""" =======================  Load Dataset ======================= """
df = np.load( "data_set.npz" )

TrainData = df['arr_0']
ValidData = df['arr_1']
TestData = df['arr_2']
  
X1 = TrainData[:,0]
Y1 = TrainData[:,1]

X2 = ValidData[:,0]
Y2 = ValidData[:,1]
   
X3 = TestData[:,0]
Y3 = TestData[:,1]
    
""" ========================  Train the Model ============================= """
def rbf(x, s, M, Rand_w=True): 
    mu = X1 #take equal centre points from train data
    Phi = np.zeros((len(X1), M))
    Phi_T = np.zeros((len(x), M))
    for i in range(M):
            Phi[:,i] = np.exp(-0.5*((X1-mu[i])/s)**2)  #compute phi for train data
            Phi_T[:,i] = np.exp(-0.5*((x-mu[i])/s)**2)  #compute phi for test data
    Phi = np.concatenate((np.ones(len(X1))[:, np.newaxis], Phi), axis=1) #append column of ones to phi of train data
    Phi_T = np.concatenate((np.ones(len(x))[:, np.newaxis], Phi_T), axis=1) #append column of ones to phi of test data
    if Rand_w:
        np.random.seed(97880696) #Seed Value = UFID,  Fixing the random value for each run for fair comparison 
        r = np.random.randn(M+1)
        w = r/max(abs(r)) #Normalizing weights to get better performance (Optional Step)
    else:    
        w = np.linalg.inv(Phi.T@Phi+(0.001*np.identity(M+1)))@Phi.T@Y1 #Calculate weights from train data
    return w, Phi, Phi_T

def elReg(w): #Calculation of Regularizer based on three different ranges
    L1=np.zeros(len(w))
    L2=np.zeros(len(w))
    L=np.zeros(len(w))
    for i in range(len(w)): 
        if w[i]>0:
            L1[i]=l1
        elif w[i]<0:
            L1[i]=-l1
        else:
            L1[i]=0
        L2[i]=2*l2*w[i]    
        L[i]=L1[i]+L2[i]    
    return Alpha*L

def GradDesc(w, lr, Phi_T, Phi, Y): # Function for Gradient descent algorithm
    for i in range(itr):
        w = w - lr * (Phi.T @ ((Phi @ w) - Y1)) - lr * elReg(w)
        losses = np.mean(abs(Pred(Phi_T, w) - Y))
    return w, losses #Gives both optimized weight and losses as output

def Pred(Phi_T, w): #Function for prediction
    return Phi_T@w

""" ====================== Calculating true function and error matrix ======================= """

Y4 = 3*(X1 + np.sin(X1)) * np.exp(-X1**2.0) # True function calculation

p=np.arange(0, 1.01, 0.01) #Range of Lambda1
q=np.arange(0,11,1) #Range of Alpha
M=len(X1)
s=0.5
lr=0.005
itr=300
arr1=[]
for Alpha in q:
    lossTr=[]
    for l1 in p:
        w, Phi, Phi_T = rbf(X1, s=s, M=M, Rand_w=True)
        l2 = 1-l1
        w , loss = GradDesc(w, lr, Phi_T, Phi, Y1)
        lossTr.append(loss)
    arr1.append(lossTr)
Train=np.array(arr1) #Train Error Matrix for 3D plot

arr2=[]
for Alpha in q:
    lossVa=[]
    for l1 in p:
        w, Phi, Phi_T = rbf(X2, s=s, M=M, Rand_w=True)
        l2 = 1-l1
        w , loss = GradDesc(w, lr, Phi_T, Phi, Y2)
        lossVa.append(loss)
    arr2.append(lossVa)
Validation=np.array(arr2) #Validation Error Matrix for 3D plot

arr3=[]
for Alpha in q:
    lossTe=[]
    for l1 in p:
        w, Phi, Phi_T = rbf(X3, s=s, M=M, Rand_w=True)
        l2 = 1-l1
        w , loss = GradDesc(w, lr, Phi_T, Phi, Y3)
        lossTe.append(loss)
    arr3.append(lossTe)
Test=np.array(arr3) #Test Error Matrix for 3D plot
 
""" ========================  Plot Results ============================== """
        
"""3D plots for different datasets"""    
ErrorPlot3D(Train, Data=True)
ErrorPlot3D(Validation, Data=False)
ErrorPlot3D(Test, Data=None)

"""plots for traindata, estimate and true function for different alpha and lambda1 combination"""

for Alpha in [0,3,6,10]:
    for l1 in [0.01, 0.1, 0.5, 1]:
        s = 0.25
        lr = 0.005
        itr =500
        M=len(X1)
        w, Phi, Phi_T = rbf(X1, s=s, M=M, Rand_w=True)
        l2 = 1-l1
        w , loss = GradDesc(w, lr, Phi_T, Phi, Y1)
        P = Pred(Phi_T, w)
        plotData(X1,Y1,X1,Y4,X1,P,['Training Data', 'True Function', 'Train Estimate'])

"""Additional tests and plots conducted"""

""" Best hyperparameter combination"""
# m=np.array(np.unravel_index(Train.argmin(), Train.shape))
# n=np.array(np.unravel_index(Test.argmin(), Test.shape))
# o=np.array(np.unravel_index(Validation.argmin(), Validation.shape))
# print('Alpha for Best Error for Train set', m[:1])
# print('Lambda1 for Best Error for Train set', m[1:]/100)
# print('Lambda2 for Best Error for Train set', 1-m[1:]/100)
# print('Alpha for Best Error for Validation set', o[:1])
# print('Lambda1 for Best Error for Validation set', o[1:]/100)
# print('Lambda2 for Best Error for Validation set', 1-o[1:]/100)
# print('Alpha for Best Error for Test set', n[:1])
# print('Lambda1 for Best Error for Test set', n[1:]/100)
# print('Lambda2 for Best Error for Test set', 1-n[1:]/100)

"""Error vs Iteration calculation"""
# def GradDescLoss(w, lr, Phi_T, Phi, Y):
#     losses=[]
#     for i in range(itr):
#         w = w - lr * (Phi.T @ ((Phi @ w) - Y1)) - lr * elReg(w)
#         losses.append(np.mean(abs(Pred(Phi_T, w) - Y)))
#     return w, losses

# l1= 0.5
# Alpha = 0
# itr = 300
# lr = 0.005

# w, Phi, Phi_T = rbf(X1, s=s, M=M)
# np.random.seed(97880696)
# r = np.random.randn(M+1)
# w = r/max(abs(r))
# l2 = 1-l1
# w , loss1 = GradDescLoss(w, lr, Phi_T, Phi, Y1)

# w, Phi, Phi_T = rbf(X2, s=s, M=M)
# np.random.seed(97880696)
# r = np.random.randn(M+1)
# w = r/max(abs(r))
# l2 = 1-l1
# w , loss2 = GradDescLoss(w, lr, Phi_T, Phi, Y2)

# w, Phi, Phi_T = rbf(X3, s=s, M=M)
# np.random.seed(97880696)
# r = np.random.randn(M+1)
# w = r/max(abs(r))
# l2 = 1-l1
# w , loss3 = GradDescLoss(w, lr, Phi_T, Phi, Y3)

# plt.figure(figsize=(8,6), dpi=100)
# plt.plot(range(itr), loss1, color='r', label='Train Set')
# plt.plot(range(itr), loss2, color='b', label='Validation Set')
# plt.plot(range(itr), loss3, color='g', label='Test Set')
# plt.xlabel('Iteration')
# plt.ylabel('Error')
# plt.title('Error vs Iteration')
# plt.legend()

"""Std. Deviation for reliability comparison"""

# def GradDescLossStd(w, lr, Phi_T, Phi, Y):
#     losses=[]
#     for i in range(itr):
#         w = w - lr * (Phi.T @ ((Phi @ w) - Y1)) - lr * elReg(w)
#         losses=(abs(Pred(Phi_T, w) - Y))
#     return w, np.std(losses)

# l1= 0.21
# Alpha = 1
# itr = 300
# lr = 0.005
# s=0.5
# w, Phi, Phi_T = rbf(X3, s=s, M=M)
# np.random.seed(97880696)
# r = np.random.randn(M+1)
# w = r/max(abs(r))
# l2 = 1-l1
# w , lossTest = GradDescLossStd(w, lr, Phi_T, Phi, Y3)
# print('Standard Deviation of Test Errors For Selected Experimental Model (Alpha = 1, Lambda1 = 0.21)', lossTest)
# l1= 0
# Alpha = 0
# s=0.5
# w, Phi, Phi_T = rbf(X3, s=s, M=M)
# np.random.seed(97880696)
# r = np.random.randn(M+1)
# w = r/max(abs(r))
# l2 = 1-l1
# w , lossTest = GradDescLossStd(w, lr, Phi_T, Phi, Y3)
# print('Standard Deviation of Test Errors For Initial Unregularized Model (Alpha = 0, Lambda1 = 0)', lossTest)

"""Minimum error for each sets"""
# print('Min error value for Train set', Train.min())
# print('Min error value for Validation set', Validation.min())
# print('Min error value for Test set', Test.min())

"""Scatter plot for train and validation data"""
# plt.figure(figsize=(8,6), dpi=100)
# plt.plot(X2,Y2, 'bo', label = 'Validation data')
# plt.plot(X1,Y1, 'ro', label = 'Train data')
# plt.legend()

""" Mean Absolute Error Comparison"""
# def GradDescLossMAE(w, lr, Phi_T, Phi, Y):
#     losses=[]
#     for i in range(itr):
#         w = w - lr * (Phi.T @ ((Phi @ w) - Y1)) - lr * elReg(w)
#         losses=(np.mean(abs(Pred(Phi_T, w) - Y)))
#     return w, losses

# l1= 0.5
# Alpha = 0
# itr = 300
# lr = 0.005

# w, Phi, Phi_T = rbf(X3, s=s, M=M)
# np.random.seed(97880696)
# r = np.random.randn(M+1)
# w = r/max(abs(r))
# l2 = 1-l1
# w , loss3 = GradDescLossMAE(w, lr, Phi_T, Phi, Y3)
# print('Mean Absolute Error for Test set over unregularized model',loss3)

# l1= 0.21
# Alpha = 1
# itr = 300
# lr = 0.005

# w, Phi, Phi_T = rbf(X3, s=s, M=M)
# np.random.seed(97880696)
# r = np.random.randn(M+1)
# w = r/max(abs(r))
# l2 = 1-l1
# w , loss3 = GradDescLossMAE(w, lr, Phi_T, Phi, Y3)
# print('Mean Absolute Error for Test set over selected regularized model',loss3)

""" Mean Squared Error Comparison"""

# def GradDescLossMSE(w, lr, Phi_T, Phi, Y):
#     losses=[]
#     for i in range(itr):
#         w = w - lr * (Phi.T @ ((Phi @ w) - Y1)) - lr * elReg(w)
#         losses=(np.mean(np.square(Pred(Phi_T, w) - Y)))
#     return w, losses
# l1= 0.21
# Alpha = 1
# itr = 300
# lr = 0.005
# s=0.5
# w, Phi, Phi_T = rbf(X3, s=s, M=M)
# np.random.seed(97880696)
# r = np.random.randn(M+1)
# w = r/max(abs(r))
# l2 = 1-l1
# w , lossTest = GradDescLossMSE(w, lr, Phi_T, Phi, Y3)
# print('Mean Squared Error for Test set over selected regularized model',lossTest)

# l1= 0.0
# Alpha = 0
# itr = 300
# lr = 0.005
# s=0.5

# w, Phi, Phi_T = rbf(X3, s=s, M=M)
# np.random.seed(97880696)
# r = np.random.randn(M+1)
# w = r/max(abs(r))
# l2 = 1-l1
# w , lossTests = GradDescLossMSE(w, lr, Phi_T, Phi, Y3)
# print('Mean Squared Error for Test set over unregularized model',lossTests)