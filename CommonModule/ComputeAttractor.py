import numpy as np
import glob
from scipy.spatial.distance import cdist


def FindClosest(T,X):
    return np.argmin((T-X[:,0])**2)

def Compute_LogDerivative(X):
    T0, T1 = 0.1*X[-1,0], X[-1,0]
    X0, X1 = np.log(X[FindClosest(T0,X),1:]), np.log(X[-1,1:])
    dX = abs(X1-X0)/(np.log(T1)-np.log(T0))
    return np.max(dX)

def ComputeAttractor(Folder,MinNumTrajectory,DerTol=1e-2,ConvTol=1e-4,Tmax=1e8,ShowCriteriaValues=False):

    files = glob.glob(Folder+'/conc*.dat')
    if len(files) < MinNumTrajectory:
        return False, []
    
    #Load the data
    FinalConc = []
    for f in files:
        try:
            X = np.loadtxt(f)
        except ValueError: #If the complex number is in the file (note that MATLAB uses "j" for the imaginary unit)) 
            pass
        else:
            #Check if the final concentration is converged using the derivative
            dXdt = Compute_LogDerivative(X)
            if ShowCriteriaValues:
                print(dXdt)
            if X[-1,0] > Tmax*0.9 and np.all(X[:,1:] > -1e-12) and dXdt < DerTol:
                FinalConc.append(X[-1,1:])

    #Check if the length of the final concentration is larger than the minimum number of trajectories    
    if len(FinalConc) < MinNumTrajectory:
        return False, []

    #Check if the final concentration are the same among the initial points
    Y = np.log(FinalConc)
    MaxDist = np.max(cdist(Y,Y))
    if ShowCriteriaValues:
        print(MaxDist)
    if MaxDist < ConvTol:
        return True, np.average(FinalConc,axis=0)
    else:
        return False, []
