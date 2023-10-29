import numpy as np
import os, copy, joblib
from tqdm import tqdm
from collections import defaultdict

def isfloat(s):
    try:
        float(s)
    except ValueError:
        return False
    else:
        return True

# In the Khodayari model, some metabolites involved in conservation laws are not pre-solved.
# So, by fixing, for instance, nad, we need to fix nadh as well. 
# In the code in FixSingleMetabolite, I removed the corresponding metabolites from the variables (e.g., nad and nadh together)
# Thus, the dimension of the vector is reduced by some. Here, I add the removed metabolite to the vector. (Only for the Khodayari model)
def ConcatenateData(y,concatenate_size):
    InsertData = np.ones((np.shape(y)[0],concatenate_size))
    return np.insert(y,1,InsertData.T,axis=1)

def ImportSingle(ini, DataDir, N, Tmax):
    if not os.path.isfile(DataDir+f'conc{ini}.dat'):
        return np.array(-1)
    
    y = []
    with open(DataDir+f'conc{ini}.dat', 'r') as fp:
        for line in fp:
            l = [float(x) if isfloat(x) else -1 for x in line.replace('\n', '').split()]
            if any([x < 0 for x in l]):
                return np.array(-1)
            y.append(l)

    if len(set([len(l) for l in y])) != 1:
        return np.array(-1)

    y = np.array(y)
    w = np.shape(y)[1]
    t = y[-1,0]
    if t < Tmax or len(y[y<0]) > 0 or any([abs(np.log(y[0,i]/y[-1,i])) > 0.7 for i in range(1,w)]):
        return np.array(-1)
    
    concatenate_size = N - (np.shape(y)[1] - 1) # note that y contains time information
    if concatenate_size:
        y = ConcatenateData(y,concatenate_size)

    return y


def MainPart(N, R, INIT_MAX, loop, DataFolder, dx, Tmax):
    

    x = []
    x = joblib.Parallel(n_jobs=-1)(joblib.delayed(ImportSingle)(ini,DataFolder,N,Tmax) for ini in range(1,INIT_MAX+1))
    x = [z for z in x if len(z[z<0]) == 0]

    # Check if the dynamics reaches to the attractor
    Att = np.array([0 for i in range(N)],dtype=float)
    for y in x:
        Att += y[-1,1:]/len(x)    
    tmp = copy.deepcopy(x)
    x = []
    for ii,y in enumerate(tmp):
        if np.linalg.norm(np.log(Att) - np.log(y[len(y)-1,1:])) < 0.5:
            x.append(y)
            
    Completed_Trj = len(x)

    if Completed_Trj < INIT_MAX/8:
        print(f'model{loop} has not sufficient trajectories')
        return

    with open(f'Analysis/Att{loop}.txt', 'w') as fp:
        fp.write(','.join([str(y) for y in Att]))

    # Compute the response coefficient
    LogAtt = np.log(Att)
    RespCoeffList = []
    for y in x:
        z = np.log(y[:,1:])
        Perturbation = np.linalg.norm(z[0,:]-LogAtt)
        Response = max([np.linalg.norm(z[ii,:]-LogAtt) for ii in range(len(z))])
        RespCoeffList.append(Response/Perturbation)

    with open(f'Analysis/RespCoeffList{loop}.txt', 'w') as fp:
        fp.write(' '.join([str(y) for y in RespCoeffList]))

    # Optput the distribution of the response coefficient
    Rmax, Rmin = max(RespCoeffList), 0.95
    BIN = int(np.ceil(Rmax-Rmin)/dx) + 1
    hist = [0 for i in range(BIN)]
    for r in RespCoeffList:
        hist[round((r-Rmin)/dx)] += 1.0/len(RespCoeffList)/dx 
    output = ''
    for i in range(BIN):
        output += f'{Rmin + i*dx} {hist[i]}\n'
    with open(f'Analysis/RespCoeff{loop}.txt','w') as fp:
        fp.write(output)

