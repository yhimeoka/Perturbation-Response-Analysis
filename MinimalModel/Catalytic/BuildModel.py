
import os
import sys
import numpy as np
sys.path.append('../CommonModule')
import minimalmodel_functions as mmf


def Loop(N, R):
    MAX_NETWORK = 1    
    for sample in range(MAX_NETWORK):
        mmf.GenerateKineticModel(sample, N, R, beta=20.0, cofactor_coupling=False)

if __name__ == '__main__':
        
    os.makedirs('DynamicsData', exist_ok=True)
    os.makedirs('DynamicsExample', exist_ok=True)
    os.makedirs('matlabscript', exist_ok=True)
    os.makedirs('AnalyzeResult', exist_ok=True)
    os.makedirs('Stoichiometry', exist_ok=True)

    N = 64 + 3
    Rlist = list(np.linspace(N, round(2.0*N), 6, dtype=int))
    print(Rlist)
    for R in Rlist:
        print(R)
        Loop(N, R)

