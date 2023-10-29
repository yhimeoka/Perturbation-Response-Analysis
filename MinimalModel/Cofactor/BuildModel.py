
import os
import sys
import numpy as np
sys.path.append('../CommonModule')
import minimalmodel_functions as mmf


def Loop(N, R):
    MAX_NETWORK = 512
    for sample in range(MAX_NETWORK):
        mmf.GenerateKineticModel(sample, N, R, beta=10.0)

if __name__ == '__main__':
        
    os.makedirs('DynamicsData', exist_ok=True)
    os.makedirs('DynamicsExample', exist_ok=True)
    os.makedirs('matlabscript', exist_ok=True)
    os.makedirs('AnalyzeResult', exist_ok=True)
    os.makedirs('Stoichiometry', exist_ok=True)

    N = 64
    Rlist = list(np.linspace(N, round(2.0*N), 6, dtype=int))
    print(Rlist)
    for R in Rlist:
        print(R)
        Loop(N, R)
