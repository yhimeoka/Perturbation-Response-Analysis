import sys
sys.path.append('../../CommonModule')
import RCRN_Model as RCRN
import settings
import numpy as np

if __name__ == '__main__':
    
    N = len(settings.allmetabolites_khodayari)  
    R = int(np.ceil(N*4.0))
    
    parallel = True
    ParameterFilePath = 'ModelData/RateBi.txt'
    Start, End = 0, 128
    
    RCRN.MainLoop(N, R, parallel, ParameterFilePath, Start, End)
