import sys
sys.path.append('../../CommonModule')
import RCRN_Model as RCRN
import settings
import numpy as np

if __name__ == '__main__':
    
    N = len(settings.metabolites_boecker)  
    R = int(np.ceil(N*3.5))

    parallel = False
    ParameterFilePath = 'ModelData/v_params.txt'
    Start, End = 0, 1
    
    RCRN.MainLoop(N,R,parallel,ParameterFilePath,Start,End)
