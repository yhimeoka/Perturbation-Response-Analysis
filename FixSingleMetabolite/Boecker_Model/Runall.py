import sys, itertools, os
sys.path.append('../../Model_Comparison/Boecker_Model')
sys.path.append('../../CommonModule/')
import settings
import BuildModel as BM

if __name__ == '__main__':
    print('Update has not yet completed, 2023 Nov 16')
    exit()

    DataFile = '../../Model_Comparison/Boecker_Model/ModelData/Boecker_KineticModel_Version2.xlsx'
    for met in settings.metabolites_boecker:
        print(met)
        BM.MainLoop(DataFile,Var2Const=[met],INIT_MAX=1024)
        