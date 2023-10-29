import sys
sys.path.append('../../Model_Comparison/Chassagnole_Model')
sys.path.append('../../CommonModule/')
import settings
import BuildModel as BM

if __name__ == '__main__':

    metabolites = list(settings.metabolites_chassagnole[:])
    metabolites.remove('cadp') #because atp + adp is the constnant in the Chassagnole model

    for met in metabolites:
        BM.BuildModel(SourceDir='../../Model_Comparison/Chassagnole_Model/',Var2Const=[met],INIT_MAX=1024)
        