import sys
sys.path.append('../../Model_Comparison/Khodayari_Model')
sys.path.append('../../CommonModule/')
import settings
import BuildModel as BM
if __name__ == '__main__':

    common_metabolite_list = settings.metabolites_khodayari[:]
    boecker_formated = [x.lower().replace('in','') for x in settings.metabolites_boecker]
    chassagnole_formated = [x[1:] for x in settings.metabolites_chassagnole]

    common_metabolite_list = [x for x in settings.metabolites_khodayari if any([y == x[:-2] for y in boecker_formated]) or any([y == x[:-2] in x for y in chassagnole_formated])] + ['2pg_c', '3pg_c', '6pgc_c', '6pgl_c', 'ac_c','etoh_c','lac_D_c','r5p_c', 'rib_D_c', 'ru5p_D_c', 's7p_c']
    common_metabolite_list = sorted(list(set(common_metabolite_list)))
    print(common_metabolite_list,len(common_metabolite_list),len(boecker_formated),len(chassagnole_formated))
    DataFile = '../../Model_Comparison/Khodayari_Model/ModelData/1-s2.0-S1096717614000731-mmc4.xlsx'
    specific_metabolite_list = [x for x in settings.metabolites_khodayari if x not in common_metabolite_list]
    
    for met in common_metabolite_list + specific_metabolite_list:
        BM.BuildModel(DataFile,Var2Const=[met],INIT_MAX=512)
        