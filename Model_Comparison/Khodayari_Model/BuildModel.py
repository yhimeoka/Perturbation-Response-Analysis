import os, sys, pickle, itertools
import numpy as np
import networkx as nx

sys.path.append('../../CommonModule')
from classmodule import *
import settings

def GenerateNetwork_Metabolites(model,SkipAdditional=False):
    ZeroFix = ['amp_c','adp_c','atp_c','nad_c','nadh_c','nadp_c','nadph_c','q8_c','q8h2_c']
    G = nx.DiGraph()
    for ii,RxnName in enumerate(model.EnzNameList):
        if SkipAdditional and 'AddRxn' in RxnName:
            continue
        M = np.array([0 for i in range(len(model.MetNameList))],dtype='float64')
        for i,rxn in enumerate(model.RxnNameList):
            if RxnName + '_' in rxn and 'inhibit' not in rxn and rxn[-2:] == '_f':
                M += np.array(model.stoich[:,i],dtype='float64')
                if len(model.reactions[rxn].subs) > 2:
                    print('more than bi reaction ',model.reactions[rxn].name,model.reactions[rxn].subs,model.reactions[rxn].prod)
                    exit()
        
            Rate = model.reactions[rxn].v.val-model.reactions[rxn[:-2]+'_b'].v.val 

        if Rate > 0:
            subs, prod = [m for i,m in enumerate(model.MetNameList) if model.metabolites[m].attribute == 'metabolite' and M[i] < -0.1 and m not in ZeroFix], [m for i,m in enumerate(model.MetNameList) if model.metabolites[m].attribute == 'metabolite' and M[i] > 0.1 and m not in ZeroFix]
        else:
            prod, subs = [m for i,m in enumerate(model.MetNameList) if model.metabolites[m].attribute == 'metabolite' and M[i] < -0.1 and m not in ZeroFix], [m for i,m in enumerate(model.MetNameList) if model.metabolites[m].attribute == 'metabolite' and M[i] > 0.1 and m not in ZeroFix]
        for l in itertools.product(subs,prod):
            G.add_edge(l[0],l[1])
        
        
    return G

def FetchIndex(t,target):
    tlist = np.array([abs(x - target) for x in t])
    return np.argmin(tlist)

def CheckNonRelaxedChemicals(idx,variablelist):

    if not os.path.isfile(f'DynamicsData/conc_norm{idx}.dat'):
        return
    t, x = [], []
    with open(f'DynamicsData/conc_norm{idx}.dat','r') as fp:
        for line in fp:
            l = line.replace('\n','').split()
            l = [float(h) for h in l]
            t.append(l.pop(0))
            x.append(l)
    
    h = FetchIndex(t,1e+7)
    print(f'idx={idx}')
    print([m for i,m in enumerate(variablelist) if abs(x[h][i]-1) > 1e-1 and 'complex' not in m])
    print()


def BuildModel(SourceDir='ModelData/1-s2.0-S1096717614000731-mmc4.xlsx',Var2Const=[],INIT_MAX=4096):

    Chemicals,Concentrations,Reactions,Rates,S = ImportData(SourceDir)

    model = ModelObject(Chemicals,Concentrations,Reactions,Rates,S)


    paramval = []
    for r in model.reactions.values():
        paramval.append(np.log10(r.v.val+1e-16))
    #plt.hist(paramval, bins=64)
    #plt.savefig("paramhist_all.png")

    RemoveInhibition = ['PPC_uncomp_inhibit']

    #Complex Inhibitionがあると固有値正が2つ出てくるの外す. Complex以外のchemicalは定常状態で濃度が1になるようになっているので、定数にしたければモデルから外せばOK
    #RemoveChemicals = ['co2_c','h_c','h2o_c','nh4_c','o2_c','pi_c','ppi_c'] + [m for m in model.MetNameList if m[-2:] == '_e' or model.metabolites[m].asc_rxn == []]# + [m.name 
    RemoveChemicals = ['co2_c','h_c','h2o_c','nh4_c','o2_c','pi_c','ppi_c'] + [m for m in model.MetNameList if m[-2:] == '_e' or model.metabolites[m].asc_rxn == []]# + [m.name 
    #5/31 h_cとh_eをこの状況から追加すると安定でなくなる
    #RemoveChemicals.remove('h_e')
    #RemoveChemicals.remove('h_c')
    #RemoveChemicals += [m for m in model.MetNameList if '_inhibit' in m]
    #RemoveChemicals = list(set(RemoveChemicals))

    if 'nadh_c' in Var2Const or 'nad_c' in Var2Const:
        Var2Const = ['nadh_c','nad_c']

    if 'nadph_c' in Var2Const or 'nadp_c' in Var2Const:
        Var2Const = ['nadph_c','nadp_c']

    if 'q8_c' in Var2Const or 'q8h2_c' in Var2Const:
        Var2Const = ['q8_c','q8h2_c']
    
    RemoveReactions = [r.name for r in model.reactions.values() if any([s in r.name for s in RemoveInhibition])]
    RemoveReactions += [r.name for r in model.reactions.values() if abs(r.v.val) < 1e-10]
    RemoveChemicals += Var2Const 

    while True:
        print('Reduce')
        print(RemoveChemicals)
        print(RemoveReactions)
        model.ReduceModel(RemoveChemicals,RemoveReactions)
        RemoveChemicals = [m for m in model.MetNameList if model.metabolites[m].asc_rxn == []]
        RemoveReactions = [r.name for r in model.reactions.values() if r.subs == [] and r.prod == []]
        if RemoveChemicals == [] and RemoveReactions == []:
            break

    model.EvalRHS()

    for folders in ['DynamicsData','matlabscript','ModelInfo']:
        os.makedirs(folders,exist_ok=True)
    
    model.DataDir = 'DynamicsData/'
    model.ScriptDir = 'matlabscript/'
    model.InfoDir = 'ModelInfo/'
        
    model.cokerS()


    model.GenerateRandomInitial(INIT_MAX,Strength = settings.noiselevel_khodayari) 
    variablelist = model.GenerateMatlabScript(INIT_MAX,ParamScript=True,ReactionScript=True,ODEScript=True,runScript=True,Jacobi=True)
            
    with open(model.InfoDir + 'model.pickle', mode="wb") as f:
        pickle.dump(model, f)
   
    return model

if __name__ == '__main__':
    #Input "parallel" for parallel computation, digit for a single computation
    BuildModel()
