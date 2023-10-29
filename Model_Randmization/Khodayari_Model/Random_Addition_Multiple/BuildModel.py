import os, sys, random, pickle, itertools
import numpy as np
import networkx as nx

sys.path.append('../../CommonModule')
from classmodule import *


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


def BuildModel(model_seed,ADD_MAX,generate_initial = False,matlabscript = False):

    Chemicals,Concentrations,Reactions,Rates,S = ImportData('/home/himeoka/Projects/E_Coli_Core_Revisit/Khodayari_Model/ModelData/1-s2.0-S1096717614000731-mmc4.xlsx')
    
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
    
    RemoveReactions = [r.name for r in model.reactions.values() if any([s in r.name for s in RemoveInhibition])]
    RemoveReactions += [r.name for r in model.reactions.values() if abs(r.v.val) < 1e-10]

    while True:
        print('Reduce')
        print(RemoveChemicals)
        print(RemoveReactions)
        model.ReduceModel(RemoveChemicals,RemoveReactions)
        RemoveChemicals = [m for m in model.MetNameList if model.metabolites[m].asc_rxn == []]
        RemoveReactions = [r.name for r in model.reactions.values() if r.subs == [] and r.prod == []]
        if RemoveChemicals == [] and RemoveReactions == []:
            break

    #for e in model.EnzNameList:
    #    model.OutputEnzStructure(e)


    
    model.EvalRHS()

    model.ComputeDistributions()

    #Distribution_ReactionScheme, Distribution_NodeDegree, Distribution_Rates = model.ComputeDistributions()

    ModelFolder = f'Model{model_seed}'
    os.system(f'rm -r Model{model_seed}')
    for add in range(ADD_MAX):
        os.makedirs(ModelFolder+f'/Add{add}/DynamicsData',exist_ok=True)
        os.makedirs(ModelFolder+f'/Add{add}/Script',exist_ok=True)
        os.makedirs(ModelFolder+f'/Add{add}/ModelInfo',exist_ok=True)

    model.additionloop = 0
    AddRxnPerLoop = 1
    
    G = GenerateNetwork_Metabolites(model).to_undirected()

    random.seed(model_seed)
    model.additionloop = 0
    for addition in range(ADD_MAX):
            
        model.DataDir = ModelFolder + f'/Add{addition}/DynamicsData/'
        model.ScriptDir = ModelFolder + f'/Add{addition}/Script/'
        model.InfoDir = ModelFolder + f'/Add{addition}/ModelInfo/'

        model.DataDir = f'DynamicsData/'
        
        model.Initialize()

        if addition >= 2:
            AddRxnPerLoop = 2
        if addition == 6:
            AddRxnPerLoop = 10
            
        for loop in range(AddRxnPerLoop):
            model.AddSingleReaction(G,Distance=[range(1,128)],PreferredDirection=False,UniUniReaction=False,CofactorCoupling=True)
            model.additionloop += 1
        
        print(f'model seed = {model_seed}, loop = {addition}')
        model.cokerS()

        INIT_MAX = 32
        model.GenerateRandomInitial(INIT_MAX,Strength = 0.5)    
        variablelist = model.GenerateMatlabScript(INIT_MAX,ParamScript=True,ReactionScript=True,ODEScript=True,runScript=True,Jacobi=True)
        
        print('model information out')
        with open(model.InfoDir + 'model.pickle', mode="wb") as f:
            pickle.dump(model, f)

    return model

if __name__ == '__main__':
    #Input "parallel" for parallel computation, digit for a single computation
    seed = sys.argv[1]
    ADD_MAX = int(sys.argv[2])
    BuildModel(int(seed),ADD_MAX,generate_initial=True,matlabscript=True)
