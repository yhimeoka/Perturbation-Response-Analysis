import pandas as pd
import sympy, random, os, sys
from tqdm import tqdm
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
sys.path.append('../../CommonModule')
import settings
#### Format ####
# Function (dict)
#'Function_for_ACEtrsp_0': {'name': 'Function_for_ACEtrsp_0',
#  'args': ['V_max_ACEtrsp', 'k_ACEtrsp', 'ACEin'],
#  'math': ACEin*V_max_ACEtrsp/(ACEin + k_ACEtrsp)},
#
# Parameters (dict)
# 'k_NGAM_ATP': {'name': 'k_NGAM_ATP',
#  'scope': 'local:Reaction(NGAM)',
#  'value': 0.008,
#  'const': True}
#
# Usedreactions : list of reactions used in the model
#
# RelatedReactions : (dict) metabolites are the key and you can see by which reaction the metabolites are consumed / produced
# 'ADP': {'Consumed': ['PYK', 'FRD', 'PGK', 'PTACK'],
#  'Produced': ['PCK', 'Growth', 'PFK', 'ADK', 'ATPase', 'NGAM']},
#
# Stoich[n,rxn] : Dictionary of stoichiometric matrix (n and rxn are the name of metabolites and reactions, respectively)
#
# Reactions : Dictionary of the reactions. 'math' item is not used to build the model
# 'PTACK': {'name': 'PTACK',
#  'reactants': ['AcCoA', 'ADP'],
#  'products': ['ATP', 'ACEin'],
#  'modifiers': ['PYR', 'PEP'],
#  'math': 'Out*Function_for_PTACK(Vmax_PTACK,AcCoA,k_PTACK_AcCoA,n_PTACK_AcCoA,ADP,k_PTACK_ADP,ATP,ACEin,CoA,k_PTACK_eq,k_PTACK_ATP,k_PTACK_CoA,k_PTACK_i_ATP,PYR,k_PTACK_a_PYR,PEP,k_PTACK_a_PEP)'},

def AddReactions(Nadd,VariablesInPaper,Function,Parameters, UsedReactions, RelatedReactions, Reactions, Stoich):
    
    NrxnCplATP, NrxnCplNADH = 6, 7
    Cofactors = ['ATP','ADP','NADH','AcCoA']
    MetabolitesList = [x for x in VariablesInPaper if x not in Cofactors]    
    #Add Reactions
    Reactants = []
    Coupling = []
    for i in range(Nadd):
        while 1:
            mets = random.sample(MetabolitesList,2)
            if set(mets) not in Reactants:
                Reactants.append(set(mets))
                break
        random_number = random.random()
        if random_number < NrxnCplATP  / len(Reactions):#if connection is ATP
            Coupling.append(['ATP','ADP'])
        elif (NrxnCplATP + NrxnCplNADH) / len(Reactions):#if connection is NADH
            Coupling.append(['NADH','NAD'])
        else: #no coupling
            Coupling.append([])
    

    UniUni_Reaction_Args = []
    
    for rxn in ['ACEtrsp_0', 'ATPase', 'ENO', 'FUMuptake', 'PGI']: #The reaction having uni-uni reaction shceme and uni-none reaction (transporter reaction have the same parameter, and thus, Acetate transporter is used as a representative)
        UniUni_Reaction_Args += [x for x in Function['Function_for_'+rxn]['args'] if x not in VariablesInPaper]
    
    BiBi_Reaction_Args = [] 
    for rxn in ['PTS_0','GDH','PGK','PFL','ALDH','ADH','MDH','ADK']:
        BiBi_Reaction_Args += [x for x in Function['Function_for_'+rxn]['args'] if x not in VariablesInPaper]
    
    VmaxSet, KmSet, EqSet_Uni, EqSet_Bi = [Parameters[x]['value'] for x in Parameters.keys() if ('Vmax' in x or 'V_max' in x) and Parameters[x]['value'] > 1e-8], [Parameters[x]['value'] for x in Parameters.keys() if x[0] == 'k' and not 'EQ' in x  and Parameters[x]['value'] > 1e-8], [Parameters[x]['value'] for x in UniUni_Reaction_Args if x[0] == 'k' and 'EQ' in x and Parameters[x]['value'] > 1e-8],[Parameters[x]['value'] for x in BiBi_Reaction_Args if x[0] == 'k' and 'EQ' in x and Parameters[x]['value'] > 1e-8] 

    for i in range(Nadd):
        rxn_name = f'RxnAdd{i}'
        mets = list(Reactants[i])
        if random.random() < 0.5:
            mets.reverse()
        
        if Coupling[i] == []:
            v, ka, kb, keq = f'Vmax_' + rxn_name, 'Ka_' + rxn_name, 'Kb_' + rxn_name, 'Keq_' + rxn_name
            math = sympy.sympify(f'{v}*{mets[0]}/{ka}*(1 - {mets[1]}/{mets[0]}/{keq})/(1. + {mets[0]}/{ka} + {mets[1]}/{kb})')
            Function[f'Function_for_RxnAdd{i}'] = {'name': f'Function_for_RxnAdd{i}','args':[v,ka,kb,keq,mets[0],mets[1]],'math':math}
            for p,DataSet in zip([v,ka,kb,keq],[VmaxSet,KmSet,KmSet,EqSet_Uni]):
                Parameters[p] = {'name': p,'scope': f'local:Reaction(RxnAdd{i})','value': random.choice(DataSet),'const': True}

            RelatedReactions[mets[0]]['Consumed'].append(rxn_name)
            RelatedReactions[mets[1]]['Produced'].append(rxn_name)
            
            Reactions[rxn_name] = {'name':rxn_name,'reactants':[mets[0]],'products':[mets[1]],'modifiers':[]}

            UsedReactions.append(rxn_name)

            Stoich[mets[0],rxn_name] = -1
            Stoich[mets[1],rxn_name] = 1
        else:
            cpl = Coupling[i]
            if random.random() < 0.5:
                cpl.reverse()
            
            cpl = ['(2-NADH)' if x == 'NAD' else x for x in cpl]
            v, ka, kb, kc, kd, keq = f'Vmax_' + rxn_name, 'Ka_' + rxn_name, 'Kb_' + rxn_name, 'Kc_' + rxn_name, 'Kd_' + rxn_name, 'Keq_' + rxn_name
            math = f'{v}*{mets[0]}/{ka}*{cpl[0]}/{kc}*(1 - {mets[1]}*{cpl[1]}/{mets[0]}/{cpl[0]}/{keq})/(1. + {mets[0]}/{ka} + {mets[1]}/{kb} + {mets[0]}/{ka}*{cpl[0]}/{kc} + {cpl[1]}/{kd}*{mets[1]}/{kb})' 
            print('='*20)
            print(math)
            print('='*20,end='\n')
            math = sympy.sympify(math)
            Function[f'Function_for_RxnAdd{i}'] = {'name': f'Function_for_RxnAdd{i}','args':[v,ka,kb,kc,kd,keq]+mets+cpl,'math':math}
            
            for p,DataSet in zip([v,ka,kb,kc,kd,keq],[VmaxSet,KmSet,KmSet,KmSet,KmSet,EqSet_Bi]):
                print(p)
                print(DataSet)
                Parameters[p] = {'name': p,'scope': f'local:Reaction(RxnAdd{i})','value': random.choice(DataSet),'const': True}

            Reaction_Type = ['Consumed','Produced']
            for i in range(2):
                RelatedReactions[mets[i]][Reaction_Type[i]].append(rxn_name)
                if cpl[i] != '(2-NADH)':
                    RelatedReactions[cpl[i]][Reaction_Type[i]].append(rxn_name)

            Reactions[rxn_name] = {'name':rxn_name,'reactants':[mets[0]],'products':[mets[1]],'modifiers':[]}
            if cpl[0] != '(2-NADH)':
                Reactions[rxn_name]['reactants'].append(cpl[0])
            if cpl[1] != '(2-NADH)':
                Reactions[rxn_name]['reactants'].append(cpl[1])

            UsedReactions.append(rxn_name)

            Stoich[mets[0],rxn_name] = -1
            Stoich[mets[1],rxn_name] = 1
            if cpl[0] != '(2-NADH)':
                Stoich[cpl[0],rxn_name] = -1
            if cpl[1] != '(2-NADH)':
                Stoich[cpl[1],rxn_name] = 1
            


def GenerateInitials(Attractor,VariablesInPaper,Strength,INIT_MAX,OutPutDir,Var2Const=[]):
    random.seed(0)
    N = len(VariablesInPaper)
    Conservation = [sympy.sympify('2 - NADH'),sympy.sympify('3.67 - AcCoA'),sympy.sympify('2.70058 - ADP - ATP')]
    NADt, CoAt, At = 2.0, 3.67, 2.70058
    Xini = np.zeros((INIT_MAX,len(Attractor)))
    for i in tqdm(range(INIT_MAX)):
        while 1:
            x = []
            for j in range(N):
                g = 1 + random.uniform(-Strength,Strength)
                x.append(Attractor[j]*g)
            #Conservation Quantities
            NADH, AcCoA, ATP, ADP = Attractor[VariablesInPaper.index('NADH')], Attractor[VariablesInPaper.index('AcCoA')], Attractor[VariablesInPaper.index('ATP')], Attractor[VariablesInPaper.index('ADP')]
            NAD, CoA, AMP = NADt - NADH, CoAt - AcCoA, At - ATP - ADP
            #random
            NADHr, AcCoAr, ATPr, ADPr, NADr, CoAr, AMPr = NADH*random.uniform(1.-Strength,1.+Strength), AcCoA*random.uniform(1.-Strength,1.+Strength), ATP*random.uniform(1.-Strength,1.+Strength), ADP*random.uniform(1.-Strength,1.+Strength), NAD*random.uniform(1.-Strength,1.+Strength), CoA*random.uniform(1.-Strength,1.+Strength), AMP*random.uniform(1.-Strength,1.+Strength) 

            for met in Var2Const:
                x[VariablesInPaper.index(met)] = Attractor[VariablesInPaper.index(met)]            

            subslist = [(met,x[VariablesInPaper.index(met)]) for met in ['NADH','ADP','ATP','AcCoA']]
            if all([float(Q.subs(subslist)) > 1e-4 for Q in Conservation]):
                break
            
        for j in range(N):
            Xini[i,j] = x[j]

    np.savetxt(OutPutDir+'matlabscript/initials.txt',Xini,delimiter=' ')


def BuildModel_AddReactions(seed,DataFile,OutPutDir ='',Nadd=0,INIT_MAX = 128,Var2Const=[]):
    


    random.seed(seed)

    for foldername in [OutPutDir+'matlabscript',OutPutDir+'DynamicsData']:
        os.makedirs(foldername,exist_ok=True)


    #論文Appendixでの変数（比較しやすいように順番込み）
    VariablesInPaper = settings.metabolites_boecker
    Attractor = settings.attractor_boecker

    #外部濃度（Glc以外はダイナミクスに影響すら与えない）とかとか。変数でないもの
    ExternalChemicals = ['FOR','LAC','ACE','ETH','SUC','Biomass'] + ['s'+str(i) for i in range(50)]
    #外部濃度を変更させたりするだけ（Productionを測るため）の関数なので無視
    Skip_Functions = ['Function_for_Fumarate_conversion','Function_for_FOR_excr','Function_for_LACexcr','Function_for_ACE_excr','Function_for_ETH_excr','Function_for_SUC_excretion','Function_for_Glucose_Uptake']
    
    Species_Data = pd.read_excel(DataFile, sheet_name='Species')
    Reactions_Data = pd.read_excel(DataFile, sheet_name='Reactions')
    Functions_Data = pd.read_excel(DataFile, sheet_name='Functions')
    Parameters_Data = pd.read_excel(DataFile, sheet_name='Parameters')
    Rules_Data = pd.read_excel(DataFile, sheet_name='Rules')

    Rules = {}
    N = len(Rules_Data)
    for i in range(N):
        math, vtype, variable = str(Rules_Data['math'][i]).replace(' ',''), str(Rules_Data['variable type'][i]).replace(' ',''),str(Rules_Data['variable'][i]).replace(' ','')
        Rules[variable] = {'variable':variable,'math':math,'vtype':vtype}

    ExternalChemicals += [rule['variable'] for rule in Rules.values() if rule['vtype'] == 'Species']    

    #取り込んだら次にRulesを元に変数を消去、sympyで代入してみてfloat変換ができるのであれば成功
    Chemicals = {}
    N = len(Species_Data)
    for i in range(N):
        name, group, initial, const = str(Species_Data['name'][i]).replace(' ',''), str(Species_Data['class'][i]).replace(' ',''), str(Species_Data['initialQuantity'][i]).replace(' ',''),str(Species_Data['constants'][i]).replace(' ','')
        if name not in ExternalChemicals:
            Chemicals[name] = {'name':name,'class':group,'initialQuantity':float(initial),'const':const=='True'}

    Reactions = {}
    N = len(Reactions_Data)
    for i in range(N):
        name, reactants, products, modifiers, math = str(Reactions_Data['id'][i]).replace(' ',''), str(Reactions_Data['reactants'][i]).replace(' ','').split(','), str(Reactions_Data['products'][i]).replace(' ','').split(','), str(Reactions_Data['modifiers'][i]).replace(' ','').split(','), str(Reactions_Data['math'][i]).replace(' ','')
        reactants = [x for x in reactants if x not in ExternalChemicals]
        products = [x for x in products if x not in ExternalChemicals]
        modifiers = [x for x in modifiers if x not in ExternalChemicals]

        Reactions[name] = {'name':name,'reactants':reactants,'products':products,'modifiers':modifiers,'math':math}

    FunctionsNameList = []
    Functions = {}
    N = len(Functions_Data)
    subslist = [(rule['variable'],rule['math']) for rule in Rules.values() if rule['vtype'] == 'Species']
    SolvedChemicals = [rule['variable'] for rule in Rules.values() if rule['vtype'] == 'Species']    
    print(subslist)
    for i in range(N):
        name, args, math = str(Functions_Data['id'][i]).replace(' ',''), str(Functions_Data['arguments'][i]).replace(' ',''), str(Functions_Data['math'][i]).replace(' ','')
        if name not in Skip_Functions:
            args = [x for x in args.split(',') if x not in SolvedChemicals]
            Functions[name] = {'name':name,'args':args,'math':sympy.sympify(math).subs(subslist)}
            FunctionsNameList.append(name)


    Parameters = {}
    N = len(Parameters_Data)
    for i in range(N):
        name, scope, value, const = str(Parameters_Data['id'][i]).replace(' ',''), str(Parameters_Data['scope'][i]).replace(' ',''), str(Parameters_Data['value'][i]).replace(' ',''), str(Parameters_Data['constant'][i]).replace(' ','')
        Parameters[name] = {'name':name,'scope':scope,'value':float(value),'const':const=='True'}

    Parameters['default'] = {'name':'default','scope':'compartment','value':1.0,'const':True}
    Parameters['Out'] = {'name':'Out','scope':'compartment','value':1.0,'const':True}
    Parameters['Cytoplasm'] = {'name':'Cytoplasm','scope':'compartment','value':1.0,'const':True}

    for i in Chemicals:
        if Chemicals[i]['const']:
            Parameters[i] = {'name':i,'scope':'species','value':Chemicals[i]['initialQuantity'],'const':True}

    #======================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================
    RelatedReactions = {}
    UsedReactions = []
    for i in Chemicals.keys():
        RelatedReactions[i] = {'Consumed':[],'Produced':[]}
        
    for rxn in Reactions.values():
        for i in rxn['reactants']:
            RelatedReactions[i]['Consumed'].append(rxn['name'])
        for i in rxn['products']:
            RelatedReactions[i]['Produced'].append(rxn['name'])

    for i in Chemicals.keys():
        if i in VariablesInPaper:
            UsedReactions += RelatedReactions[i]['Consumed'] + RelatedReactions[i]['Produced']
    UsedReactions = sorted(list(set(UsedReactions)))
    print(UsedReactions)

    #for i in VariablesInPaper:
    #    print(i,RelatedReactions[i])
    #print(set(Chemicals.keys()) - set(VariablesInPaper))


    #Make ODEs
    #OutとCytoplasmの違いがよく分からんがまぁどうせ全部1なので気にしないことにする
    #===============================================================================================================================================================================================================================================================================
    #sympyで代入してみて値を吐くか.ついでにStoichiometryつくる
    Stoich = {}
    for rxn in UsedReactions:
        if rxn[:2] == 'D_':
            continue
        func = 'Function_for_'+rxn
        if func not in Functions.keys():
            print(func,'not in the list')
            continue
        #print(rxn,Functions[func]['math'])
        #print(Functions[func]['args'])
        subslist = [(i,Parameters[i]['value']) if i in Parameters.keys() else (i,Chemicals[i]['initialQuantity']) for i in Functions[func]['args']]
        #print(rxn,Functions[func]['math'].subs(subslist))
        print(rxn,float(Functions[func]['math'].subs(subslist)))
        #print()
        for n in Reactions[rxn]['reactants']:
            Stoich[n,rxn] = -1
        for n in Reactions[rxn]['products']:
            Stoich[n,rxn] = 1

    #PDFにしか多分データがない、正しいStoichiometryを入れる
    Stoich['FORin','Growth'] = -107.59
    Stoich['ACEin','Growth'] = 581.01
    Stoich['SUCin','Growth'] = 1040.71
    Stoich['G6P','Growth'] = -600.28
    Stoich['F6P','Growth'] = -466.42
    Stoich['GAP','Growth'] = -459.27
    Stoich['D3PG','Growth'] = -1717.48
    Stoich['PEP','Growth'] = -810.19
    Stoich['PYR','Growth'] = -2784.13
    Stoich['AcCoA','Growth'] = -3856.57
    Stoich['OAA','Growth'] = -2924.97
    Stoich['AKG','Growth'] = -1075.32
    Stoich['ATP','FRD'] = 0.75
    Stoich['ADP','FRD'] = -0.75
    Stoich['ATP','Growth'] = -53107.60
    Stoich['ADP','Growth'] = 49739.50
    Stoich['NADH','Growth'] = -14889.30
    Stoich['GAP','FBA'] = 2.0
    Stoich['ADP','ADK'] = 2.0


    #反応追加
    if Nadd:
        AddReactions(Nadd,VariablesInPaper,Functions,Parameters, UsedReactions, RelatedReactions, Reactions, Stoich)

    #Sign Consistency Check
    print('Consistency Check')
    okay = True
    for rxn in UsedReactions:
        if rxn[:2] == 'D_':
            continue
        for n in set(VariablesInPaper) & set(Reactions[rxn]['reactants']):
            if Stoich[n,rxn] > 0:
                print(n,rxn)
                okay = False
        for n in set(VariablesInPaper) & set(Reactions[rxn]['products']):
            if Stoich[n,rxn] < 0:
                print(n,rxn)
                okay = False
    if okay:
        print('okay')    


    #パラメーター
    output = ''
    for i in Parameters.keys():
        output += i + '=' + str(Parameters[i]['value']) + ';\n'
    with open(OutPutDir+'matlabscript/parameters.m','w') as fp:
        fp.write(output)

    #微分方程式
    output = ''
    for rxn in UsedReactions:
        if rxn[:2] == 'D_':
            continue
        func = 'Function_for_'+rxn
        
        subslist = [(i,f'x({VariablesInPaper.index(i)+1})') for i in Functions[func]['args'] if i in VariablesInPaper]
        output += '%'+rxn+'\n'
        print(rxn)
        print(Functions[func]['math'])
        print(subslist)
        print()
        output += 'J_'+rxn+' = @(x) '+str(Functions[func]['math'].subs(subslist)) + ';\n\n'
    with open(OutPutDir+'matlabscript/reactions.m','w') as fp:
        fp.write(output.replace('**','^'))

    output = ''
    for i,n in enumerate(VariablesInPaper,start=1):
        output += '%'+n+'\n'
        if n in Var2Const:
            output += f'dx{i}_dt = @(t,x) 0;\n'
        else:     
            output += f'dx{i}_dt = @(t,x) ' + ' '.join([str(Stoich[n,rxn])+'*J_'+rxn + '(x)' for rxn in RelatedReactions[n]['Consumed'] if rxn[:2] != 'D_']) + ' ' + ' '.join(['+'+str(Stoich[n,rxn])+'*J_'+rxn+'(x)' for rxn in RelatedReactions[n]['Produced']]) + f' -J_Growth(x)*x({i});\n\n'
        #Constant dilution
        #output += f'dx{i}_dt = @(t,x) ' + ' '.join([str(Stoich[n,rxn])+'*J_'+rxn + '(x)' for rxn in RelatedReactions[n]['Consumed'] if rxn[:2] != 'D_']) + ' ' + ' '.join(['+'+str(Stoich[n,rxn])+'*J_'+rxn+'(x)' for rxn in RelatedReactions[n]['Produced']]) + f' -{1.3e-4}*x({i});\n\n'


    output += 'df_dt = @(t,x)[' + ';'.join([f'dx{i}_dt(t,x)' for i in range(1,len(VariablesInPaper)+1)]) + '];\n'
    with open(OutPutDir+'matlabscript/ODEs.m','w') as fp:
        fp.write(output)

    #初期値
    GenerateInitials(Attractor,VariablesInPaper,settings.noiselevel_boecker,INIT_MAX,OutPutDir,Var2Const=Var2Const)

    #Main Script
    N = len(VariablesInPaper)
    R = len(['J_'+rxn+'(x)' for rxn in UsedReactions if rxn[:2] != 'D_'])
    output = 'clearvars\nxini = importdata(\'initials.txt\');\nNini=size(xini,1);\nparfor init=1:Nini\ndisp(init)\nSingle(init);\nend\ndlmwrite(\'done.txt\',[1]);\n\n'#poolobj = gcp(\'nocreate\');\ndelete(poolobj);\n\n'
    output += 'function Single(init)\n'
    output += 'xini = importdata(\'initials.txt\');\ndisp(init)\nparameters;reactions;ODEs;\n'
    output += 'y0 = [' + ' '.join([f'xini(init,{n+1})' for n in range(N)])+'];\n'
    output += 'Reaction = @(x)[' + ';'.join(['J_'+rxn+'(x)' for rxn in UsedReactions if rxn[:2] != 'D_']) + '];\n'
    output += 'tstart = tic;\n'
    output += 'options = odeset(\'RelTol\',1e-4,\'AbsTol\',1e-6);\n'
    output += '[t,y] = ode15s(df_dt,[0,1e+8],y0,options);\n'
    
    output += 'clear tau z\n'
    
    output += 'w=size(y,1);\n'
    output += 'for n = 1:' + str(N) + f'\nz(1,n)=y(2,n);\nend\ntau(1,1)=t(2);\nQ=Reaction(y(2,:));\nfor n=1:{R}\nF(1,n)=Q(n);\nend\n'
    output += 'count=2;\n'
    output += 'for c = 3:w\n'
    output += 'dist = 0;\ntnext=1;\n'
    output += 'for n = 1:' + str(N) + '\ndist = dist + (log(y(c,n)) - log(z(count-1,n)))^2;\nend\ndist = sqrt(dist);\n'
    output += f'if dist > 1e-1 | t(c) > tnext\ntau(count,1)=t(c);\nQ=Reaction(y(c,:));\nfor n=1:{R}\nF(count,n)=Q(n);\nend\ntnext=t(c)*2;\nfor n = 1:' + str(N) + '\nz(count,n) = y(c,n);\nend\ncount = count + 1;\nend\nend\n'
    output += f'for n = 1:' + str(N) + f'\nz(count,n)=y(w,n);\nend\ntau(count,1)=t(w);\nQ=Reaction(y(w,:));\nfor n=1:{R}\nF(count,n)=Q(n);\nend\n'
    output += 'file=sprintf(\'../DynamicsData/flux%d.dat\',init);\ndlmwrite(file,[tau F],\'precision\', \'%e\',\'delimiter\',\' \');\n'
    output += 'file=sprintf(\'../DynamicsData/conc%d.dat\',init);\ndlmwrite(file,[tau z],\'precision\', \'%e\',\'delimiter\',\' \');\nend\n'



    with open(OutPutDir+'matlabscript/run.m','w') as fp:
        fp.write(output)
    



if __name__ == '__main__':
    DataFile = '../../Model_Comparison/Boecker_Model/ModelData/Ralf_KineticModel_Version2.xlsx'
    MainLoop(DataFile)



