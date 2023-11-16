import pandas as pd
import sympy, random
from tqdm import tqdm
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os

INIT_MAX = 4096

def GenerateInitials(Attractor,VariablesInPaper,Strength):
    random.seed(0)
    N = len(VariablesInPaper)
    Conservation = [sympy.sympify('2 - NADH'),sympy.sympify('3.67 - AcCoA'),sympy.sympify('2.70058 - ADP - ATP')]
    NADt, CoAt, At = 2.0, 3.67, 2.70058
    Xini = np.zeros((INIT_MAX,len(Attractor)))
    for i in tqdm(range(INIT_MAX)):
        x = []
        for j in range(N):
            #x.append(Attractor[j]*random.uniform(1./PTB,PTB))
            #while 1:
            #g = np.random.normal(1.0,Strength)
            g = 1 + random.uniform(-Strength,Strength)
            #if g > 0 and abs(g-1) > 0.05:
            #    break
            #g = random.uniform(1.0-Strength,1.0+Strength)
            x.append(Attractor[j]*g)
        #Conservation Quantities
        NADH, AcCoA, ATP, ADP = Attractor[VariablesInPaper.index('NADH')], Attractor[VariablesInPaper.index('AcCoA')], Attractor[VariablesInPaper.index('ATP')], Attractor[VariablesInPaper.index('ADP')]
        NAD, CoA, AMP = NADt - NADH, CoAt - AcCoA, At - ATP - ADP
        #random
        NADHr, AcCoAr, ATPr, ADPr, NADr, CoAr, AMPr = NADH*random.uniform(1.-Strength,1.+Strength), AcCoA*random.uniform(1.-Strength,1.+Strength), ATP*random.uniform(1.-Strength,1.+Strength), ADP*random.uniform(1.-Strength,1.+Strength), NAD*random.uniform(1.-Strength,1.+Strength), CoA*random.uniform(1.-Strength,1.+Strength), AMP*random.uniform(1.-Strength,1.+Strength) 
        x[VariablesInPaper.index('ATP')], x[VariablesInPaper.index('ADP')] = At*ATPr/(ATPr + ADPr + AMPr), At*ADPr/(ATPr + ADPr + AMPr)
        x[VariablesInPaper.index('NADH')] = NADt*NADHr/(NADr + NADHr)
        x[VariablesInPaper.index('AcCoA')] = CoAt*AcCoAr/(CoAr + AcCoAr)

        subslist = [(met,x[VariablesInPaper.index(met)]) for met in ['NADH','ADP','ATP','AcCoA']]
        #if all([float(Q.subs(subslist)) > 1e-4 for Q in Conservation]):
        if any([float(Q.subs(subslist)) < 1e-10 for Q in Conservation]):
            print('too low')
            exit()
        #    break
        for j in range(N):
            Xini[i,j] = x[j]

    np.savetxt('matlabscript/initials.txt',Xini,delimiter=' ')


def MainLoop(DataFile,Var2Const=[],INIT_MAX=1024):

    os.makedirs('matlabscript',exist_ok=True)

    #Variables
    VariablesInPaper = ['FORin','ACEin','ETHin','LACin','SUCin','G6P','F6P','FBP','GAP','BPG','D3PG','PEP','PYR','AcCoA','ACTLD','OAA','AKG','FUM','ATP','ADP','NADH']
    Attractor = [0.6052,0.2702,0.2584,0.0107,0.0230,4.1176,0.9080,9.7478,0.0729,0.1425,1.4275,0.2730,1.1126,3.4554,0.3543,0.5595,0.4949,0.1235,2.4911,0.1782,0.0977]

    #External metabolites
    ExternalChemicals = ['FOR','LAC','ACE','ETH','SUC','Biomass'] + ['s'+str(i) for i in range(50)]
    #remove because the followig reactions are for monitering the accumulation of external metabolites
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

    #Check if import is correctory done
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


    #Make ODEs
    #===============================================================================================================================================================================================================================================================================
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

    # Maybe the following stoichiometry data are only on the Appendix pdf file
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

    output = ''
    for i in Parameters.keys():
        output += i + '=' + str(Parameters[i]['value']) + ';\n'
    with open('matlabscript/parameters.m','w') as fp:
        fp.write(output)


    #ODE
    output = ''
    for rxn in UsedReactions:
        if rxn[:2] == 'D_':
            continue
        func = 'Function_for_'+rxn
        
        subslist = [(i,f'x({VariablesInPaper.index(i)+1})') for i in Functions[func]['args'] if i in VariablesInPaper]
        output += '%'+rxn+'\n'
        output += 'J_'+rxn+' = @(x) '+str(Functions[func]['math'].subs(subslist)) + ';\n\n'
    with open('matlabscript/reactions.m','w') as fp:
        fp.write(output.replace('**','^'))

    output = ''
    for i,n in enumerate(VariablesInPaper,start=1):
        output += '%'+n+'\n'
        output += f'dx{i}_dt = @(t,x) ' + ' '.join([str(Stoich[n,rxn])+'*J_'+rxn + '(x)' for rxn in RelatedReactions[n]['Consumed'] if rxn[:2] != 'D_']) + ' ' + ' '.join(['+'+str(Stoich[n,rxn])+'*J_'+rxn+'(x)' for rxn in RelatedReactions[n]['Produced']]) + f' -J_Growth(x)*x({i});\n\n'

    output += 'df_dt = @(t,x)[' + ';'.join([f'dx{i}_dt(t,x)' for i in range(1,len(VariablesInPaper)+1)]) + '];\n'
    with open('matlabscript/ODEs.m','w') as fp:
        fp.write(output)

    #Initial points
    GenerateInitials(Attractor,VariablesInPaper,0.4)

    #Main Script
    N = len(VariablesInPaper)
    output = 'clearvars\nparfor init=1:'+str(INIT_MAX)+'\ndisp(init)\nSingle(init);\nend\npoolobj = gcp(\'nocreate\');\ndelete(poolobj);\n\n'
    output += 'function Single(init)\n'
    output += 'xini = importdata(\'initials.txt\');\ndisp(init)\nparameters;reactions;ODEs;Attractor;\n'
    output += 'y0 = [' + ' '.join([f'xini(init,{n+1})' for n in range(N)])+'];\n'
    output += 'options = odeset(\'RelTol\',1e-4,\'AbsTol\',1e-6);\n'
    output += '[t,y] = ode15s(df_dt,[0,1e+4],y0,options);\n'

    output += 'clear tau z\n'
    output += 'w=size(y,1);\n'
    output += 'for n = 1:' + str(N) + '\nz(1,n)=y(2,n)/yatt(n);\nend\ntau(1,1)=t(2);\ng(1,1)=J_Growth(y(2,:));\n'
    output += 'count=2;\n'
    output += 'for c = 3:w\n'
    output += 'dist = 0;\ntnext=1;\n'
    output += 'for n = 1:' + str(N) + '\ndist = dist + (log(y(c,n)) - log(z(count-1,n)))^2;\nend\ndist = sqrt(dist);\n'
    output += 'if dist > 1e-1 | t(c) > tnext\ntau(count,1)=t(c);\ng(count,1)=J_Growth(y(c,:));\ntnext=t(c)*2;\nfor n = 1:' + str(N) + '\nz(count,n) = y(c,n)/yatt(n);\nend\ncount = count + 1;\nend\nend\n'
    output += 'for n = 1:' + str(N) + '\nz(count,n)=y(w,n)/yatt(n);\nend\ntau(count,1)=t(w);\ng(count,1)=J_Growth(y(w,:));\n'
    output += 'file=sprintf(\'../DynamicsData/conc%d.dat\',init);\ndlmwrite(file,[tau z],\'precision\', \'%e\',\'delimiter\',\' \');\n'
    output += 'file=sprintf(\'../DynamicsData/growth%d.dat\',init);\ndlmwrite(file,[tau g],\'precision\', \'%e\',\'delimiter\',\' \');\nend\n'

    with open('matlabscript/run.m','w') as fp:
        fp.write(output)


    with open('matlabscript/Attractor.m','w') as fp:
        fp.write(''.join([f'yatt({i})={y};\n' for i,y in enumerate(Attractor,start=1)]))    





if __name__ == '__main__':
    MainLoop(DataFile='ModelData/Boecker_KineticModel_Version2.xlsx')