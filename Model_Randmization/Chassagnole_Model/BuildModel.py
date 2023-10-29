import pandas as pd
import numpy as np
import random, os, sys
sys.path.append('../../CommonModule')
import settings


def GenerateInitialconditions(seed,Xref,INIT_MAX,Folder):

    metabolites = settings.metabolites_chassagnole

    At = Xref[metabolites.index('cadp')] + Xref[metabolites.index('catp')]

    random.seed(seed)
    N = len(metabolites)
   
    Xini = np.zeros((INIT_MAX,N))
    for ini in range(INIT_MAX):
        for i in range(N):
            Xini[ini,i] = Xref[i]*random.uniform(1.0 - settings.noiselevel_chassagnole,1.0 + settings.noiselevel_chassagnole)
    
        Xini[ini,metabolites.index('catp')] = At - Xini[ini,metabolites.index('cadp')]
        Z = Xini[ini,metabolites.index('catp')] + Xini[ini,metabolites.index('cadp')]
        Xini[ini,metabolites.index('catp')] = Xini[ini,metabolites.index('catp')]/Z*At
        Xini[ini,metabolites.index('cadp')] = Xini[ini,metabolites.index('cadp')]/Z*At
        
    np.savetxt(Folder+'initials.txt',Xini)


def AddReactions(R,AddRxn,metabolites,param,eq):
    #Generate the substrates set
    ExRxnMetPair, Coupling = [], []
    metabolite_candidates = [x for x in metabolites if x not in ['catp','cadp']]
    for i in range(AddRxn):
        while 1:
            mets = set(random.sample(metabolite_candidates,2))
            if mets not in ExRxnMetPair:
                ExRxnMetPair.append(mets)
                if random.random() < 4.0/R:
                    Coupling.append(['catp','cadp'])
                else:
                    Coupling.append([])
                break
    
    UniUniRxn, BiBiRxn = ['TIS','PGluMu','ENO','PGM','PGI','Ru5P','R5PI'], ['GAPDH','PGK','ALDO','TKa','TKb','TA','PGDH','G6PDH','DAHPS','GIPAT','PFK']
    v_values_UniUni, v_values_BiBi, K_values, Eq_values_UniUni,  Eq_values_BiBi = [param[x] for x in param.keys() if x[:4]=='rmax' and any([y in x for y in UniUniRxn])],  [param[x] for x in param.keys() if x[:4]=='rmax' and any([y in x for y in BiBiRxn])], [param[x] for x in param.keys() if x[0] in ['k','K'] and 'eq' not in x], [param[x] for x in param.keys() if x[0] in ['k','K'] and 'eq' in x and any([y in x for y in UniUniRxn])], [param[x] for x in param.keys() if x[0] in ['k','K'] and 'eq' in x and any([y in x for y in BiBiRxn])]
    
    #Generate the reaction equations and set parameters
    for add_idx, chem in enumerate(zip(ExRxnMetPair,Coupling)):
        met, cpl = chem[0], chem[1]
        #Sample Parameter Values
        v_values = v_values_UniUni if cpl == [] else v_values_BiBi
        Eq_values = Eq_values_UniUni if cpl == [] else Eq_values_BiBi
        if cpl == []:
            param[f'v_add{add_idx}'], param[f'Ka_add{add_idx}'], param[f'Kb_add{add_idx}'], param[f'Keq_add{add_idx}'], = random.sample(v_values,1)[0], *random.sample(K_values,2), random.sample(Eq_values,1)[0]
        else:
            param[f'v_add{add_idx}'], param[f'Ka_add{add_idx}'], param[f'Kb_add{add_idx}'], param[f'Kc_add{add_idx}'], param[f'Kd_add{add_idx}'], param[f'Keq_add{add_idx}'], = random.sample(v_values,1)[0], *random.sample(K_values,4), random.sample(Eq_values,1)[0]
        #Reaction Rate Equation
        reactant = sorted(list(met))
        if random.random() < 0.5:
            reactant.reverse()
        if random.random() < 0.5:
            cpl.reverse()
        
        if cpl == []:
            J = f'v_add{add_idx}*(x({metabolites.index(reactant[0])+1}) - x({metabolites.index(reactant[1])+1})/Keq_add{add_idx})/(x({metabolites.index(reactant[0])+1}) + Ka_add{add_idx}*(1+x({metabolites.index(reactant[1])+1})/Kb_add{add_idx}))'
        else:
            J = f'v_add{add_idx}*(x({metabolites.index(reactant[0])+1})*x({metabolites.index(cpl[0])+1}) - x({metabolites.index(reactant[1])+1})*x({metabolites.index(cpl[1])+1})/Keq_add{add_idx})/(x({metabolites.index(reactant[0])+1}) + Ka_add{add_idx}*(1+x({metabolites.index(cpl[0])+1})/Kb_add{add_idx}))/(x({metabolites.index(reactant[1])+1}) + Kc_add{add_idx}*(1+x({metabolites.index(cpl[1])+1})/Kd_add{add_idx}))'

            eq[cpl[0]] += ' - ' + J + ' '
            eq[cpl[1]] += ' + ' + J + ' '
        
        eq[reactant[0]] += ' - ' + J + ' '
        eq[reactant[1]] += ' + ' + J + ' '
        
    return param, eq

def MainLoop(seed,AddRxn,INIT_MAX):

    for folder in [f'SimulationResult/Model{seed}/Add{AddRxn}/matlabscript',f'SimulationResult/Model{seed}/Add{AddRxn}/DynamicsData',f'SimulationResult/Model{seed}/Add{AddRxn}/Analysis']:
        os.makedirs(folder,exist_ok=True)

    random.seed(seed)
    
    ParameterFile, EquationFile = '../../Model_Comparison/Chassagnole_Model/Parameters.txt', '../../Model_Comparison/Chassagnole_Model/Equation.txt'
    df = pd.read_table(ParameterFile)

    param, conc = {}, {}
    for i in range(len(df)):
        m = df['Quantity Name'][i]
        v = df['Value'][i]
        t = df['Type'][i]
        if t == 'species' and m != 'X':
            conc[m] = v 
        if t == 'parameter':
            param[m] = v

    eq = {}
    with open(EquationFile,'r') as fp:
        for line in fp:
            l = line.replace('\n','').replace(' ','').split(',')
            eq[l[0]] = l[1].replace('**','^')

    #print(len(conc),len(eq))
    #for m in conc.keys():
    #    print(m,eq[m])


    param['Km_NGAM'] = 1.0
    param['v_NGAM'] = 0.1


    metabolites = [(m,len(m)) for m in list(conc.keys())]
    metabolites = sorted(metabolites, key=lambda x:x[1], reverse=True)
    metabolites = [m[0] for m in metabolites]
    replacelist = [(m,f'x({i})') for i,m in enumerate(metabolites,start=1)]
    for i,n in enumerate(metabolites,start=1):
        for r in replacelist:
            eq[n] = eq[n].replace(*r) 

    AdditionalReactionNumbers = [1,2,4,6]
    param, eq = AddReactions(len([param[x] for x in param.keys() if 'rmax' in x]),AdditionalReactionNumbers[AddRxn],metabolites,param,eq)
    
    #パラメーター
    output = ''
    for i in param.keys():
        output += i + '=' + str(param[i]) + ';\n'
    with open(f'SimulationResult/Model{seed}/Add{AddRxn}/matlabscript/parameters.m','w') as fp:
        fp.write(output)

    output = ''
    for i,n in enumerate(metabolites,start=1):
        output += '%'+n+'\n'
        output += f'dx{i}_dt = @(t,x) ' + eq[n] + ';\n'


    #Dynamicsをまず計算したあとのConc
    yatt = [1.059937e-01,3.394233e+01,3.961649e-01,7.053498e-02,1.386998e-01,6.560053e-01,2.037646e-01,3.134937e-01,4.137934e+00,2.508028e-01,1.818086e+00,8.929278e-01,1.035190e-03,2.111069e-04,1.547474e-03,5.588319e-02,2.188191e+00,4.864751e+00,2.481806e-04,4.616661e-01]
    for i,m in enumerate(metabolites):
        conc[m] = yatt[i]

    #yatt = [1 for i in range(len(metabolites))]
    #for i,m in enumerate(metabolites):
    #    yatt[i] = conc[m]

    output += 'df_dt = @(t,x)[' + ';'.join([f'dx{i}_dt(t,x)' for i in range(1,len(metabolites)+1)]) + '];\n'
    with open(f'SimulationResult/Model{seed}/Add{AddRxn}/matlabscript/ODEs.m','w') as fp:
        fp.write(output)

    GenerateInitialconditions(seed,yatt,INIT_MAX,f'SimulationResult/Model{seed}/Add{AddRxn}/matlabscript/')

    #Main Script
    N = len(metabolites)
    output = 'clearvars\nxini = importdata(\'initials.txt\');\nNini=size(xini,1);\nparfor init=1:Nini\ndisp(init)\nSingle(init);\nend\ndlmwrite(\'done.txt\',[1 1]);\n\n\n'
    #output = 'clearvars\nfor init=1:'+str(INIT_MAX)+'\ndisp(init)\nSingle(init);\nend\npoolobj = gcp(\'nocreate\');\ndelete(poolobj);\n\n'
    output += 'function Single(init)\n'
    output += 'xini = importdata(\'initials.txt\');\ndisp(init)\nparameters;ODEs;'
    output += 'y0 = [' + ' '.join([f'xini(init,{n+1})' for n in range(N)])+'];\n'
    output += 'tstart = tic;\n'
    output += 'options = odeset(\'RelTol\',1e-4,\'AbsTol\',1e-6,\'OutputFcn\', @(t,y,flag) myOutPutFnc(t,y,flag,tstart));\n'
    output += '[t,y] = ode15s(df_dt,[0,1e+8],y0,options);\n'

    output += 'clear tau z\n'
    output += 'w=size(y,1);\n'
    output += f'if t(w) < 1e7\n'
    output += 'msg = sprintf(\'terminated %d\',init);\ndisp(msg);\nclear t y;\n'
    output += 'return\nend\n'


    output += 'for n = 1:' + str(N) + '\nz(1,n)=y(2,n);\nend\ntau(1,1)=t(2);\n'
    output += 'count=2;\n'
    output += 'for c = 3:w\n'
    output += 'dist = 0;\ntnext=1;\n'
    output += 'for n = 1:' + str(N) + '\ndist = dist + (log(y(c,n)) - log(z(count-1,n)))^2;\nend\ndist = sqrt(dist);\n'
    output += 'if dist > 1e-1 | t(c) > tnext\ntau(count,1)=t(c);\ntnext=t(c)*2;\nfor n = 1:' + str(N) + '\nz(count,n) = y(c,n);\nend\ncount = count + 1;\nend\nend\n'
    output += 'for n = 1:' + str(N) + '\nz(count,n)=y(w,n);\nend\ntau(count,1)=t(w);\n'
    output += 'file=sprintf(\'../DynamicsData/conc%d.dat\',init);\ndlmwrite(file,[tau z],\'precision\', \'%e\',\'delimiter\',\' \');\nend\n'

    output += '\n\n'
    output += 'function states = myOutPutFnc(t,y,flag,tstart);\nTmax = 60;\nstates=0;\nswitch(flag)\ncase []\nif toc(tstart) > Tmax\nstates = 1;\nend\nend\nend\n'

    with open(f'SimulationResult/Model{seed}/Add{AddRxn}/matlabscript/run.m','w') as fp:
        fp.write(output)



if __name__ == '__main__':
    INIT_MAX = 128
    ADD_MAX = 7 
    for add in range(ADD_MAX):
        MainLoop(0,add,INIT_MAX)
    with open(f'Model{0}/runall.m','w') as fp:
        for i in range(ADD_MAX):
            fp.write(f'run(\'Add{i}/matlabscript/run\');\n')