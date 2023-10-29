import pandas as pd
import numpy as np
import random, sys, os
sys.path.append('../../CommonModule')
import settings


def BuildModel(SourceDir='ModelData/',Var2Const=[],INIT_MAX=4096,Attractor=settings.attractor_chassagnole):

    #If atp or adp is constant, then catp and cadp are also constant
    if 'catp' in Var2Const or 'cadp' in Var2Const:
        Var2Const = ['catp','cadp']    

    for folder in ['DynamicsData','matlabscript','Analysis']:
        os.makedirs(folder,exist_ok=True)

    df = pd.read_table(SourceDir+'Parameters.txt')

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
    with open(SourceDir+'Equation.txt','r') as fp:
        for line in fp:
            l = line.replace('\n','').replace(' ','').split(',')
            eq[l[0]] = l[1].replace('**','^')

    print(len(conc),len(eq))
    for m in conc.keys():
        print(m,eq[m])


    param['Km_NGAM'] = 1.0
    param['v_NGAM'] = 0.1

    #パラメーター
    output = ''
    for i in param.keys():
        output += i + '=' + str(param[i]) + ';\n'
    with open('matlabscript/parameters.m','w') as fp:
        fp.write(output)

    metabolites = [(m,len(m)) for m in list(conc.keys())]
    metabolites = sorted(metabolites, key=lambda x:x[1], reverse=True)
    metabolites = [m[0] for m in metabolites]
    replacelist = [(m,f'x({i})') for i,m in enumerate(metabolites,start=1)]
    for i,n in enumerate(metabolites,start=1):
        for r in replacelist:
            eq[n] = eq[n].replace(*r) 

    output = ''
    for i,n in enumerate(metabolites,start=1):
        output += '%'+n+'\n'
        if n in Var2Const:
            output += f'dx{i}_dt = @(t,x) 0;\n'
        else:
            output += f'dx{i}_dt = @(t,x) ' + eq[n] + ';\n'


    #Dynamicsをまず計算したあとのConc
    yatt = [1.059937e-01,3.394233e+01,3.961649e-01,7.053498e-02,1.386998e-01,6.560053e-01,2.037646e-01,3.134937e-01,4.137934e+00,2.508028e-01,1.818086e+00,8.929278e-01,1.035190e-03,2.111069e-04,1.547474e-03,5.588319e-02,2.188191e+00,4.864751e+00,2.481806e-04,4.616661e-01]
    for i,m in enumerate(metabolites):
        conc[m] = yatt[i]

    #yatt = [1 for i in range(len(metabolites))]
    #for i,m in enumerate(metabolites):
    #    yatt[i] = conc[m]

    #Parameter output for the Furusawamodel
    with open('../../FurusawaModel_DistributedParameter/Chassagnole_Model/ModelData/v_params.txt','w') as fp:
        fp.write(','.join([f'{v_val}' for v_name,v_val in param.items() if v_name.startswith('v_') or v_name.startswith('rmax')]))  

    output += 'df_dt = @(t,x)[' + ';'.join([f'dx{i}_dt(t,x)' for i in range(1,len(metabolites)+1)]) + '];\n'
    with open('matlabscript/ODEs.m','w') as fp:
        fp.write(output)

    At = conc['cadp'] + conc['catp']

    random.seed(0)
    N = len(metabolites)
    
    Xini = np.zeros((INIT_MAX,N))
    for ini in range(INIT_MAX):
                
        for i in range(N):
            Xini[ini,i] = yatt[i]*random.uniform(1.0-settings.noiselevel_chassagnole,1.0+settings.noiselevel_chassagnole)
        Z = Xini[ini,metabolites.index('catp')] + Xini[ini,metabolites.index('cadp')]
        Xini[ini,metabolites.index('catp')] = Xini[ini,metabolites.index('catp')]/Z*At
        Xini[ini,metabolites.index('cadp')] = Xini[ini,metabolites.index('cadp')]/Z*At
        
        for met in Var2Const:
            Xini[ini,metabolites.index(met)] = yatt[metabolites.index(met)]
        

    np.savetxt('matlabscript/initials.txt',Xini)

    #Main Script
    N = len(metabolites)
    output = 'clearvars\nparfor init=1:'+str(INIT_MAX)+'\ndisp(init)\nSingle(init);\nend\npoolobj = gcp(\'nocreate\');\ndelete(poolobj);\n\n'
    #output = 'clearvars\nfor init=1:'+str(INIT_MAX)+'\ndisp(init)\nSingle(init);\nend\npoolobj = gcp(\'nocreate\');\ndelete(poolobj);\n\n'
    output += 'function Single(init)\n'
    output += 'xini = importdata(\'initials.txt\');\ndisp(init)\nparameters;ODEs;Attractor;\n'
    output += 'y0 = [' + ' '.join([f'xini(init,{n+1})' for n in range(N)])+'];\n'
    output += 'options = odeset(\'RelTol\',1e-4,\'AbsTol\',1e-6);\n'
    output += '[t,y] = ode15s(df_dt,[0,1e+8],y0,options);\n'

    output += 'clear tau z\n'
    output += 'w=size(y,1);\n'
    output += 'for n = 1:' + str(N) + '\nz(1,n)=y(2,n)/yatt(n);\nend\ntau(1,1)=t(2);\n'
    output += 'count=2;\n'
    output += 'for c = 3:w\n'
    output += 'dist = 0;\ntnext=1;\n'
    output += 'for n = 1:' + str(N) + '\ndist = dist + (log(y(c,n)) - log(z(count-1,n)))^2;\nend\ndist = sqrt(dist);\n'
    output += 'if dist > 1e-1 | t(c) > tnext\ntau(count,1)=t(c);\ntnext=t(c)*2;\nfor n = 1:' + str(N) + '\nz(count,n) = y(c,n)/yatt(n);\nend\ncount = count + 1;\nend\nend\n'
    output += 'for n = 1:' + str(N) + '\nz(count,n)=y(w,n)/yatt(n);\nend\ntau(count,1)=t(w);\n'
    output += 'file=sprintf(\'../DynamicsData/conc%d.dat\',init);\ndlmwrite(file,[tau z],\'precision\', \'%e\',\'delimiter\',\' \');\nend\n'

    with open('matlabscript/run.m','w') as fp:
        fp.write(output)



    with open('matlabscript/Attractor.m','w') as fp:
        fp.write('\n'.join([f'yatt({i})={conc[m]};' for i,m in enumerate(metabolites,start=1)]))



if __name__ == '__main__':
    BuildModel()