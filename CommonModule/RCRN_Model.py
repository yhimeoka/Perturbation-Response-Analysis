import random
import glob
import joblib
import itertools
import os
import time
import networkx as nx
import numpy as np
from tqdm import tqdm


def ComputeAttractor(N, datadir):
    INIT_MAX = 32
    Count = 0
    skip = False
    Xall = []
    Attractor = [0 for i in range(N)]
    for init in range(1, INIT_MAX+1):
        with open(datadir+f'/conc{init}.dat', 'r') as fp:
            lall = []
            for line in fp:
                l = line.replace('\n', '').replace('i', 'j').split()
                try:
                    l = [float(y) for y in l]
                except ValueError:
                    skip = True
                    break
                lall.append(l[:])

        if skip:
            continue
        L = len(lall)
        t = lall[L-1].pop(0)
        tb = lall[L-2].pop(0)

        if t < 5e+6:
            continue

        try:
            delta = [abs(lall[L-1][i]-lall[L-2][i])/abs(t-tb)*t/lall[L-1][i] for i in range(N)]
        except ZeroDivisionError:
            continue

        if max(delta) > 1e-2:
            continue

        if max([abs(complex(y).imag) for y in l]) > 1e-16:
            print('Imaginaly Number Error')
            print(l)
            exit()

        x = [complex(y).real for y in l]

        if any([y < 0 for y in x]):
            print(f'minus {init}')
            continue

        Count += 1
        x = [np.log(complex(y).real) for y in l]
        x.pop(0)
        Xall.append(x)
        for i in range(N):
            Attractor[i] += x[i]

    if Count < 4:
        return [],  [0],  False

    for i in range(N):
        Attractor[i] /= Count

    MonoStable = max([np.linalg.norm(np.array(Attractor) - np.array(Xall[i])) for i in range(Count)]) < 1e-3

    if MonoStable:
        return [np.exp(y) for y in Attractor],  [Count],  MonoStable
    else:
        Xall = [np.array(y) for y in Xall]
        AttIdx = [0 for i in range(Count)]
        Population = [0 for i in range(Count)]
        Attractors = [np.copy(Xall[0])]
        for i in range(1, Count):
            dist_to_att = [np.linalg.norm([att-Xall[i]]) for att in Attractors]
            MinDist,  MinIdx = min(dist_to_att),  np.argmin(np.array(dist_to_att)) 
            if MinDist > 1:
                AttIdx[i] = len(Attractors)
                Population[len(Attractors)] += 1
                Attractors.append(Xall[i])
            else:
                AttIdx[i] = MinIdx
                Population[MinIdx] += 1
                L = len([j for j in range(i+1) if AttIdx[j] == MinIdx])
                Attractors[MinIdx] = (L-1)*Attractors[MinIdx]/L + Xall[i]/L
        
        print(f'{max(AttIdx)+1} Attractors,  population = ', [len([i for i in range(Count) if AttIdx[i] == j]) for j in range(max(AttIdx)+1)])
        LargestAttIdx = np.argmax([len([i for i in range(Count) if AttIdx[i] == j]) for j in range(max(AttIdx)+1)])
        return [np.exp(y) for y in Attractors[LargestAttIdx]],  [_ for _ in Population if _ != 0],  True


def GenerateRandomInitials(N, matlabdir):
    
    INIT_MAX = 32
    random.seed(0)
    Xini = np.zeros((INIT_MAX, N))
    for i in range(INIT_MAX):
        for j in range(N):
            Xini[i, j] = 10.0**random.uniform(-2, 2)

    np.savetxt(matlabdir+'/initials.txt', Xini.T, delimiter=' ')


def GeneratePerturbedInitials(N, Attractor, matlabdir, Strength):

    INIT_MAX = 128
    random.seed(0)
    Xini = np.zeros((INIT_MAX, len(Attractor)))
    for i in tqdm(range(INIT_MAX)):
        for j in range(N):
            Xini[i, j] = Attractor[j]*random.uniform(1.0-Strength, 1.0+Strength)
    
    np.savetxt(matlabdir+'/initials.txt', Xini.T, delimiter=' ')


def ImportModelParameters(DataFilePath):
    
    Rates = []
    with open(DataFilePath, 'r') as fp:
        l = fp.readline().replace('\n', '').split(', ')
        for v in l:
            if abs(float(v)) > 1e-10:
                Rates.append(np.log(float(v)))
    Param_Max,  Param_Min = max(Rates),  min(Rates)
    BIN = 32
    dx = (Param_Max-Param_Min)/BIN
    dx_param = dx
    ParamDist = [[Param_Min + dx*i, 0] for i in range(BIN+1)]
    for x in Rates:
        ParamDist[round((x-Param_Min)/dx)][1] += 1.0/len(Rates)

    return ParamDist,  dx_param


def MakeDirectories(N, R, loop):
    os.makedirs(f'NetworkN{N}R{R}/Structure', exist_ok=True)
    os.makedirs(f'NetworkN{N}R{R}/AnalyzeResult', exist_ok=True)
    os.makedirs(f'NetworkN{N}R{R}/Dist', exist_ok=True)
    
    matlabdir = f'NetworkN{N}R{R}/matlabscript{loop}'
    os.makedirs(matlabdir, exist_ok=True)
    datadir = f'NetworkN{N}R{R}/DynamicsData{loop}'
    os.makedirs(datadir, exist_ok=True)    
    os.system('rm ' + datadir + '/*dat') #Just in case for when the previous simulation was performed

    return matlabdir,  datadir


def ConstructNetwork(Variables, N, R, loop):
    while 1:
        G = nx.Graph()
        G.add_nodes_from(Variables)
        
        AllReactions = list(itertools.combinations(Variables, 2))
        random.shuffle(AllReactions)
        Reactions = AllReactions[:R]
        G.add_edges_from(Reactions)
        Degree = [G.degree[i] for i in range(N)]
        if nx.is_connected(G) and all([G.degree[i] > 1.5 for i in range(1, N)]):
            break
    
    nx.write_edgelist(G,  f'NetworkN{N}R{R}/Structure/Net{loop}.edgelist')
    
    Catalysts = []
    for r in range(R):
        while 1:
            c = random.randint(1, N-2)
            if c not in Reactions[r]:
                break
        Catalysts.append(c)

    return Reactions,  Catalysts


def ParameterAssignment(matlabdir, R, ParamDist, dx_param):
    LogParam = []
    Parameters = {}
    for r in range(R):
        for pname in [f'V{r}p',  f'V{r}m']:  
            #Parameters[pname] = 10**random.uniform(-1.3, 2.0)
            rho = random.uniform(0, 1)
            h = 0
            for dist in ParamDist:
                if rho > h and h + dist[1] > rho:
                    p = dist[0] + dx_param*random.uniform(0, 1)
                    Parameters[pname] = np.exp(p)
                    LogParam.append(p)
                    break
                h += dist[1]
            
    ParameterNames = sorted(list(Parameters.keys()))
    output = ''
    for i in Parameters.keys():
        output += i + '=' + str(Parameters[i]) + ';\n'
    output += 'kinetic_param = [' + ';'.join([str(Parameters[i]) for i in ParameterNames]) + '];\n'
    with open(matlabdir+f'/parameters.m', 'w') as fp:
        fp.write(output)

    return ParameterNames,  Parameters


def SetupODEs(matlabdir, Reactions, Catalysts, N, R):

    # prepare reactions string
    output = ''
    for r in range(R):
        s,  p,  c = Reactions[r][0]+1,  Reactions[r][1]+1,  Catalysts[r]+1
        output += f'J{r}p = @(x) V{r}p*x({s})*x({c});\n'
        output += f'J{r}m = @(x) V{r}m*x({p})*x({c});\n'
    output += f'Uptake = @(x) 1.0*x({N});\n'
    
    with open(matlabdir+'/reactions.m', 'w') as fp:
        fp.write(output)

    # prepare ODEs string
    output = ''
    for i in range(N):
        output += f'dx{i+1}_dt = @(t, x) '
        if i == 0:
            output += ' + Uptake(x) '
        
        for r in range(R):
            s,  p = Reactions[r][0],  Reactions[r][1]
            if i == s:
                output += f' - (J{r}p(x) - J{r}m(x)) '
            if i == p:
                output += f' + (J{r}p(x) - J{r}m(x)) '

        output += f' - Uptake(x)*x({i+1});\n'

    output += 'df_dt = @(t, x)[' + ';'.join([f'dx{i+1}_dt(t, x)' for i in range(N)]) + '];\n'
    with open(matlabdir+'/ODEs.m', 'w') as fp:
        fp.write(output)


def SetupJacobiMatrix(matlabdir, Reactions, Catalysts, ParameterNames, N, R):

    def SingleJacobi(species, Reactions, ParameterNames, N, R):
        Jac = ['' for j in range(N)]
        if species == 0:
            Jac[N-1] += ' + 1.0 '
            
        for r in range(R):
            s,  p,  c = Reactions[r][0],  Reactions[r][1],  Catalysts[r]
            if species == s:
                ip,  im = ParameterNames.index(f'V{r}p') + 1,  ParameterNames.index(f'V{r}m') + 1
                Jac[s] += f' + (-kinetic_param({ip})*x({c+1}))'
                Jac[p] += f' + (kinetic_param({im})*x({c+1}))'
                Jac[c] += f' + (kinetic_param({im})*x({p+1}) - kinetic_param({ip})*x({s+1}))'
                
            if species == p:
                ip,  im = ParameterNames.index(f'V{r}p') + 1,  ParameterNames.index(f'V{r}m') + 1
                Jac[s] += f' - (-kinetic_param({ip})*x({c+1}))'
                Jac[p] += f' - (kinetic_param({im})*x({c+1}))'
                Jac[c] += f' - (kinetic_param({im})*x({p+1}) - kinetic_param({ip})*x({s+1}))'

        if species == N-1:
            Jac[N-1] += f' - 2*x({N}) '
        else:
            Jac[N-1] += f' - x({species+1}) '
            Jac[species] += f' - x({N}) '
        return Jac

    Jacobi = joblib.Parallel(n_jobs=-1)(joblib.delayed(SingleJacobi)(species, Reactions, ParameterNames, N, R) for species in range(N))
    
    output = f'function J = Jacobi(t, x, kinetic_param)\nJ=zeros({N}, {N});\n'
    for i in range(N):
        for j in range(N):
            if Jacobi[i][j] != '':
                output += f'J({i+1}, {j+1}) = ' + Jacobi[i][j] + ';\n'
    output += 'end\n'    

    with open(matlabdir+'/jacobi.m', 'w') as fp:
        fp.write(output)


def SetupRunningScript(matlabdir, N, loop, parallel=True):

    # Main Script
    output = 'clearvars\nxini = importdata(\'initials.txt\');\nw=size(xini);\nINIT_MAX=w(2);\n'
   
    if parallel:
        output += 'parfor init=1:INIT_MAX\nSingle(init);\nend\npoolobj = gcp(\'nocreate\');\ndelete(poolobj);\n\n'
    else:
        output += 'for init=1:INIT_MAX\nSingle(init);\nend\n\n'
    
    output += 'function Single(init)\n'
    output += f'xini = importdata(\'initials.txt\');\ndisp([{loop} init])\nparameters;reactions;ODEs;\n'
    output += 'y0 = [' + ' '.join([f'xini({n+1}, init)' for n in range(N)])+'];\n'
    output += 'options = odeset(\'RelTol\', 1e-3, \'AbsTol\', 1e-5, \'Jacobian\', @(t, y)jacobi(t, y, kinetic_param));\n'
            
    output += '[t, y] = ode15s(df_dt, [0, 1e+7], y0, options);\n'
    
    output += 'disp(init);\nclear tau z\nw=size(y, 1);\n'
    output += 'for n = 1:' + str(N) + '\nz(1, n)=y(2, n);\nend\ntau(1, 1)=t(2);\n'
    output += 'count=2;\nfor c = 3:w\ndist = 0;\ntnext=1e-4;\n'
    output += 'for n = 1:' + str(N) + '\ndist = dist + (log(y(c, n)) - log(z(count-1, n)))^2;\nend\ndist = sqrt(dist);\n'
    output += 'if dist > 1e-1 | t(c) > tnext\ntau(count, 1)=t(c);\ntnext=t(c)*2;\nfor n = 1:' + str(N) + '\nz(count, n) = y(c, n);\nend\ncount = count + 1;\nend\nend\n'
    output += f'file=sprintf(\'../DynamicsData{loop}/conc%d.dat\', init);\ndlmwrite(file, [tau z], \'precision\',  \'%e\', \'delimiter\', \' \');\nend\n'
            
    with open(matlabdir+'/run.m', 'w') as fp:
        fp.write(output)

       
def RunMatlab(WAIT_MAX, datadir, matlabdir, FILENUM_MAX):
    dt = 5
    waittime = 0
    os.system('(cd ' + matlabdir + ' && /usr/local/MATLAB/R2022b/bin/matlab -nodesktop -nosplash -r \'run;exit\' &)')
    while 1:
        if len(glob.glob(datadir+'/*dat')) == FILENUM_MAX:
            time.sleep(10)
            break
        else:
            time.sleep(dt)
            waittime += dt
        if waittime > WAIT_MAX:
            os.system('pgrep MATLAB | xargs kill -9')
            break


def MainLoop(N, R, parallel, ParameterFilePath, start, end):
    # 0:Nutrient,  N-1:Transporter
    StabilityResult = ''

    Variables = [i for i in range(N)]
    ParamDist,  dx_param = ImportModelParameters(ParameterFilePath)

    for loop in range(start, end):

        random.seed(loop)
        
        matlabdir,  datadir = MakeDirectories(N, R, loop)
        
        Reactions,  Catalysts = ConstructNetwork(Variables, N, R, loop)

        ParameterNames,  Parameters = ParameterAssignment(matlabdir, R, ParamDist, dx_param)

        SetupODEs(matlabdir, Reactions, Catalysts, N, R)

        SetupJacobiMatrix(matlabdir, Reactions, Catalysts, ParameterNames, N, R)

        SetupRunningScript(matlabdir, N, loop, parallel=parallel)
        
        GenerateRandomInitials(N, matlabdir)

        RunMatlab(20*60, datadir, matlabdir, 32)

        Attractor,  Population,  MonoStable = ComputeAttractor(N, datadir)
        with open(f'NetworkN{N}R{R}/AnalyzeResult/Population_Attractor{loop}.txt', 'w') as fp:
            fp.write(', '.join([str(y) for y in Population]))
        with open(f'NetworkN{N}R{R}/AnalyzeResult/Attractor{loop}.txt', 'w') as fp:
            fp.write(', '.join([str(y) for y in Attractor]))
        
        if not MonoStable:
            continue
        
        StabilityResult += f'{loop} {int(MonoStable)}\n'
        
        GeneratePerturbedInitials(N, Attractor, matlabdir, Strength=0.4)
        
        RunMatlab(40*60, datadir, matlabdir, 128)

    with open(f'NetworkN{N}R{R}/Stability.txt', 'w') as fp:
        fp.write(StabilityResult)



