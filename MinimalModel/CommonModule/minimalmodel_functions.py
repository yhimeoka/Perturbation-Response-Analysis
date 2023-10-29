
import numpy as np
import os, random, sympy, copy, glob, joblib, sys
import gurobipy as gp
from gurobipy import GRB
from tqdm import tqdm
from scipy import linalg


def IllStoichiometry(S):
    Ssum = np.sum(abs(S), axis=0)
    return np.any(Ssum < 1.9)

#Here, S0 is the stoichiometric matrix without efflux
def FluxBalanceAnalysis(S0):
    N, R = S0.shape
    T = np.zeros((N, 2))
    T[0, 0] = 1 # 0 is input
    T[N-4, 1] = -1 # N-4 is output (N-3, N-2 and N-1 are for ATP, ADP, and AMP)
    S = np.hstack((S0, T))
    N, R = S.shape
    m = gp.Model()
    m.setParam('OutputFlag',0)
    v = m.addMVar(R, lb=-10, ub=10)
    m.setObjective(v.sum(), sense=GRB.MAXIMIZE)
    for n in range(N):
        m.addConstr(S[n, :]@v == 0)
    m.addConstr(v[R-1] >= 1)
    m.addConstr(v[R-2] >= 1)
    m.optimize()

    return m.status == 2


def RandomNetwork(N, R):
    Edges = []
    for i in range(N-1):
        Edges.append([i, i+1])
    for i in range(R-(N-1)):
        while 1:
            edge = sorted(random.sample([j for j in range(N)], 2))
            if edge not in Edges:
                Edges.append(edge)
                break
    S = np.zeros((N, R))
    for r, edge in enumerate(Edges):
        s, p = edge[0], edge[1]
        S[s, r] = -1
        S[p, r] = 1

    return S


def ComputeJacobi(N, subs, prod, Rate, filename):
    R = len(Rate)
    Connected_Reactions = [[] for n in range(N)]
    for n in range(N):
        for r in range(R):
            if n in subs[r] + prod[r]:
                Connected_Reactions[n].append(r)
    
    Differentiate = {}
    for r in range(R):
        for n in range(N):
            if n in subs[r] + prod[r]:
                Differentiate[r, n] = sympy.diff(Rate[r], f'x({n+1})')

    NonZero = np.identity(N)
    J = [[0 for n in range(N)] for m in range(N)]
    #Normal Reactions
    for n in range(N):
        for m in range(N):
            common_reactions = list(set(Connected_Reactions[n]) & set(Connected_Reactions[m]))
            for r in common_reactions:
                sign = 1
                if n in subs[r]:
                    sign = -1
                J[n][m] += sign*Differentiate[r, m]
                NonZero[n, m] = 1
    
    output = f'function J = Jacobi(t,x)\nJ=zeros({N},{N});\n'
    for n in range(N):
        for m in range(N):
            if NonZero[n, m] > 0.5:
                output += f'J({n+1}, {m+1}) = ' + str(J[n][m]) + ';\n'
    output += 'end\n'
    with open('matlabscript/Jac'+filename, 'w') as fp:
        fp.write(output)


def GenerateMatlabScript(filename, N, R, DiffusiveChemicals, subs, prod, v, kp, km, u, Params, cat=[]):
    Rext = len(DiffusiveChemicals)
    At, Nut = Params[0], Params[1]
    output = ''
    Rate = []
    for r in range(R):
        if cat == []:
            output += f'V{r+1} = @(x) ' + f'{v[r]*kp[r]}*'+'*'.join([f'x({i+1})' for i in subs[r]]) + f' - {v[r]*km[r]}*' + '*'.join([f'x({i+1})' for i in prod[r]]) + ';\n'
            rateform = f'{v[r]*kp[r]}*'+'*'.join([f'x({i+1})' for i in subs[r]]) + f' - {v[r]*km[r]}*' + '*'.join([f'x({i+1})' for i in prod[r]])
        else:
            output += f'V{r+1} = @(x) ' + f'{v[r]*kp[r]}*'+'*'.join([f'x({i+1})' for i in subs[r]+cat[r]]) + f' - {v[r]*km[r]}*' + '*'.join([f'x({i+1})' for i in prod[r]+cat[r]]) + ';\n'
            rateform = f'{v[r]*kp[r]}*'+'*'.join([f'x({i+1})' for i in subs[r]+cat[r]]) + f' - {v[r]*km[r]}*' + '*'.join([f'x({i+1})' for i in prod[r]+cat[r]])
        Rate.append(sympy.sympify(rateform))
    
    NutConc = Nut
    for i, r in enumerate(DiffusiveChemicals):
        output += f'Vext{i} = @(x) {u[i]}*({NutConc} - x({r+1}));\n'
        rateform = f'{u[i]}*({NutConc} - x({r+1}))'
        Rate.append(sympy.sympify(rateform))    
        # The first diffucive chemical is the nutrient
        NutConc = Nut*0.01
    output += '\n\n\n'

    ComputeJacobi(N, subs, prod, Rate, filename)
    
    for n in range(N):
        output += f'dx{n+1}_dt = @(t,x) 0 ' + ' '.join([f'- V{r+1}(x)' for r in range(R) if n in subs[r]]) + ' '.join([f' + V{r+1}(x)' for r in range(R) if n in prod[r]]) + ' '.join([f' + Vext{r}(x)' for r in range(Rext) if n in prod[r+R]]) + ';\n'
        
    output += 'df_dt = @(t,x)['+';'.join([f'dx{i}_dt(t,x)' for i in range(1 ,N+1)]) + '];\n'
    
    with open('matlabscript/ode'+filename, 'w') as fp:
        fp.write(output)
    output = 'myCluster = parcluster(\'Processes\');\ndelete(myCluster.Jobs);\nclearvars\nparfor ini=1:64\nresult(:,:,ini)=SingleRun(ini);\nend\n'
    output += 'all_result = result(:,:,1);\nfor i=2:64\nall_result = vertcat(all_result,result(:,:,i));\nend\n'
    output += 'file=sprintf(\'../DynamicsData/Resp'+filename[:-2]+'.txt\');\ndlmwrite(file,all_result,\'precision\', \'%e\',\'delimiter\',\' \');\n\n'
    output += 'function results = SingleRun(ini)\nclearvars -except ini\n'
    output += 'ode'+filename[:-2] + f';\noptions = odeset(\'RelTol\',1e-4,\'AbsTol\',1e-6,\'Jacobian\',@(t,y)Jac'+filename[:-2]+'(t,y));\n'
    output += f'rng(ini);\nfor i=1:{N+1}\nxini(i)=10^(2.0*rand-1.0);\nend\n'
    if cat == []:
        output += f'Z =  xini({N-2}) + xini({N-1}) + xini({N});\n'
        output += f'for i={N-2}:{N}\nxini(i)={At}*xini(i)/Z;\nend\n'
    output += '[t y] = ode15s(df_dt,[0,1e+15],[' + ' '.join([f'xini({i})' for i in range(1,N+1)]) + '],options);\n'
    output += 'if ini<4\nfile=sprintf(\'../DynamicsExample/'+filename[:-2]+'_relax%d.txt\',ini);\ndlmwrite(file,[t y],\'precision\', \'%e\',\'delimiter\',\' \');\nend\n'
    output += 'w=size(y,1);\n'
    output += f'for i=1:{N}\nyatt(i)=y(w,i);\nend\n'
    # return all -1 if computation failed
    output += f'if t(w) < 1e14\nfor ptb=1:4\nresults(ptb,1)=ini;\nresults(ptb,2)=ptb;\nfor i=3:{5+2*N}\nresults(ptb,i)=-1;\nend\nend\nreturn\nend\n\n'
    output += 'dx_ori = Compute_dx(t,y);\n'
    
    # Loop for the perturbation simualtion
    output += 'for ptb = 1:4\nclear t y z tau xini Dist\n'
    output += f'for i=1:{N}\nxini(i)=yatt(i)*(0.6 + 0.8*rand);\nend\n'
    if cat == []:
        output += f'Z = xini({N-2}) + xini({N-1}) + xini({N});\n'
        output += f'for i={N-2}:{N}\nxini(i)={At}*xini(i)/Z;\nend\n'
    output += '[t y] = ode15s(df_dt,[0,1e+15],[' + ' '.join([f'xini({i})' for i in range(1,N+1)]) + '],options);\n'
    output += 'w=size(y,1);\n'
    output += 'if ini<4 & ptb==1\nfile=sprintf(\'../DynamicsExample/'+filename[:-2]+'_time%d.txt\',ini);\ndlmwrite(file,[t y],\'precision\', \'%e\',\'delimiter\',\' \');\nend\n'
    # Output index
    output += 'results(ptb,1)=ini;\nresults(ptb,2)=ptb;\n'
    # Original Attractor
    output += f'for i=1:{N}\nresults(ptb,i+2)=yatt(i);\nend\n'
    # Attractor Returned
    output += f'for i=1:{N}\nresults(ptb,i+{N}+2)=y(w,i);\nend\n'

    # If there is no coupling we do not include the ATP, ADP, and AMP for the computation of respcoeff (output format is the same)
    if 'Cpl0' in filename:
        output += f'Nmax={N-3};\n'
    else:
        output += f'Nmax={N};\n'

    output += f'Resp=-1;\ndx=-1;\nif min(y) > 0 & t(w) > 1e14\nfor i=1:w\nfor j=1:Nmax\nx(j)=log(y(i,j))-log(y(w,j));\nend\nDist(i)=norm(x);\nend\nResp=max(Dist)/Dist(1);\ndx = Compute_dx(t,y);\nend\n'
    output += f'results(ptb,{2*N}+3)=dx_ori;\nresults(ptb,{2*N}+4)=dx;\nresults(ptb,{2*N}+5)=Resp;\nend\nend\n'
    output += 'function dx = Compute_dx(t,y)\nw=size(y,1);\nidx=getId(t);\ndx=norm(log(y(w,:))-log(y(idx,:)));\nend\n\n'
    output += 'function idx = getId(t)\n[val,idx]=min(abs(1e14-t(:)));\nend\n\n'

    with open('matlabscript/'+filename, 'w') as fp:
        fp.write(output)
    

def AssignArrehiusParameters(R, subs, prod, mu, beta):
    kp, km = [], []
    for r in range(R):
        dmu = sum([mu[s] for s in subs[r]]) - sum([mu[p] for p in prod[r]])
        kp.append(min(1.0, np.exp(beta*dmu)))
        km.append(min(1.0, np.exp(-beta*dmu)))
    return kp, km


def GenerateKineticModel(seed, N, R, beta=0, RateRange=[0.000220983886641477, 13528987.1929646], cofactor_coupling=True):
    
    At, Nut = 1.0, 100.0
    Params = [At, Nut]

    random.seed(seed)
    
    while 1:
        S = RandomNetwork(N, R)
        mu = []
        for n in range(N):
            mu.append(random.uniform(0, 1))
        if cofactor_coupling:
            mu += [1.0, 0.5, 0.0] # ATP, ADP, AMP (Only for Coupling)
    
        v = []
        subs, prod = [], []
        for r in range(R):
            v.append(np.exp(random.uniform(np.log(RateRange[0]), np.log(RateRange[1]))))
            tmp1, tmp2 = [], []
            for n in range(N):
                if int(S[n, r]) == -1:
                    tmp1.append(n)
                if int(S[n, r]) == 1:
                    tmp2.append(n)
            if len(tmp1) != 1 or len(tmp2) != 1:
                print(seed, tmp1, tmp2)
                print(S)
                exit()
            subs.append(tmp1)
            prod.append(tmp2)

        Sref = np.copy(S)
        subs_ref, prod_ref = copy.deepcopy(subs), copy.deepcopy(prod)
        
        # DiffusiveChemicals = [0] + random.sample([i for i in range(1,N-1)],1) + [N-1]
        DiffusiveChemicals = [0, N-1] + random.sample([i for i in range(1, N-1)], int(np.ceil(0.05*N)))
        u = [np.exp(random.uniform(np.log(RateRange[0]), np.log(RateRange[1]))) for i in range(len(DiffusiveChemicals))]
        DiffusiveChemicals = sorted(DiffusiveChemicals)
        T = np.zeros((N, len(DiffusiveChemicals)))
        for i, r in enumerate(DiffusiveChemicals):
            subs_ref.append([])
            prod_ref.append([r])
            T[r, i] = 1
        Rext = len(DiffusiveChemicals)
        
        Sref = np.hstack((Sref, T))

        if cofactor_coupling and len(linalg.null_space(Sref.T)[0]) == 0:
            break
        if not cofactor_coupling and len(linalg.null_space(Sref.T)[0]) == 0 and FluxBalanceAnalysis(S) and not IllStoichiometry(S):
            break
    
    # Coupling
    if cofactor_coupling:
        for ii in range(11):
            coupling = round(0.1*ii*R)
            if coupling == 1:
                continue
            
            while 1:
                S0 = np.copy(Sref[:, :R]) # S0 is the stoichiometric matrix without efflux
                S = np.copy(Sref) # S is the stoichiometric matrix with efflux
                subs, prod = copy.deepcopy(subs_ref), copy.deepcopy(prod_ref)
                coupled_reaction = random.sample([i for i in range(R)], coupling)
                T = np.zeros((3, R+Rext))
                for c in coupled_reaction:
                    l = random.sample([N, N+1, N+2], 2)
                    subs[c].append(l[0])
                    prod[c].append(l[1])
                    T[l[0]-N, c] = 1
                    T[l[1]-N, c] = -1
                T0 = np.copy(T[:, :R])
                
                S = np.vstack((S, T))
                S0 = np.vstack((S0, T0))

                if FluxBalanceAnalysis(S0) and not IllStoichiometry(S0): # Check if the network is feasible and all metabolite have at least two reactions
                    np.savetxt(f'Stoichiometry/Nut{N}Rxn{R}Cpl{ii}Net{seed}.txt', S)
                    kp, km = AssignArrehiusParameters(R, subs, prod, mu, beta)
                    GenerateMatlabScript(f'Nut{N}Rxn{R}Cpl{ii}Net{seed}.m', len(mu), len(v), DiffusiveChemicals, subs,prod, v, kp, km, u, Params)
                    break
    else:

        S = np.copy(Sref)
        subs, prod = copy.deepcopy(subs_ref), copy.deepcopy(prod_ref)
        cat = []
        for r in range(R):
            tmp = [i for i in range(N) if i not in subs[r]+prod[r]]
            c = random.sample(tmp, 1)
            cat.append(c)
        np.savetxt(f'Stoichiometry/Nut{N}Rxn{R}Cpl0Net{seed}.txt', S)
        kp, km = AssignArrehiusParameters(R, subs, prod, mu, beta)
        GenerateMatlabScript(f'Nut{N}Rxn{R}Cpl0Net{seed}.m', len(mu), len(v), DiffusiveChemicals, subs, prod, v, kp, km, u, Params, cat=cat)
        

         

