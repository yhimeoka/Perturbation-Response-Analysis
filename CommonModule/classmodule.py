import sympy, re, os, copy, random, joblib, itertools, pulp, copy
import numpy as np
import pandas as pd
import networkx as nx
from matplotlib import pyplot as plt 
from scipy.linalg import null_space
from pulp import PULP_CBC_CMD
import gurobipy as gp
from gurobipy import GRB

############################################################
# Structure of Model Object
# Model Class
#   |- Metabolite Class
#       |- Variable Class
#   |- Enzyme Class
#       |- Variable Class
#   |- Reaction Class
#       |- Variable Class
############################################################

#Variable Class : symbolic variable (SymPy) and concentration of chemicals (metabolites or enzymes)
class Variable:
    def __init__(self,prefix,name,val):
        self.symbol = sympy.Symbol(prefix+name) #SymPy variable
        self.val = val                          #Concentration

#Metabolite Class 
class MetaboliteClass:
    def __init__(self,idx,name,conc,S,Reactions,asc_rxn=[]):
        self.id = idx                           #Index in MetNameList
        self.name = name                        #Name of the metabolite
        self.x = Variable('x_',name,conc)       #Variable Class (Sympy and Conc)
        self.attribute = 'enzyme'               #Attribute (metabolite or enzyme)
        if name[-2:] == '_e' or name[-2:] == '_c':
            self.attribute = 'metabolite'    
        if asc_rxn == []:
            self.asc_rxn = [x for i,x in enumerate(Reactions) if abs(S[i]) > 0.1]
        else:
            self.asc_rxn = asc_rxn[:]           #List of the associated reactions 
        self.expr = sympy.sympify('x_'+name)    #Sympy variable for substitution. This is modified when the matlab script is written
        self.org_expr = sympy.sympify('x_'+name)
        
class ReactionClass:
    def __init__(self,idx,name,rate,S,Chemicals,subs=[],prod=[]):
        self.id = idx                           #Index of the reaction in RxnNameList
        self.name = name                        #Name of the reaction
        self.v = Variable('v_',name,rate)       #Sympy variable for the rate constant
        if subs == []:                          #List of the substrate and product
            self.subs = [x for i, x in enumerate(Chemicals) if S[i] < -0.1]
        else: 
            self.subs = subs[:]
        if prod == []:           
            self.prod = [x for i, x in enumerate(Chemicals) if S[i] > 0.1]
        else:
            self.prod = prod[:]
        
#Enzymeclass is the collection of the enzyme-metabolite complexes for a single enzyme
#Representative chemical is the naked enzyme forming no complex 
class EnzymeClass:
    def __init__(self,idx,name,Chemicals,Concentrations):
        s = len(name)
        exclude = []
        if name == 'FBP':
            exclude = ['FBP_Glpx'] #self.cplxを作る時に下の判定方法だとFBP_GlpxもFBPとしてカウントされてしまうので除外。PGLも同様
        if name == 'PGL':
            exclude = ['PGL_spon']
        self.id = idx                           #Index of the enzyme in EnzNameList        
        self.name = name                        #Name of the naked enzyme
        self.cplx = [name] + [x for x in Chemicals if name + '_' == x[:s+1] and any([y in x for y in ['complex','comp_','uncomp_']]) and all([y not in x for y in exclude])]                           #List of all possible complex form of the representative naked enzyme
        self.tot = Variable('Etot_',name,sum([Concentrations[x] for x in self.cplx])) #Total Concentration of Enzyme (Normally, unity)
        
class ModelObject:

    def __init__(self,Chemicals,Concentrations,Reactions,Rates,S):
        self.additionloop = 0                   #Counter for the reaction addition loop
        self.DataDir = ''                       #Data directory
        self.ScriptDir = ''                     #Script Directory   
        self.InfoDir = ''                       #Model info directory
        self.metabolites = {m:MetaboliteClass(i,m,Concentrations[m],S[i,:],Reactions) for i,m in enumerate(Chemicals)} #Metabolite Class
        self.reactions = {r:ReactionClass(i,r,Rates[r],S[:,i],Chemicals) for i,r in enumerate(Reactions)} #Reaction Class
        self.stoich = np.copy(S)                #Stoichiometry
        self.MetNameList = Chemicals[:]         #Name list of metabolites
        self.RxnNameList = Reactions[:]         #Name list of reactions
        self.ignore_rxn = None
        self.dimcoker = None                    #dimension of cokernel
        if set(Chemicals) & set(Reactions):
            print('Intersection',set(Chemicals) & set(Reactions))
            exit()

        EnzList = []                        
        for rxn in Reactions:
            m = re.findall(r'_\d+_[fb]',rxn)
            if 'inhibit' in rxn:
                continue
            if len(m) == 0:
                print('Match 0 ',rxn)
                continue
            if len(m) > 1:
                print('Match > 1',rxn)
            x = rxn.replace(m[0],'')
            if x not in EnzList:
                EnzList.append(x)
        
        self.EnzNameList = EnzList[:]               #NameList of representative enzymes
        self.enzymes = {e:EnzymeClass(i,e,Chemicals,Concentrations) for i,e in enumerate(EnzList)} #Enzyme Class
        self.__AutoCheck()                          #Checking model structure
        self.ZeroFix = []                           #Only for computing the rate distribution
        self.Jacobi = None                          #Jacobi Matrix 
        self.eigval = None                          #eigenvalue
        self.eigvec = None                          #eiggen vector
        self.Conservation = []                      #List of conservation
        self.DistRxnScheme = None                   #Distributions of quantities related to the reaction 
        self.DistDegree = None
        self.DistRate = None
        self.DistCouple = None
    
    #Initializing the temporal variables in the model object for repeating the reaction addition
    def Initialize(self):                           
        self.dimcoker = None
        self.param_basis = None
        self.ZeroFix = []
        self.Jacobi = None
        self.eigval = None
        self.eigvec = None
        self.Conservation = []
        for m in self.MetNameList:
            self.expr = sympy.sympify('x_'+m)
       
    #Computing Distributions of the rate constants and coupling scheme
    def ComputeDistributions(self):
        if not os.path.isdir('OriginalDistribution'):
            os.mkdir('OriginalDistribution')
        
        
        ZeroFix = ['amp_c','adp_c','atp_c','nad_c','nadh_c','nadp_c','nadph_c','q8_c','q8h2_c'] 
        self.ZeroFix = ZeroFix
        Distribution_ReactionScheme = np.zeros((8,8))
        Distribution_NodeDegree = np.array([0 for i in range(len(self.MetNameList))],dtype='float64')
        RatesList_Uni, RatesList_Bi = [], []

        RxnRateGroupedbyEnz_Uni, RxnRateGroupedbyEnz_Bi = '', '' 
        CouplingScheme = list(itertools.combinations(['amp_c','adp_c','atp_c'],2)) + [['nad_c','nadh_c'],['nadp_c','nadph_c'],['q8_c','q8h2_c']]
        Distribution_Coupling = [[sorted(x),0] for x in CouplingScheme]
        COV_Uni, COV_Bi = [], []
        for ii,RxnName in enumerate(self.EnzNameList):
            tmp_uni, tmp_bi = [], []
            M = np.array([0 for i in range(len(self.MetNameList))],dtype='float64')
            for i,rxn in enumerate(self.RxnNameList):
                if RxnName + '_' in rxn and 'inhibit' not in rxn and rxn[-2:] == '_f':
                    M += np.array(self.stoich[:,i],dtype='float64')
                    if len(self.reactions[rxn].subs) > 2:
                        print('more than bi reaction ',self.reactions[rxn].name,self.reactions[rxn].subs,self.reactions[rxn].prod)
                        exit()
                    if len(self.reactions[rxn].subs) == 1:
                        tmp_uni += [self.reactions[rxn].v.val,self.reactions[rxn[:-2]+'_b'].v.val]
                    if len(self.reactions[rxn].subs) == 2:
                        tmp_bi += [self.reactions[rxn].v.val,self.reactions[rxn[:-2]+'_b'].v.val]
                    
            subs, prod = [m for i,m in enumerate(self.MetNameList) if self.metabolites[m].attribute == 'metabolite' and M[i] < -0.1 and m not in ZeroFix], [m for i,m in enumerate(self.MetNameList) if self.metabolites[m].attribute == 'metabolite' and M[i] > 0.1 and m not in ZeroFix]
            
            all_reactants = [m for i,m in enumerate(self.MetNameList) if self.metabolites[m].attribute == 'metabolite' and M[i] < -0.1] + [m for i,m in enumerate(self.MetNameList) if self.metabolites[m].attribute == 'metabolite' and M[i] > 0.1]
            CoupledWith = [int(set(x[0]) <= set(all_reactants)) for x in Distribution_Coupling]
            if sum(CoupledWith) >= 2 and RxnName != 'ADK1':
                print(RxnName + f'coupled with {sum(CoupledWith)} reactions!')
                print([m for i,m in enumerate(self.MetNameList) if self.metabolites[m].attribute == 'metabolite' and M[i] < -0.1],[m for i,m in enumerate(self.MetNameList) if self.metabolites[m].attribute == 'metabolite' and M[i] > 0.1])
                #exit()
            for i in range(len(CoupledWith)):
                Distribution_Coupling[i][1] += CoupledWith[i]/len(self.EnzNameList)


            if len(subs) > 0 and len(prod) > 0:
                reactants = sorted([len(subs),len(prod)],reverse=True)
                Distribution_ReactionScheme[reactants[0],reactants[1]] += 1
                for m in subs + prod:
                    Distribution_NodeDegree[self.MetNameList.index(m)] += 1
                RatesList_Uni += tmp_uni
                RatesList_Bi += tmp_bi

                COV_Uni.append(np.std(tmp_uni)/np.average(tmp_uni))
                COV_Bi.append(np.std(tmp_bi)/np.average(tmp_bi))

                RxnRateGroupedbyEnz_Uni += ''.join([f'{ii} {x}\n' for x in tmp_uni])
                RxnRateGroupedbyEnz_Bi += ''.join([f'{ii} {x}\n' for x in tmp_bi])

        with open("OriginalDistribution/RateUni.txt","w") as fp:
            fp.write(RxnRateGroupedbyEnz_Uni)

        with open("OriginalDistribution/RateBi.txt","w") as fp:
            fp.write(RxnRateGroupedbyEnz_Bi)

        fig = plt.figure()
        plt.title("log10 of COV Uni- reaction")
        plt.hist(COV_Uni, bins=32, ec='black')
        fig.savefig("OriginalDistribution/COV_Uni.png")
        
        fig = plt.figure()
        plt.title("log10 of COV Bi- reaction")
        plt.hist(COV_Bi, bins=32, ec='black')
        fig.savefig("OriginalDistribution/COV_Bi.png")
        
        self.DistCouple = copy.deepcopy(Distribution_Coupling)

        print(f'All Reaction Edges Before Decomposition = {len(self.EnzNameList)}')        
        print(f'Coupling Probability = {sum([x[1] for x in self.DistCouple])}')

        BIN = 32
        Vmax_Uni, Vmin_Uni = np.log10(max(RatesList_Uni)), np.log10(min(RatesList_Uni))
        RatesList_Uni = [np.log10(x) for x in RatesList_Uni]
        Delta = Vmax_Uni - Vmin_Uni
        dV_Uni = Delta/BIN
        Distribution_Rates_Uni = [0 for i in range(BIN+1)]
        for x in RatesList_Uni:
            Distribution_Rates_Uni[round((x-Vmin_Uni)/dV_Uni)] += 1

        Vmax_Bi, Vmin_Bi = np.log10(max(RatesList_Bi)), np.log10(min(RatesList_Bi))
        RatesList_Bi = [np.log10(x) for x in RatesList_Bi]
        Delta = Vmax_Bi - Vmin_Bi
        dV_Bi = Delta/BIN
        Distribution_Rates_Bi = [0 for i in range(BIN+1)]
        for x in RatesList_Bi:
            Distribution_Rates_Bi[round((x-Vmin_Bi)/dV_Bi)] += 1

        output = ''
        for i in range(8):
            for j in range(8):
                output += f'{i} {j} {Distribution_ReactionScheme[i,j]}\n'
            output += '\n'
        with open('OriginalDistribution/DistScheme.txt','w') as fp:
            fp.write(output)

        output = ''
        for i in range(len(self.MetNameList)):
            output += f'{i} {Distribution_NodeDegree[i]}\n'
        with open('OriginalDistribution/DistNDegree.txt','w') as fp:
            fp.write(output)

        output = ''
        for i in range(BIN+1):
            output += f'{i*dV_Uni+Vmin_Uni} {Distribution_Rates_Uni[i]}\n'
        with open('OriginalDistribution/DistRatesUni.txt','w') as fp:
            fp.write(output)

        output = ''
        for i in range(BIN+1):
            output += f'{i*dV_Bi+Vmin_Bi} {Distribution_Rates_Bi[i]}\n'
        with open('OriginalDistribution/DistRatesBi.txt','w') as fp:
            fp.write(output)

        for m in ZeroFix:
            Distribution_NodeDegree[self.MetNameList.index(m)] = 0

        Distribution_Rates_Uni = np.array(Distribution_Rates_Uni,dtype=float)
        Distribution_Rates_Bi = np.array(Distribution_Rates_Bi,dtype=float)

        tmp = []
        for i in range(8):
            for j in range(8):    
                if Distribution_ReactionScheme[i,j] > 0.9:
                    tmp.append([(i,j),Distribution_ReactionScheme[i,j]/np.sum(Distribution_ReactionScheme)])
        self.DistRxnScheme = copy.deepcopy(tmp)

        tmp = []
        for i in range(len(Distribution_NodeDegree)):
            if Distribution_NodeDegree[i] > 0.9:
                tmp.append([i,Distribution_NodeDegree[i]/np.sum(Distribution_NodeDegree)])
        self.DistDegree = copy.deepcopy(tmp)
        
        self.DistRate = []

        tmp = []
        for i in range(len(Distribution_Rates_Uni)):
            if Distribution_Rates_Uni[i] > 0.9:
                tmp.append([i*dV_Uni+Vmin_Uni,Distribution_Rates_Uni[i]/sum(Distribution_Rates_Uni)])
        self.DistRate.append(copy.deepcopy(tmp))
         

        tmp = []
        for i in range(len(Distribution_Rates_Bi)):
            if Distribution_Rates_Bi[i] > 0.9:
                tmp.append([i*dV_Bi+Vmin_Bi,Distribution_Rates_Bi[i]/sum(Distribution_Rates_Bi)])
        self.DistRate.append(copy.deepcopy(tmp))
    
    #Adding a single reaction. For the multiple parameter mode, set RandomParam to more than or equal to zero 
    #2023/7/4 Realistic reactionを追加できるように, PresetRxnを追加
    def AddSingleReaction(self,G,Distance,PresetRxn={},PreferredDirection=False,UniUniReaction=False,CofactorCoupling=False,RandomParam=-1):
        
        if PresetRxn == {}:
                
            #Selecting the reaction scheme
            h, r = 0, random.uniform(0,1)
            selected = False
            for n in self.DistRxnScheme:
                if h < r and h + n[1] > r:
                    Scheme = n[0]
                    selected = True
                    break
                h += n[1]
                if selected:
                    break
            
            if UniUniReaction:
                """""
                UniUniReaction = True is the case of a single reaction addition
                In this case, unreacheable metabolites are not connected because then we cannot compare the distance and responsiveness
                """""
                mets = []
                DistList = {}
                with open('Analysis/DistList.txt','r') as fp:
                    for line in fp:
                        l = line.replace('\n','').split(',')
                        DistList[(l[0],l[1])] = float(l[2])
                        mets.append(l[0])
                mets = list(set(mets))
                Reachable = {}
                for x in mets:
                    Reachable[x] = len([y for y in mets if DistList[(x,y)] <= 100])
            
            if UniUniReaction: #多数の反応を足すためにモードを拡張
                Scheme = [1,1] #1 to 1 反応のみに限定。2022/08/12 (Num. 23 ~ )
        
            #反応物質の選択
            while 1:
                subs, prod = [], []
                for n in range(Scheme[0]):
                    h, r = 0, random.uniform(0,1)
                    for elem in self.DistDegree:
                        if h < r and h + elem[1] > r:
                            subs.append(self.MetNameList[elem[0]])
                            break
                        h += elem[1]
                
                for n in range(Scheme[1]):
                    h, r = 0, random.uniform(0,1)
                    for elem in self.DistDegree:
                        if h < r and h + elem[1] > r:
                            prod.append(self.MetNameList[elem[0]])
                            break
                        h += elem[1]
                
                #距離が5以下
                #Dist_RxnCombination = self.CombDistance(subs,prod)
                #反応物質にダブりがない
                if UniUniReaction:
                    if min([Reachable[subs[0]],Reachable[prod[0]]]) > 10 and len(subs) == len(list(set(subs))) and len(prod) == len(list(set(prod))) and set(subs) & set(prod) == set():
                        break
                else:
                    if len(subs) == len(list(set(subs))) and len(prod) == len(list(set(prod))) and set(subs) & set(prod) == set():
                        break
            
            #2022/08/17~ Couplingはなし
            #2022/08/26 多数の反応を足すためにモードを拡張
            if CofactorCoupling:
                #Couplingの設定
                CouplingChems = []
                if CofactorCoupling:
                    h, r = 0, random.uniform(0,1)
                    for elem in self.DistCouple:
                        if h < r and h + elem[1] > r:
                            CouplingChems = elem[0] 
                            break
                        h += elem[1]
                
                if CouplingChems != []:
                    if bool(random.randint(0,1)):
                        CouplingChems.reverse()
                    subs.append(CouplingChems[0])
                    prod.append(CouplingChems[1])
        else:
            subs, prod = PresetRxn['subs'], PresetRxn['prod']
        
        #反応構造分解とレートの設定
        prefix = f'AddRxn{self.additionloop}_'
    
        print(prefix,subs,prod)
        #===============================================================================
        #======================Chemical Classの追加 & 反応スキームの設定====================
        #===============================================================================
        FwdRxnScheme, BwdRxnScheme = [], []
        chem = prefix[:-1]
        tmp = [chem] + subs
        
        
        for n in range(2,len(tmp)+1):
            subs_decomp, prod_decomp = [tmp[n-1],'_'.join([tmp[i] for i in range(n-1)]) + '_complex'], ['_'.join([tmp[i] for i in range(n)])+'_complex']
            
            FwdRxnScheme.append([subs_decomp,prod_decomp])
            BwdRxnScheme.append([prod_decomp,subs_decomp])
        
        subs_decomp = ['_'.join([tmp[i] for i in range(len(tmp))])+ '_complex']

        tmp = [chem] + prod

        prod_decomp = ['_'.join([tmp[i] for i in range(len(tmp))])+ '_complex']

        FwdRxnScheme.append([subs_decomp,prod_decomp])#最大Complex同士で実際の化学反応が起こる部分
        BwdRxnScheme.append([prod_decomp,subs_decomp])

        
        for n in reversed(range(2,len(tmp)+1)):
            subs_decomp, prod_decomp = [tmp[n-1],'_'.join([tmp[i] for i in range(n-1)]) + '_complex'], ['_'.join([tmp[i] for i in range(n)])+'_complex']
            
            #上と逆にしておかないと方向が狂う
            BwdRxnScheme.append([subs_decomp,prod_decomp])
            FwdRxnScheme.append([prod_decomp,subs_decomp])
            
        #prod_decomp, subs_decomp = [tmp[0],tmp[1]], [tmp[0]+'_'+tmp[1]+'_complex']
        #FwdRxnScheme.append([subs_decomp,prod_decomp])#Enzymeと最初のchemicalをくっつける反応
        #BwdRxnScheme.append([prod_decomp,subs_decomp])

        #ReactionName + complexはリストにないので"_complex"を除外. AddMetNameListを作成
        AddMetNameList = []
        for n in range(len(FwdRxnScheme)):
            subs_decomp, prod_decomp = FwdRxnScheme[n][0], FwdRxnScheme[n][1]
            if prefix + 'complex' in subs_decomp:
                subs_decomp.remove(prefix + 'complex')
                subs_decomp.append(prefix[:-1])
            if prefix + 'complex' in prod_decomp:
                prod_decomp.remove(prefix + 'complex')
                prod_decomp.append(prefix[:-1])
            FwdRxnScheme[n] = [subs_decomp,prod_decomp]
            AddMetNameList += subs_decomp + prod_decomp

            subs_decomp, prod_decomp = BwdRxnScheme[n][0], BwdRxnScheme[n][1]
            if prefix + 'complex' in subs_decomp:
                subs_decomp.remove(prefix + 'complex')
                subs_decomp.append(prefix[:-1])
            if prefix + 'complex' in prod_decomp:
                prod_decomp.remove(prefix + 'complex')
                prod_decomp.append(prefix[:-1])
            BwdRxnScheme[n] = [subs_decomp,prod_decomp]
            AddMetNameList += subs_decomp + prod_decomp
        AddMetNameList = sorted(list(set(AddMetNameList)))
        AddMetNameList = [x for x in AddMetNameList if x not in self.MetNameList]
            

        Z = 0
        Concentrations = {}
        for n in AddMetNameList:
            Concentrations[n] = random.uniform(0,1)
            Z += Concentrations[n]
        for i,n in enumerate(AddMetNameList):
            Concentrations[n] /= Z
            self.metabolites[n] = MetaboliteClass(len(self.MetNameList)+i,n,Concentrations[n],[],[],['dummy']) #accosiated reactionは全て終わったら作り直す
            
        Nm = len(AddMetNameList) #新しく追加される代謝物質数


        #===============================================================================
        #==============================Reaction Classの追加==============================
        #===============================================================================
        Nr = len(subs)+len(prod)+1 #新しく追加される反応数
        FwdRxnName, BwdRxnName = [prefix+f'{i}_f' for i in range(Nr)], [prefix+f'{i}_b' for i in range(Nr)]
        AddRxnNameList = [prefix+f'{i}_f' for i in range(Nr)] + [prefix+f'{i}_b' for i in range(Nr)]

        if RandomParam >= 0: #RandomParamにゼロ以上の引数が与えられる場合はレートを計算するときに乱数を初期化（デフォルトは-1）
            random.seed(RandomParam + sum([int(self.MetNameList.index(met)) for met in subs + prod]))

        if PreferredDirection:
            direction = random.choice([-1,1]) #1:subs->prod, -1:prod->subs

        while 1:
            FwdRateCollection, BwdRateCollection = [], []
            for n in range(Nr):
                while 1:
                    FwdRate, BwdRate = 0, 0
                    #Forward
                    Uni_or_Bi = len(FwdRxnScheme[n][0]) - 1
                    h, r = 0, random.uniform(0,1)
                    for elem in self.DistRate[Uni_or_Bi]:
                        if h < r and h + elem[1] > r:
                            FwdRate = 10.0**elem[0]
                            break
                        h += elem[1]
                    #Backward
                    Uni_or_Bi = len(BwdRxnScheme[n][0]) - 1
                    h, r = 0, random.uniform(0,1)
                    for elem in self.DistRate[Uni_or_Bi]:
                        if h < r and h + elem[1] > r:
                            BwdRate = 10.0**elem[0]
                            break
                        h += elem[1]
                    
                    if not PreferredDirection:
                        break

                    if direction*(FwdRate - BwdRate) > 0: #順方向(1)なら正、逆なら負の時break
                        break
                FwdRateCollection.append(FwdRate)
                BwdRateCollection.append(BwdRate)
                
            tmp = FwdRateCollection + BwdRateCollection
            if max(tmp)/min(tmp) < 1e+4:
                break
        for n in range(Nr):    
            self.reactions[FwdRxnName[n]] = ReactionClass(len(self.RxnNameList)+n,FwdRxnName[n],FwdRateCollection[n],[],[],FwdRxnScheme[n][0],FwdRxnScheme[n][1])
            self.reactions[BwdRxnName[n]] = ReactionClass(len(self.RxnNameList)+n,BwdRxnName[n],BwdRateCollection[n],[],[],BwdRxnScheme[n][0],BwdRxnScheme[n][1])
                        

        Nr = len(AddRxnNameList)
        #===============================================================================
        #===============================Enzyme Classの追加===============================
        #===============================================================================
        AddEnzNameList = [chem]
        self.enzymes[chem] = EnzymeClass(len(self.EnzNameList),chem,AddMetNameList,Concentrations)

        #===============================================================================
        #==========================Stoichiometric Matrixの拡大===========================
        #===============================================================================
        Mm, Mr = len(self.MetNameList), len(self.RxnNameList)
        
        self.stoich = np.hstack((self.stoich,np.zeros((Mm,Nr))))
        self.stoich = np.vstack((self.stoich,np.zeros((Nm,Nr+Mr))))
        
        self.MetNameList += AddMetNameList
        self.RxnNameList += AddRxnNameList
        self.EnzNameList += AddEnzNameList
        for rxn in AddRxnNameList:
            subs, prod = self.reactions[rxn].subs, self.reactions[rxn].prod
            for s in subs:
                self.stoich[self.getIndex(s),self.getIndex(rxn)] = -1
            for p in prod:
                self.stoich[self.getIndex(p),self.getIndex(rxn)] = 1
        
        self.Renew_AscRxn()
        
    #Update Associated Reaction List
    def Renew_AscRxn(self):
        for m in self.MetNameList:
            self.metabolites[m].asc_rxn = []
        
        for r in self.RxnNameList:
            tmp = self.reactions[r].subs + self.reactions[r].prod
            for m in tmp:
                self.metabolites[m].asc_rxn.append(r)
                
    #Remove the specified reactions and chemicals consistently from the model
    def ReduceModel(self,RemoveChemicals,RemoveReactions):
        
        N, R = np.shape(self.stoich)
        
        S = np.zeros((N-len(RemoveChemicals),R-len(RemoveReactions)))
        Nn = 0
        for i,n in enumerate(self.MetNameList):
            if n in RemoveChemicals:
                continue
            Nr = 0
            for j,r in enumerate(self.RxnNameList):
                if r not in RemoveReactions:
                    S[Nn,Nr] = self.stoich[i,j]
                    Nr += 1
            Nn += 1

        self.stoich = np.array(S)

        for m in RemoveChemicals:
            self.MetNameList.remove(m)
            self.metabolites.pop(m)
        for r in RemoveReactions:
            self.RxnNameList.remove(r)
            self.reactions.pop(r)
        
        for e in self.enzymes.values():
            rem = set(RemoveChemicals) & set(e.cplx)
            for t in rem:
                e.cplx.remove(t)
        
        for m in self.metabolites.values():
            rem = set(RemoveReactions) & set(m.asc_rxn)
            for r in rem:
                m.asc_rxn.remove(r)

        for r in self.reactions.values():
            rem = set(RemoveChemicals) & set(r.subs)
            for m in rem:
                r.subs.remove(m)
            rem = set(RemoveChemicals) & set(r.prod)
            for m in rem:
                r.prod.remove(m)

        for m in self.metabolites.values():
            m.id = self.getIndex(m.name)
        for r in self.reactions.values():
            r.id = self.getIndex(r.name)
        

        N, R = np.shape(self.stoich)
        if N != len(self.metabolites) or R != len(self.reactions):
            print('Length error')
            exit()        

    
    def __AutoCheck(self):
        print('='*4+'AutoCheck'+'='*4)
        for e in self.EnzNameList:
            if abs(1.0-self.enzymes[e].tot.val) > 0.05:
                print(e,self.enzymes[e].tot.val,[[x,self.metabolites[x].x.val] for x in self.enzymes[e].cplx])
                print()
        print('='*6+'Done'+'='*6)

    #Returning the index of the given name of reaction (in RxnNameList) of chemical (in MetNameList)
    def getIndex(self,x):
        if x in self.MetNameList:
            return self.MetNameList.index(x)
        elif x in self.RxnNameList:
            return self.RxnNameList.index(x)
        else:
            print(x,'is not in the list')
            return None

    #Computing cokernelS
    def cokerS(self):
        S = self.stoich.transpose()
        coker = null_space(S)
        dimcoker = np.shape(coker)[1]
        
        #数値の分布を計算
        tmp = [np.log10(x) for x in list(np.ravel(abs(coker) + 1e-20))]
        fig = plt.figure()
        plt.title("log10 of abs val of matrix elements plus 1e-20")
        plt.hist(tmp, bins=32, ec='black')
        fig.savefig(self.InfoDir+f"coker{self.additionloop}.png")
        
        cutoff = 1e-10
        coker = np.where(abs(coker) < cutoff, 0, coker)
        
        Concentration = np.array([self.metabolites[met].x.val for met in self.MetNameList])
        self.Conservation = [{'weight':coker[:,i],'tot':np.dot(coker[:,i],Concentration)} for i in range(dimcoker)]
        
    #Evaluate the right hand side of the equations
    def EvalRHS(self,display=True):
        S = self.stoich
        N, R = np.shape(S)
        v = np.zeros(R)

        conc = {}
        for met in self.MetNameList:
            subslist = [('x_'+met,self.metabolites[met].x.val)]
            conc[met] = float(self.metabolites[met].expr.subs(subslist))
        
        for r,rxn in enumerate(self.RxnNameList):
            v[r] = self.reactions[rxn].v.val
            for n,met in enumerate(self.MetNameList):
                if S[n,r] < 0.1:
                    subslist = [('x_'+met,self.metabolites[met].x.val)]
                    v[r] *= conc[met]**abs(S[n,r])
        RHS = S@v
        if display:
            print(f'max(dx/dt) = {max(abs(RHS))}',self.MetNameList[np.argmax(abs(RHS))])
        return S@v, v
    ####################################################################################################################
    #
    # * Generating a random initial concentration by using liner programming
    # * First, the code generate target concentration with specified size of the multiplicative noise
    # * Since the target concentration normaly violate the conservation law, linear prog. find the nearest point satisfying all the conservation law
    #
    ####################################################################################################################
    

    def RandomInitial(self,ini,Xorg,VariableList,Strength,Cofactors,Flip):
        N = len(VariableList)
        RxnList = sorted(list(set([r[:-2] for r in self.RxnNameList])))
        R = len(RxnList)
        random.seed(ini)
        
        while 1:
            target = {}
            
            for i in range(N):
                if self.MetNameList[i] in Cofactors:
                    target[i] = Xorg[i]
                else:
                    target[i] = random.uniform(1.0-Strength,1.0+Strength)*Xorg[i]
                
            model = gp.Model()
            model.params.OutputFlag = 0        #supressing output
            model.params.TimeLimit = 300        #setting timelimit to 5 mins
            if Flip > 0:
                model.params.NonConvex = 2

            #=======Add variables to model======#
            Limit = 1
            if Flip > 0:
                Limit = 10
            a, b = [], []
            for i in range(N):
                if self.MetNameList[i] in Cofactors:
                    a.append(model.addVar(lb=0,ub=0,vtype=GRB.CONTINUOUS))
                    b.append(model.addVar(lb=0,ub=0,vtype=GRB.CONTINUOUS))
                else:
                    a.append(model.addVar(lb=0,ub=Limit,vtype=GRB.CONTINUOUS))
                    b.append(model.addVar(lb=0,ub=0.95*Limit,vtype=GRB.CONTINUOUS))

            if Flip > 0:
                flipped = []
                for n in range(R):
                    flipped.append(model.addVar(vtype=GRB.BINARY))

            #================Objective function==========
            expr = gp.LinExpr()
            for i in range(N):
                expr += a[i] + b[i]
            model.setObjective(expr,GRB.MINIMIZE)

            #==========Conserved Quantity==========
            for eq in self.Conservation:
                tot = eq['tot']
                expr = gp.LinExpr()
                for i in range(N):
                    expr += eq['weight'][i]*target[i]*(1+a[i]-b[i])
                model.addLConstr(expr - tot == 0)

            BigM = 1e+4
            #==========Flipped Reactions========
            if Flip > 0:
                flip_constr = gp.LinExpr()
                for i,r in enumerate(RxnList):
                    forward, backward = r + '_f', r + '_b'
                    Jp0, Jm0 = self.reactions[forward].v.val, self.reactions[backward].v.val
                    Jp, Jm = gp.QuadExpr(), gp.QuadExpr()
                    Jp += self.reactions[forward].v.val
                    Jm += self.reactions[backward].v.val
                    for m in self.reactions[forward].subs:
                        idx = self.getIndex(m)
                        Jp0 *= self.metabolites[m].x.val
                        Jp *= target[idx]*(1+a[idx]-b[idx])
                    for m in self.reactions[backward].subs:
                        idx = self.getIndex(m)
                        Jm0 *= self.metabolites[m].x.val
                        Jm *= target[idx]*(1+a[idx]-b[idx])
                    J0 = max(1,abs(Jp0-Jm0))
                    if Jp0 > Jm0: #Originally Forward
                        model.addQConstr(Jp/J0 - Jm/J0 >= -BigM*flipped[i] + 1/BigM)
                        model.addQConstr(Jp/J0 - Jm/J0 <= BigM*(1.-flipped[i]) - 1/BigM)
                    else: #Originally Backward
                        model.addQConstr(Jp/J0 - Jm/J0 <= BigM*flipped[i] - 1/BigM)
                        model.addQConstr(Jp/J0 - Jm/J0 >= -BigM*(1.-flipped[i]) + 1/BigM)

                    flip_constr += flipped[i]
                model.addLConstr(flip_constr == Flip)

            # Solve
            model.optimize()
            if model.status == 2:
                break

        result = {}
        for i,met in enumerate(VariableList):
            result[met] = target[i]*(1+a[i].x-b[i].x)
        tmp = np.array(list(result.values()))

        if len(tmp[tmp<0]) > 0:
            print('Error',tmp)
            exit()
        return result



    def RandomInitial_Flip(self,ini,Xorg,VariableList,Strength,Flip):
        N = len(VariableList)
        RxnList = sorted(list(set([r[:-2] for r in self.RxnNameList])))
        R = len(RxnList)
        random.seed(ini)
        
        Flip = 5
        while 1:
            print(f'Flip={Flip}')
            Flip -= 1
            target = {}
            
            for i in range(N):
                #target[i] = random.uniform(1.0-Strength,1.0+Strength)*Xorg[i]
                target[i] = Xorg[i]
                
            model = gp.Model()
            model.params.OutputFlag = 1         #supressing output
            model.params.TimeLimit = 600       #setting timelimit to 5 mins
            model.params.SolutionLimit = 1
            if Flip > 0:
                model.params.NonConvex = 2

            #=======Add variables to model======#
            x = []
            for i in range(N):
                #x.append(model.addVar(lb=1e-4,ub=1e4,vtype=GRB.CONTINUOUS))
                x.append(model.addVar(lb=0,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS))
            
            if Flip > 0:
                flipped = []
                for n in range(R):
                    flipped.append(model.addVar(vtype=GRB.BINARY))

            #================Objective function==========
            expr = gp.QuadExpr()
            for i in range(N):
                expr += (1.0 - x[i])**2
            model.setObjective(expr,GRB.MINIMIZE)

            #=============Non-Negative==========
            for eq in self.Conservation:
                tot = eq['tot']
                expr = gp.LinExpr()
                for i in range(N):
                    expr += eq['weight'][i]*x[i]*target[i]
                model.addLConstr(expr - tot == 0)

            BigM = 1e+4
            #==========Flipped Reactions========
            if Flip > 0:
                flip_constr = gp.LinExpr()
                for i,r in enumerate(RxnList):
                    forward, backward = r + '_f', r + '_b'
                    
                    if 'inhibit' in r or min(self.reactions[forward].v.val, self.reactions[backward].v.val) < 1e-8:
                        continue
                    Jp0, Jm0 = self.reactions[forward].v.val, self.reactions[backward].v.val
                    Jp, Jm = gp.QuadExpr(), gp.QuadExpr()
                    Jp += self.reactions[forward].v.val
                    Jm += self.reactions[backward].v.val
                    for m in self.reactions[forward].subs:
                        idx = self.getIndex(m)
                        Jp0 *= Xorg[i]
                        Jp *= x[idx]*target[i]
                    for m in self.reactions[backward].subs:
                        idx = self.getIndex(m)
                        Jm0 *= Xorg[i]
                        Jm *= x[idx]*target[i]
                    J0 = max(1,abs(Jp0-Jm0))
                    J0 = 1
                    if abs(Jp0-Jm0) < 1e-5:
                        if Jp0 > Jm0:
                            model.addQConstr(Jp - Jm >= 1e-6)
                        else:
                            model.addQConstr(Jp - Jm <= -1e-6)
                        continue
                    
                    if Jp0 > Jm0: #Originally Forward
                        model.addQConstr(Jp/J0 - Jm/J0 >= -BigM*flipped[i] + 1/BigM)
                        model.addQConstr(Jp/J0 - Jm/J0 <= BigM*(1.-flipped[i]) - 1/BigM)
                    else: #Originally Backward
                        model.addQConstr(Jp/J0 - Jm/J0 <= BigM*flipped[i] - 1/BigM)
                        model.addQConstr(Jp/J0 - Jm/J0 >= -BigM*(1.-flipped[i]) + 1/BigM)

                    flip_constr += flipped[i]
                model.addLConstr(flip_constr == Flip)

            # Solve
            model.optimize()
            if model.status in [2,9]:
                break
            print('infeasible')
            print()

        result = {}
        for i,met in enumerate(VariableList):
            result[met] = x[idx].x
        tmp = np.array(list(result.values()))

        if len(tmp[tmp<0]) > 0:
            print('Error',tmp)
            exit()
        return result


    '''This is pulp version
    def RandomInitial(self,ini,Xorg,VariableList,Strength,Flip):
        N = len(VariableList)
        
        while 1:
            target = {}
            
            for i in range(N):
                target[i] = random.uniform(1.0-Strength,1.0+Strength)*Xorg[i]
            
            problem = pulp.LpProblem('Initial_Point_Generation', pulp.LpMinimize) 

            #=======Add variables to model======#
            a, b = [pulp.LpVariable(f'a{i}', 0, 1, 'Continuous') for i in range(N)], [pulp.LpVariable(f'b{i}', 0, 0.95, 'Continuous') for i in range(N)]
            
            #================Objective function==========
            problem += pulp.lpSum( a[i] + b[i] for i in range(N) )
            
            #=============Non-Negative==========
            for eq in self.Conservation:
                problem += pulp.lpSum(eq['weight'][i]*target[i]*(1+a[i]-b[i]) for i in range(N) ) == eq['tot']

            problem.solve(PULP_CBC_CMD(msg=0))    
            if pulp.LpStatus[problem.status] == 'Optimal':
                break

        result = {}
        ax, bx = [pulp.value(a[i]) for i in range(N)], [pulp.value(b[i]) for i in range(N)]
        for i,met in enumerate(VariableList):
            result[met] = target[i]*(1+ax[i]-bx[i])
        tmp = np.array(list(result.values()))

        if len(tmp[tmp<0]) > 0:
            print('Error',tmp)
            exit()
        return result
    '''

    # This function runs "RandomInitial" function above in parallel manner and output the text file of the initial concentrations
    def GenerateRandomInitial(self,INIT_MAX,Strength,Cofactors=[],Flip=-1):
        
        #Cofactors = ['atp_c','adp_c','amp_c','nadh_c','nad_c','nadph_c','nadp_c']

        N = len(self.MetNameList)
        X = np.zeros((N,INIT_MAX))

        Xorg = np.array([self.metabolites[x].x.val for x in self.MetNameList])
        
        if Flip < 0:
            result = joblib.Parallel(n_jobs=-1)(joblib.delayed(self.RandomInitial)(ini,Xorg,self.MetNameList,Strength,Cofactors,Flip) for ini in range(INIT_MAX))
        else:
            result = joblib.Parallel(n_jobs=-1)(joblib.delayed(self.RandomInitial_Flip)(ini,Xorg,self.MetNameList,Strength,Flip) for ini in range(INIT_MAX))
        print('End')
        for ini in range(INIT_MAX):    
            for i,n in enumerate(self.MetNameList): 
                X[i,ini] = result[ini][n]
        print('generate')
        directory = self.ScriptDir
        
        np.savetxt(directory+'InitAll.txt',X)


    ####################################################################################################################
    #
    # Output the reaction network structure of the enzymatic reactions by the specified enzyme 
    # Enzymatic reaction in this model has at least a single loop.
    #
    ####################################################################################################################
    
    def OutputEnzStructure(self,e):
        if not os.path.isdir('Graph'):
            os.system('mkdir Graph')
        
        G = nx.Graph()
        G.add_nodes_from(self.enzymes[e].cplx)
        for p in self.enzymes[e].cplx:
            for q in self.enzymes[e].cplx:
                if p == q:
                    continue
                p_reactant, q_reactant = [], []
                for r in self.metabolites[p].asc_rxn:
                    p_reactant += self.reactions[r].subs + self.reactions[r].prod
                    
                for r in self.metabolites[q].asc_rxn:
                    q_reactant += self.reactions[r].subs + self.reactions[r].prod
                    
                if q in p_reactant or p in q_reactant:
                    G.add_edge(p,q)
        
        plt.figure(figsize=(15,15))
        pos = nx.spring_layout(G)
        nx.draw_networkx(G, pos, with_labels=True, node_shape='.')
        plt.axis("off")
        plt.show()
        
        #plt.savefig('Graph/'+e+'.png')


    def GenerateMatlabScript(self,INIT_MAX,Cofactors=[],ParamScript=False,ReactionScript=False,ODEScript=False,runScript=False,Jacobi=False,ShowStability=False):
        
        for m in self.metabolites.values():
            m.expr = sympy.Symbol('x_'+m.name)

        S = self.stoich
        data_dir = self.DataDir
        directory = self.ScriptDir
        
        VariableList = [x for x in self.MetNameList]
        N = len(VariableList)
        
        for m in self.metabolites.values():
            m.expr = 'x(' + str(VariableList.index(m.name)+1) + ')'
        
        if ParamScript:
            print('Parameter')
            #Parameter
            output = ';\n'.join([str(r.v.symbol)+'='+str(r.v.val) for r in self.reactions.values()]) + ';\n' 
            #output += ';\n'.join([str(m.x.symbol)+'='+str(m.x.val) for m in self.metabolites.values()]) + ';\n'
            output += 'kinetic_param = [' + ';'.join([str(self.reactions[r].v.val) for r in self.RxnNameList]) + '];\n'
            with open(directory+'Parameters.m','w') as fp:
                fp.write(output)

        if ReactionScript:
            print('Reactions')
            #Reactions
            RateForm = {}
            output = ''
            for i, r in enumerate(self.RxnNameList):
                rxn = self.reactions[r]
                output += 'V_' + r + ' = @(x) ' + str(rxn.v.symbol) + '.*' + '.*'.join([self.metabolites[m].expr+'^('+str(abs(S[self.getIndex(m),i]))+')' for m in rxn.subs]) + ';\n'
                RateForm[r] = sympy.sympify(str(rxn.v.symbol) + '*' + '*'.join([self.metabolites[m].expr+'**('+str(abs(S[self.getIndex(m),i]))+')' for m in rxn.subs]))
                
            with open(directory+'Reactions.m','w') as fp:
                fp.write(output)
            
        
        if ODEScript:
            print('ODE')
            output = ''
            for i,n in enumerate(VariableList):
                output += '%'+n+'\n'
                if n in Cofactors:
                    output += 'dx'+str(i+1)+'_dt = @(t,x) 0;\n'
                else: 
                    idx = self.getIndex(n)
                    output += 'dx'+str(i+1)+'_dt = @(t,x) ' + ' + '.join([str(S[idx,j])+'*V_'+r+'(x)' for j,r in enumerate(self.RxnNameList) if S[idx,j] > 0.1]) + ' '.join([str(S[idx,j])+'*V_'+r+'(x)' for j,r in enumerate(self.RxnNameList) if S[idx,j] < - 0.1]) + ';\n'
                    
 
            output += '\ndf_dt = @(t,x)[' + ';'.join(['dx' + str(n+1) + '_dt(t,x)' for n in range(N)]) + '];\n\n'
            
            with open(directory+'Equations.m','w') as fp:
                fp.write(output)
        
        Tmax = 1e+9
        if runScript:
            output = 'clearvars\nxini = importdata(\'InitAll.txt\');\nw=size(xini);\nINIT_MAX=w(2);\n'
            output += 'parfor init=1:INIT_MAX\nSingle(init);\nend\npoolobj = gcp(\'nocreate\');\ndelete(poolobj);\n\n'
            
            #output += 'parfor init=1:INIT_MAX\nSingle(init);\nend\n\n'
            output += 'function Single(init)\n'
            output += 'xini = importdata(\'InitAll.txt\');\ndisp(init)\nParameters;Reactions;Equations;\n'
            output += 'y0 = [' + ' '.join(['xini(' + str(n+1) + ',init)' for n in range(N)])+'];\n'
            
            output += 'fileidx = init;\n'
            output += 'tstart = tic;\n'
            output += 'options = odeset(\'RelTol\',1e-3,\'AbsTol\',1e-5,\'Jacobian\',@(t,y)jacobi(t,y,kinetic_param),\'OutputFcn\', @(t,y,flag) myOutPutFnc(t,y,flag,tstart));\n'
            output += f'[t,y] = ode15s(df_dt,[0,{Tmax}],y0,options);\n'
            output += 'w=size(y,1);\n'
            output += f'if t(w) < {0.1*Tmax}\n'
            output += 'msg = sprintf(\'terminated %d\',init);\ndisp(msg);\nif init > 4\nclear t y;\n'
            output += 'return\nend\n'
            output += 'fileidx = -fileidx;\nelse\n'

            output += 'msg = sprintf(\'sccesfully computed %d\',init);\n'
            output += 'disp(msg);\nend\n'
            
            output += 'clear tau z\n'
            output += 'w=size(y,1);\n'
            output += 'for n = 1:' + str(N) + '\nz(1,n)=y(2,n);\nend\ntau(1,1)=t(2);\n'
            output += 'count=2;\n'
            output += 'for c = 3:w\nif mod(c,2)==0\ncontinue\nend\n\n'
            output += 'dist = 0;\ntnext=1e-5;\n'
            output += 'for n = 1:' + str(N) + f'\ndist = dist + (log(y(c,n)) - log(z(count-1,n)))^2;\nend\ndist = sqrt(dist)/{N};\n'
            output += 'if dist > 10.0 | t(c) > tnext\ntau(count,1)=t(c);\ntnext=t(c)*2;\nfor n = 1:' + str(N) + '\nz(count,n) = y(c,n);\nend\ncount = count + 1;\nend\nend\n'
            output += 'for n = 1:' + str(N) + '\nz(count,n)=y(w,n);\nend\ntau(count,1)=t(w);\n'
            output += 'file=sprintf(\'../'+data_dir+'conc%d.dat\',fileidx);\ndlmwrite(file,[tau z],\'precision\', \'%e\',\'delimiter\',\' \');\nend\n'
            
            output += '\n\n'

            output += 'function states = myOutPutFnc(t,y,flag,tstart);\nTmax = 900;\nstates=0;\nswitch(flag)\ncase []\nif toc(tstart) > Tmax\nstates = 1;\nend\nend\nend\n'
            #output += 'function [values,isterminal,direction] = myevent(t,y,tstart)\nvalues(1) = t;\nvalues(2) = toc(tstart) < 900;\nisterminal = true(size(values));\ndirection = zeros(size(values));\nend\n'
            
            with open(directory+'run.m','w') as fp:
                fp.write(output)

        if Jacobi:
            #if Elimination == []:
                    
            Diff_RateForm = {}
            print('Differentiating...')
                
            def Single_Differentiate(r):
                result = []
                #反応rのnによる微分
                for n in self.reactions[r].subs:
                    expression = f'kinetic_param({self.RxnNameList.index(r)+1})'
                    tmp = {m:int(abs(S[self.getIndex(m),self.getIndex(r)])) for m in self.reactions[r].subs}
                    for m in self.reactions[r].subs:
                        if n == m:
                            if tmp[n] > 1.1:
                                expression += f'*{tmp[n]}*x({VariableList.index(n) + 1})^({tmp[n]-1})'
                        else:
                            expression += f'*x({VariableList.index(m) + 1})^({tmp[m]})'
                    result.append([(r,n),expression])
                    
                return result
            
            result = joblib.Parallel(n_jobs=-1)(joblib.delayed(Single_Differentiate)(r) for r in self.RxnNameList)
            
            for r_single in result:
                for r in r_single:
                    Diff_RateForm[r[0]] = r[1]
            
            print('Constructing Jacobian')
            
            
            def Single_Jacobi_Elem(i,n):
                    
                Jacobi = {}    
                output = ''
                output2 = ''
                if n in Cofactors:
                    return output, output2, Jacobi
                for j,m in enumerate(VariableList,start=1):        
                    if m in Cofactors:
                        continue
                    NonZero = False
                    J = ''
                    for r in self.metabolites[m].asc_rxn:
                        k = self.RxnNameList.index(r)
                        if m not in self.reactions[r].subs + self.reactions[r].prod:
                            continue
                        idx1 = self.getIndex(n)
                        if S[idx1,k] != 0 and (r,m) in Diff_RateForm.keys():
                            J += ' + (' + str(S[idx1,k])+'*' + Diff_RateForm[r,m] + ')'
                            NonZero = True
                    if NonZero:
                        output += 'J(%d,%d) = '%(i,j) + J + ';\n'
                        
                return output, output2, Jacobi
            

            result = joblib.Parallel(n_jobs=-1)(joblib.delayed(Single_Jacobi_Elem)(i,n) for i,n in enumerate(VariableList,start=1))
            print('joblib end')
            RxnLine, OriginalJacobi, Jacobi = [], [], []
            for res in result:
                RxnLine.append(res[0])
            
            output = 'function J = Jacobi(t,x,kinetic_param)\nJ=zeros(%d,%d);\n' % (N,N)
            for l in RxnLine:
                output += l
        
            output += 'end\n'
            with open(directory+'jacobi.m','w') as fp:
                fp.write(output)
            
        return VariableList

        
    # Calculate Reaction Flux from the concentration data
    def CalcFlux(self,conc):
        S = self.stoich
        N, R = np.shape(S)
        v = np.zeros(R)
        for r,rxn in enumerate(self.RxnNameList):
            v[r] = self.reactions[rxn].v.val
            for n,met in enumerate(self.MetNameList):
                if S[n,r] < 0.1:
                    v[r] *= conc[met]**abs(S[n,r])
        return v
    

    ####################################################################################################################
    #
    # This function "Minimization" and "CombDistance" together compute the distance of two chemicals on the reaction network 
    #
    ####################################################################################################################
    
    
    # cls2と3がgurobiを使えないので、この関数はNetwork Analysisのローカル関数ということにしてここでは使わない
    """
    def Minimization(self,S0,v):

        N, R = np.shape(S0)
        model = gp.Model()
        model.params.OutputFlag = 0
        model.params.TimeLimit = 300
        #=======Add variables to model======#
        
        cp, cm, sigma = [], [], []
        for i in range(R):
            #rxn = RxnNameList[i]
            cp.append(model.addVar(lb=0,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS))
            cm.append(model.addVar(lb=0,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS))
            sigma.append(model.addVar(vtype=GRB.BINARY))
            
            model.addLConstr(cp[i] + cm[i] <= sigma[i])
        #================Objective function==========
        expr = gp.LinExpr()
        for i in range(R):
            expr += sigma[i]
        model.setObjective(expr,GRB.MINIMIZE)

        #=============Difference==========
        for i in range(N):
            expr = gp.LinExpr()
            for j in range(R):
                expr += S0[i,j]*(cp[j] - cm[j])
            
            expr += -v[i]
            model.addLConstr(expr == 0)
            

        # Solve
        model.optimize()
        if model.status != 2:
            return 2*R
        obj = model.getObjective()

        return obj.getValue()

    def CombDistance(self,subs,prod):
        # Computing distance from/to any chemical below are not expected to design this function
        ZeroFix = ['amp_c','adp_c','atp_c','nad_c','nadh_c','nadp_c','nadph_c','q8_c','q8h2_c']
        S = self.stoich
        N, R = np.shape(S)

        v = np.array([0 for i in range(N)],dtype=float)
        v[self.MetNameList.index(subs[0])] = -1
        v[self.MetNameList.index(prod[0])] = 1
        
        Sreduced = []
        for RxnName in self.EnzNameList:
            if '_ex' in RxnName:
                continue
            
            M = np.array([0 for i in range(len(self.MetNameList))],dtype='float64')
            for i,rxn in enumerate(self.RxnNameList):
                if rxn[-2:] == '_b':
                    continue
                
                if RxnName + '_' in rxn and 'inhibit' not in rxn:
                    M += np.array(self.stoich[:,i],dtype='float64')
            Sreduced.append(M)
            
        Sreduced = np.array(Sreduced).T
        for m in ZeroFix:
            Sreduced[self.MetNameList.index(m),:] = 0
        
        v = np.array([0 if x in ZeroFix else v[i] for i,x in enumerate(self.MetNameList)])
        return self.Minimization(Sreduced,v)
    """
    
    #ConcRange内に目指す状態が存在するか否か
    def Overlap(self,X,C,ConcRange):
        RxnList = []
        for r in self.RxnNameList:
            rxn = r[:-2]
            if rxn not in RxnList:
                RxnList.append(rxn)
        
        #=======List up the auxility variables for the cubic objective function======#
        grb_model = gp.Model()
        grb_model.params.OutputFlag = 0
        grb_model.params.NonConvex = 2
        grb_model.params.BestBdStop = 1e-8  
        grb_model.params.NumericFocus = 1  
        
        
        #=======Add variables to grb_model======#
        v, zp, zm = [], [], []
        for i,m in enumerate(self.MetNameList):
            v.append(grb_model.addVar(lb=1/ConcRange,ub=ConcRange,vtype=GRB.CONTINUOUS,name = f'v{i}_'+m))

        for i,r in enumerate(RxnList):
            zp.append(grb_model.addVar(lb=0,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name = f'Zp{i}_'+r))
            zm.append(grb_model.addVar(lb=0,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name = f'Zm{i}_'+r))
        
        #================Directionality Constraint=====================
        for i,r in enumerate(RxnList):
            if r in self.ignore_rxn:
                continue
            Jp, Jm = gp.QuadExpr(), gp.QuadExpr()
            f = min(self.reactions[r+'_f'].v.val,self.reactions[r+'_b'].v.val)
            Jp += self.reactions[r+'_f'].v.val/f
            Jm += self.reactions[r+'_b'].v.val/f
            
            for m in self.reactions[r+'_f'].subs:
                Jp *= C[self.MetNameList.index(m)]*v[self.MetNameList.index(m)]
            for m in self.reactions[r+'_b'].subs:
                Jm *= C[self.MetNameList.index(m)]*v[self.MetNameList.index(m)]

            if X[i] == 1:
                grb_model.addQConstr(Jp - Jm  >= 0,name=f'Direction{i}')
            if X[i] == -1:            
                grb_model.addQConstr(Jp - Jm  <= 0,name=f'Direction{i}')
                    

        #================Total Enzyme Amount Constraint================
        for _,eq in enumerate(self.Conservation):
            expr = gp.LinExpr()
            for i in range(len(v)):
                expr += eq['weight'][i]*v[i]*C[i]
            grb_model.addLConstr(expr == eq['tot'],name=f'Conservation{_}')
        
            
        #=================Set Objective Function============
        for i in range(len(RxnList)):
            grb_model.setObjective(zp[i] + zm[i],GRB.MINIMIZE)

        # Solve
        grb_model.optimize()
        if grb_model.status == 15: #BestBdStop
            return False, 1, []
            
        if 2 < grb_model.status <= 6:
            #return False
            return False, 1, []

        if grb_model.status == 2:
            obj = grb_model.getObjective().getValue()
            #result.append([const_val.x,zp.x,zm.x])
            if  obj > 1e-8:
                return False, obj, []
            else:
                return True, obj, [v[i].x for i in range(len(self.MetNameList))]
        #TimeUp
        return False, 1, []


    def Overlap_MinMultiply(self,X,C):
        RxnList = []
        for r in self.RxnNameList:
            rxn = r[:-2]
            if rxn not in RxnList:
                RxnList.append(rxn)
        
        #=======List up the auxility variables for the cubic objective function======#
        grb_model = gp.Model()
        grb_model.params.OutputFlag = 1
        grb_model.params.NonConvex = 2
        grb_model.params.NumericFocus = 1  
        
        
        #=======Add variables to grb_model======#
        x, y, z, p, s  = [], [], [], [], []
        for i,m in enumerate(self.MetNameList):
            p.append(grb_model.addVar(lb=1,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name = f'm{i}_'+m))
            x.append(grb_model.addVar(lb=0,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name = f'x{i}_'+m))
            y.append(grb_model.addVar(lb=0,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name = f'y{i}_'+m))
            z.append(grb_model.addVar(lb=0,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name = f'z{i}_'+m))
            s.append(grb_model.addVar(vtype=GRB.BINARY,name = f's{i}_'+m))
            
        u = grb_model.addVar(lb=1,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name = f'ulimit')

        grb_model.setObjective(u,GRB.MINIMIZE)

        for i,m in enumerate(self.MetNameList):
            grb_model.addLConstr(y[i] == C[i]*p[i],name=f'y{i}')
            grb_model.addQConstr(z[i]*p[i] == C[i],name=f'z{i}')
            grb_model.addQConstr(y[i]*s[i] + z[i]*(1-s[i]) <= x[i],name=f'x{i} low')
            grb_model.addQConstr(y[i]*s[i] + z[i]*(1-s[i]) >= x[i],name=f'x{i} up')
            grb_model.addLConstr(p[i] <= u,name=f'Linf{i}')
            
            
        #================Directionality Constraint=====================
        for i,r in enumerate(RxnList):
            if r in self.ignore_rxn:
                continue
            Jp, Jm = gp.QuadExpr(), gp.QuadExpr()
            f = min(self.reactions[r+'_f'].v.val,self.reactions[r+'_b'].v.val)
            Jp += self.reactions[r+'_f'].v.val
            Jm += self.reactions[r+'_b'].v.val
            
            for m in self.reactions[r+'_f'].subs:
                Jp *= x[self.MetNameList.index(m)]
            for m in self.reactions[r+'_b'].subs:
                Jm *= x[self.MetNameList.index(m)]

            if X[i] == 1:
                grb_model.addQConstr(Jp - Jm  >= 0,name=f'Direction{i}')
            if X[i] == -1:            
                grb_model.addQConstr(Jp - Jm  <= 0,name=f'Direction{i}')
                    

        #================Total Enzyme Amount Constraint================
        for _,eq in enumerate(self.Conservation):
            expr = gp.LinExpr()
            for i in range(len(x)):
                expr += eq['weight'][i]*x[i]
            grb_model.addLConstr(expr == eq['tot'],name=f'Conservation{_}')
                
        # Solve
        grb_model.optimize()
        if grb_model.status == 15: #BestBdStop
            return False, -1
            
        if 2 < grb_model.status <= 6:
            #return False
            return False, -1

        if grb_model.status == 2:
            obj = grb_model.getObjective().getValue()
            #result.append([const_val.x,zp.x,zm.x])
            if  obj > 1e-8:
                return False, obj
            else:
                return True, obj
        #TimeUp
        return False, -1


    #XからYの遷移
    def Flippable_RelativeConc(self,X,Y,ConcRange):
        
        S = self.stoich
        N, R = len(self.MetNameList), len(self.RxnNameList)
        RxnList = []
        for r in self.RxnNameList:
            rxn = r[:-2]
            if rxn not in RxnList:
                RxnList.append(rxn)

        H, idx = HammingDist(X,Y)
        if H != 1:
            print('Hamming Distance between X and Y should be unity')
            exit()

        tgt_rxn = self.RxnNameList[2*idx[0]][:-2]
        
        cubic = True
        if len(self.reactions[tgt_rxn+'_f'].subs) <= 1 and len(self.reactions[tgt_rxn+'_b'].subs) <= 1:
            cubic = False


        #=======Prepare the Gradient Vector of the Hyperplane======#
        GradVector = {}
        for m in self.reactions[tgt_rxn+'_f'].subs:
            if len(self.reactions[tgt_rxn+'_f'].subs) == 1:
                GradVector[self.getIndex(m)] = {'rate':self.reactions[tgt_rxn+'_f'].v.val,'conc':None}
            else: #2体反応
                l = self.reactions[tgt_rxn+'_f'].subs[:]
                l.remove(m)
                GradVector[self.getIndex(m)] = {'rate':self.reactions[tgt_rxn+'_f'].v.val,'conc':self.getIndex(l[0])}
        
        for m in self.reactions[tgt_rxn+'_b'].subs:
            if len(self.reactions[tgt_rxn+'_b'].subs) == 1:
                GradVector[self.getIndex(m)] = {'rate':-self.reactions[tgt_rxn+'_b'].v.val,'conc':None}
            else:
                l = self.reactions[tgt_rxn+'_b'].subs[:]
                l.remove(m)
                GradVector[self.getIndex(m)] = {'rate':-self.reactions[tgt_rxn+'_b'].v.val,'conc':self.getIndex(l[0])}
        
        #=======List up the auxility variables for the cubic objective function======#
        Aux_Var = {}
        if cubic:
            count = 0
            Asc_Rxns = list(set(itertools.chain.from_iterable([self.metabolites[self.MetNameList[m]].asc_rxn for m in GradVector.keys()])))
            for rxn in Asc_Rxns:
                tmp = tuple(sorted([self.getIndex(m) for m in self.reactions[rxn].subs]))
                if len(tmp) == 2 and tmp not in Aux_Var.keys():
                    Aux_Var[tmp] = count
                    count += 1
        

        #=======List up the auxility variables for the cubic objective function======#
        grb_model = gp.Model()
        grb_model.params.OutputFlag = 0
        grb_model.params.NonConvex = 2
        #grb_model.params.TimeLimit = 600    
        grb_model.params.BestBdStop = 1e-8  
        grb_model.params.NumericFocus = 1
        
        C = [self.metabolites[m].x.val for m in self.MetNameList]
        #=======Add variables to grb_model======#
        x = []
        for i,m in enumerate(self.MetNameList):
            x.append(grb_model.addVar(lb=1/ConcRange,ub=ConcRange,vtype=GRB.CONTINUOUS,name = f'x{i}_'+m))

        if cubic:
            y = []
            for i,m in enumerate(Aux_Var.keys()):
                y.append(grb_model.addVar(lb=1/ConcRange/ConcRange,ub=ConcRange**2,vtype=GRB.CONTINUOUS,name = f'y{i}_aux_{m[0]}_{m[1]}'))
        
        zp = grb_model.addVar(lb=0,ub=1.0,vtype=GRB.CONTINUOUS,name = f'zp')
        zm = grb_model.addVar(lb=0,ub=1.0,vtype=GRB.CONTINUOUS,name = f'zm')

        #================Objective function==========
        #Stoichiometryの絶対値が0か1だということを暗に仮定して書いているので、そこは要注意
        obj = gp.QuadExpr()
        if cubic:
            for i in GradVector.keys():
                v = gp.LinExpr()
                if GradVector[i]['conc'] == None:
                    v += GradVector[i]['rate']
                else:
                    v += GradVector[i]['rate']*x[GradVector[i]['conc']]*C[GradVector[i]['conc']]
                    
                m = self.MetNameList[i]
                for r in self.metabolites[m].asc_rxn:
                    if tgt_rxn in r:
                        continue
                    
                    if len(self.reactions[r].subs) == 2:
                        tmp = tuple(sorted([self.getIndex(mm) for mm in self.reactions[r].subs]))
                        obj += S[i,self.getIndex(r)]*v*self.reactions[r].v.val*y[Aux_Var[tmp]]*C[tmp[0]]*C[tmp[1]]
                    else:
                        met = self.reactions[r].subs[0]
                        obj += S[i,self.getIndex(r)]*v*self.reactions[r].v.val*x[self.getIndex(met)]*C[self.getIndex(met)]

               
        else:
            for i in GradVector.keys():
                v = gp.LinExpr()
                if GradVector[i]['conc'] == None:
                    v += GradVector[i]['rate']
                else:
                    v += GradVector[i]['rate']*x[GradVector[i]['conc']]*C[GradVector[i]['conc']]
                    
                m = self.MetNameList[i]
                for r in self.metabolites[m].asc_rxn:
                    if tgt_rxn in r:
                        continue
                    if len(self.reactions[r].subs) == 2:
                        tmp = sorted([self.getIndex(mm) for mm in self.reactions[r].subs])
                        obj += S[i,self.getIndex(r)]*v*self.reactions[r].v.val*x[tmp[0]]*x[tmp[1]]*C[tmp[0]]*C[tmp[1]]
                        #d.append([r,m,S[i,self.getIndex(r)],self.reactions[r].v.val,self.reactions[r].subs])     
                    else:
                        met = self.reactions[r].subs[0]
                        obj += S[i,self.getIndex(r)]*v*self.reactions[r].v.val*x[self.getIndex(met)]*C[self.getIndex(met)]
                        #d.append([r,m,S[i,self.getIndex(r)],self.reactions[r].v.val,met])
        #print(d)
        #================Directionality Constraint=====================
        for i,r in enumerate(RxnList):
            if r in self.ignore_rxn:
                continue
            Jp, Jm = gp.QuadExpr(), gp.QuadExpr()
            Jp += self.reactions[r+'_f'].v.val
            Jm += self.reactions[r+'_b'].v.val
            for m in self.reactions[r+'_f'].subs:
                Jp *= x[self.getIndex(m)]*C[self.getIndex(m)]
            for m in self.reactions[r+'_b'].subs:
                Jm *= x[self.getIndex(m)]*C[self.getIndex(m)]

            if r == tgt_rxn:
                grb_model.addQConstr(Jp - Jm == 0,name=f'Direction Target')
            else:
                if X[i] == 1:
                    grb_model.addQConstr(Jp - Jm >= 0,name=f'Direction{i}')
                else:            
                    grb_model.addQConstr(Jp - Jm <= 0,name=f'Direction{i}')
                    

        #================Auxility Variables Constraint==================
        for i,m in enumerate(Aux_Var.keys()):
            grb_model.addQConstr(y[i] - x[m[0]]*x[m[1]] == 0,name=f'Auxility{i}')

        #================Total Ennzyme Amount Constraint================
        for _,eq in enumerate(self.Conservation):
            expr = gp.LinExpr()
            for i in range(len(x)):
                #if abs(eq['weight'][i]) > 1e-8:
                expr += eq['weight'][i]*x[i]*C[i]
            grb_model.addLConstr(expr == eq['tot'],name=f'Conservation{_}')
        
            
        #=================Set Objective Function============
        grb_model.setObjective(zp + zm,GRB.MINIMIZE)

        #===============Solve The Problem==============
        if X[RxnList.index(tgt_rxn)] == 1:
            grb_model.addQConstr(obj <= 0,name='Grad')
        else:
            grb_model.addQConstr(obj >= 0,name='Grad')
        # Solve
        grb_model.optimize()
        if grb_model.status == 15: #BestBdStop
            return False
            
        if 2 < grb_model.status <= 6:
            #return False
            return False

        if grb_model.status == 2:
            return True
        #TimeUp
        return False
    
    

    #XからYの遷移
    def Reachable(self,X,Y,Target,C):
        cofactors = ['atp_c','adp_c','amp_c','nad_c','nadh_c','nadp_c','nadph_c']
        S = self.stoich
        N, R = len(self.MetNameList), len(self.RxnNameList)
        RxnList = []
        for r in self.RxnNameList:
            rxn = r[:-2]
            if rxn not in RxnList:
                RxnList.append(rxn)
        

        #=======Prepare the Gradient Vector of the Hyperplane======#
        All_GradVector = {}
        All_tgt_rxn = {}
        for t in Target:
            tgt_rxn = RxnList[t]
            GradVector = {}
            for m in self.reactions[tgt_rxn+'_f'].subs:
                if len(self.reactions[tgt_rxn+'_f'].subs) == 1:
                    GradVector[self.getIndex(m)] = {'rate':self.reactions[tgt_rxn+'_f'].v.val,'conc':None}
                else: #2体反応
                    l = self.reactions[tgt_rxn+'_f'].subs[:]
                    l.remove(m)
                    GradVector[self.getIndex(m)] = {'rate':self.reactions[tgt_rxn+'_f'].v.val,'conc':self.getIndex(l[0])}
            
            for m in self.reactions[tgt_rxn+'_b'].subs:
                if len(self.reactions[tgt_rxn+'_b'].subs) == 1:
                    GradVector[self.getIndex(m)] = {'rate':-self.reactions[tgt_rxn+'_b'].v.val,'conc':None}
                else:
                    l = self.reactions[tgt_rxn+'_b'].subs[:]
                    l.remove(m)
                    GradVector[self.getIndex(m)] = {'rate':-self.reactions[tgt_rxn+'_b'].v.val,'conc':self.getIndex(l[0])}
            All_GradVector[t] = copy.deepcopy(GradVector)
            All_tgt_rxn[t] = tgt_rxn 
        #=======List up the auxility variables for the cubic objective function======#
        Aux_Var = {}
        count = 0
        Asc_Rxns = []
        for GradVector in All_GradVector.values():
            Asc_Rxns += list(set(itertools.chain.from_iterable([self.metabolites[self.MetNameList[m]].asc_rxn for m in GradVector.keys()])))
        Asc_Rxns = sorted(list(set(Asc_Rxns)))

        for rxn in Asc_Rxns:
            tmp = tuple(sorted([self.getIndex(m) for m in self.reactions[rxn].subs]))
            if len(tmp) == 2 and tmp not in Aux_Var.keys():
                Aux_Var[tmp] = count
                count += 1
    
        #=======List up the auxility variables for the cubic objective function======#
        grb_model = gp.Model()
        grb_model.params.OutputFlag = 1
        grb_model.params.NonConvex = 2 
        grb_model.params.BestBdStop = 1e-6  
        grb_model.params.NumericFocus = 1
        
        #=======Add variables to grb_model======#
        BigM = 1e+4
        x, y, xini = [], [], []
        state, flip = [], []
        Flux = []
        for t in range(len(Target)):
            tmp = []
            for i,m in enumerate(self.MetNameList):
                tmp.append(grb_model.addVar(lb=0,ub=1e3,vtype=GRB.CONTINUOUS,name = f'x{i}_ step{t}'+m))
            x.append(tmp)

            tmp = []
            for i,m in enumerate(self.MetNameList):
                tmp.append(grb_model.addVar(lb=0,ub=1e3,vtype=GRB.CONTINUOUS,name = f'xini{i}_ step{t}'+m))
            xini.append(tmp)


            tmp = []
            for i,m in enumerate(Aux_Var.keys()):
                tmp.append(grb_model.addVar(lb=0,ub=1e6,vtype=GRB.CONTINUOUS,name = f'y{i}_aux_{m[0]}_{m[1]} step{t}'))
            y.append(tmp)

            tmp1, tmp2, tmp3, tmp4 = [], [], [], []
            for i in range(len(Target)):
                tmp1.append(grb_model.addVar(vtype=GRB.BINARY,name = f'state_({i},{t})'))
                tmp2.append(grb_model.addVar(vtype=GRB.BINARY,name = f'flip_({i},{t})'))
            state.append(tmp1)
            flip.append(tmp2)
            
            tmp = []
            for r in range(R):
                tmp.append(grb_model.addVar(lb=-GRB.INFINITY,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name = f'Flux{r} step{t}'))
            Flux.append(tmp)
        z = grb_model.addVar(lb=0,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name = 'z')
        #===========State同士は必ずハミング距離で隣り合っており、単調にターゲットにたどり着いていなければならない========
        #初期状態は指定
        expr = gp.LinExpr()
        for i,t in enumerate(Target):
            if X[t] == 1:
                grb_model.addLConstr(state[0][i] == 1)
            else:
                grb_model.addLConstr(state[0][i] == 0)
            expr += flip[0][i]
        grb_model.addLConstr(expr == 1,name=f'single flip condition, step {0}')

        for step in range(1,len(Target)):
            #第nステップにおいて、ゴールからのハミング距離は元のものからstep数分減っている
            expr = gp.LinExpr()    
            for i,t in enumerate(Target):
                if Y[t] == 1:
                    expr += 1 - state[step][i]
                else:
                    expr += state[step][i] - 0
            grb_model.addLConstr(expr == len(Target) - step,name=f'Monotonic Decrease of Hamm. dist, step {step}')

            #各ステップでフリップしていいのは1つのみであり、かつ答えと合ってないもの
            expr = gp.LinExpr()
            for i,t in enumerate(Target):
                expr += flip[step][i]
                if Y[t] == 1:
                    #合っている場合は遷移しない (合っているならstate[step][i]==1なので)
                    grb_model.addLConstr(flip[step][i] + state[step][i] <= 1,name=f'flip consistency step{step} idx{t}') 
                else:
                    #合っている場合は遷移しない (合っているならstate[step][i]==0なので)
                    grb_model.addLConstr(flip[step][i] + (1-state[step][i]) <= 1,name=f'flip consistency step{step} idx{t}') 

                #flip[step][i] == 1となるとき、またその時に鍵ってstep間のstateは異なっていてOK
                grb_model.addLConstr(state[step][i]-state[step-1][i] <= 0.5 + flip[step-1][i],name=f'flip position step{step} pos at{t}') 
                grb_model.addLConstr(state[step][i]-state[step-1][i] >= -0.5 - flip[step-1][i],name=f'flip position step{step} neg at{t}') 
            grb_model.addLConstr(expr == 1,name=f'single flip condition, step {step}')#stepごとの合計flip数は1

        #多分過剰条件だが、各箇所かならず1度だけflipするという条件
        for i in range(len(Target)):
            expr = gp.LinExpr()
            for step in range(len(Target)):
                expr += flip[step][i]
            grb_model.addLConstr(expr == 1,name=f'flip-once condition, target {i}')

        #================GradVector=================
        #Stoichiometryの絶対値が0か1だということを暗に仮定して書いているので、そこは要注意
        for step in range(len(Target)):
            for i,t in enumerate(Target):
                GradVector = All_GradVector[t]
                tgt_rxn = All_tgt_rxn[t]
            
                obj = gp.QuadExpr()
            
                for k in GradVector.keys():
                    v = gp.LinExpr()
                    if GradVector[k]['conc'] == None:
                        v += GradVector[k]['rate']
                    else:
                        v += GradVector[k]['rate']*x[step][GradVector[k]['conc']]
                        
                    m = self.MetNameList[k]
                    for r in self.metabolites[m].asc_rxn:
                        if tgt_rxn in r:
                            continue
                        
                        if len(self.reactions[r].subs) == 2:
                            tmp = tuple(sorted([self.getIndex(mm) for mm in self.reactions[r].subs]))
                            obj += S[k,self.getIndex(r)]*v*self.reactions[r].v.val*y[step][Aux_Var[tmp]]
                        else:
                            met = self.reactions[r].subs[0]
                            obj += S[k,self.getIndex(r)]*v*self.reactions[r].v.val*x[step][self.getIndex(met)]

                #flipする候補として選ばれる場合、かならずそこはXと同じ符号（不正解の符号）を持っていることに注意
                #flipするなら-1e-4以下、しないならどのような値でもOK
                grb_model.addQConstr(X[t]*obj <= -1e-4 + BigM*(1 - flip[step][i]),name=f'Grad_target{t}_at_{step}th_step')

               
        #================Directionality Constraint=====================
        #Non-Target Reactions to flip
        for step in range(len(Target)):
            for i,r in enumerate(RxnList):
                #if r in self.ignore_rxn:
                #    continue
                if X[i] == 0:
                    continue
                
                K = max(self.reactions[r+'_f'].v.val,self.reactions[r+'_b'].v.val)
                Jp, Jm = gp.QuadExpr(), gp.QuadExpr()
                Jp += self.reactions[r+'_f'].v.val
                Jm += self.reactions[r+'_b'].v.val
                for m in self.reactions[r+'_f'].subs:
                    Jp *= xini[step][self.getIndex(m)]
                for m in self.reactions[r+'_b'].subs:
                    Jm *= xini[step][self.getIndex(m)]

                if i in Target:
                    #flipするならそこはゼロ
                    if step > 0:
                        grb_model.addQConstr(Jp - Jm >= -BigM*(1-flip[step-1][Target.index(i)]),name=f'flip step{step} idx{i} min')
                        grb_model.addQConstr(Jp - Jm <=  BigM*(1-flip[step-1][Target.index(i)]),name=f'flip step{step} idx{i} max')

                    grb_model.addQConstr(Jp - Jm >= -BigM*(1-state[step][Target.index(i)]),name=f'Direction step{step} idx{i} min')
                    grb_model.addQConstr(Jp - Jm <= BigM*state[step][Target.index(i)],name=f'Direction step{step} idx{i} max')
                    
                    grb_model.addLConstr(Flux[step][i] >= -BigM*(1-state[step][Target.index(i)]),name=f'Direction Flux variable step{step} idx{i} min')
                    grb_model.addLConstr(Flux[step][i] <= BigM*state[step][Target.index(i)],name=f'Direction Flux variable step{step} idx{i} max')
                else:
                        
                    if X[i] == 1:
                        grb_model.addQConstr(Jp - Jm >= 0,name=f'Non-Target Direction{i} step{step} {X[i]}')
                        grb_model.addLConstr(Flux[step][i] >= 0,name=f'Non-Target Direction Flux variable step{step} idx{i}')
                    elif X[i] == 0:
                        grb_model.addQConstr(Jp - Jm <= 1e-3,name=f'Non-Target Direction{i} step{step} {X[i]}')
                        grb_model.addQConstr(Jp - Jm >= -1e-3,name=f'Non-Target Direction{i} step{step} {X[i]}')
                        grb_model.addLConstr(Flux[step][i] <= 1e-3,name=f'Non-Target Direction Flux variable step{step} idx{i}')
                        grb_model.addLConstr(Flux[step][i] >= -1e-3,name=f'Non-Target Direction Flux variable step{step} idx{i}')
                    else:            
                        grb_model.addQConstr(Jp - Jm <= 0,name=f'Non-Target Direction{i} step{step} {X[i]}')
                        grb_model.addLConstr(Flux[step][i] <= 0,name=f'Non-Target Direction Flux variable step{step} idx{i}')


        #================Auxility Variables Constraint==================
        for step in range(len(Target)):
            for i,m in enumerate(Aux_Var.keys()):
                grb_model.addQConstr(y[step][i] - x[step][m[0]]*x[step][m[1]] == 0,name=f'Auxility{i} step{step}')

        #==============軌道がつながっているConstraint================
        #最初のstepは初期状態
        for i in range(N):
            grb_model.addLConstr(xini[0][i] == C[i])
        
        #step>0について、初期状態はひとつ前のstepの終状態
        for step in range(1,len(Target)):
            for i in range(N):
                grb_model.addLConstr(xini[step][i] == x[step-1][i],name=f'Connectedness idx{i} step{step}')

        #================Kineticに到達可能=========================
        for step in range(len(Target)):
            for i,m in enumerate(self.MetNameList):
                #if m in cofactors:
                #    continue
                F = gp.LinExpr()
                F += xini[step][i] - x[step][i]
                for r in self.metabolites[m].asc_rxn:
                    if r[-2:] == '_f':
                        F += S[i,self.getIndex(r)]*Flux[step][RxnList.index(r[:-2])]
                grb_model.addLConstr(F == 0,name = f'Kinetic Constraint {i} step {step}')

        #=================Set Objective Function============
        grb_model.setObjective(z,GRB.MINIMIZE)

        grb_model.optimize()
        
        if grb_model.status == 15: #BestBdStop
            print('BestBdStop')
            return False
        
        if grb_model.status == 2:
            print('Optimzal Found')
            return True

        grb_model.computeIIS()
        grb_model.write('test.ilp')
    


    #XからYの遷移
    def Reachable_NoGrad(self,X,Y,Target,C):
        cofactors = ['atp_c','adp_c','amp_c','nad_c','nadh_c','nadp_c','nadph_c']
        S = self.stoich
        N, R = len(self.MetNameList), len(self.RxnNameList)
        RxnList = []
        for r in self.RxnNameList:
            rxn = r[:-2]
            if rxn not in RxnList:
                RxnList.append(rxn)
        
        #=======List up the auxility variables for the cubic objective function======#
        grb_model = gp.Model()
        grb_model.params.OutputFlag = 1
        grb_model.params.NonConvex = 2 
        grb_model.params.BestBdStop = 1e-6  
        grb_model.params.NumericFocus = 1
        
        #=======Add variables to grb_model======#
        BigM = 1e+4
        x, xini = [], []
        state, flip = [], []
        
        Flux = []
        ConcRange = 2.0
        for t in range(len(Target)):
            tmp = []
            for i,m in enumerate(self.MetNameList):
                tmp.append(grb_model.addVar(lb=C[i]/ConcRange,ub=C[i]*ConcRange,vtype=GRB.CONTINUOUS,name = f'x{i}_ step{t}'+m))
            x.append(tmp)

            tmp = []
            for i,m in enumerate(self.MetNameList):
                tmp.append(grb_model.addVar(lb=C[i]/ConcRange,ub=C[i]*ConcRange,vtype=GRB.CONTINUOUS,name = f'xini{i}_ step{t}'+m))
            xini.append(tmp)
            
            tmp1, tmp2 = [], []
            for i in range(len(Target)):
                tmp1.append(grb_model.addVar(vtype=GRB.BINARY,name = f'state_({i},{t})'))
                tmp2.append(grb_model.addVar(vtype=GRB.BINARY,name = f'flip_({i},{t})'))
                
            state.append(tmp1)
            flip.append(tmp2)

            tmp = []
            for r in range(R):
                tmp.append(grb_model.addVar(lb=-GRB.INFINITY,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name = f'Flux{r} step{t}'))
            Flux.append(tmp)
        z = grb_model.addVar(lb=0,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name = f'z')
        #===========State同士は必ずハミング距離で隣り合っており、単調にターゲットにたどり着いていなければならない========
        #初期状態は指定
        expr = gp.LinExpr()
        for i,t in enumerate(Target):
            if X[t] == 1:
                grb_model.addLConstr(state[0][i] == 1)
            else:
                grb_model.addLConstr(state[0][i] == 0)
            expr += flip[0][i]
        grb_model.addLConstr(expr == 1,name=f'single flip condition, step {0}')

        for step in range(1,len(Target)):
            #第nステップにおいて、ゴールからのハミング距離は元のものからstep数分減っている
            expr = gp.LinExpr()    
            for i,t in enumerate(Target):
                if Y[t] == 1:
                    expr += 1 - state[step][i]
                else:
                    expr += state[step][i] - 0
            grb_model.addLConstr(expr == len(Target) - step,name=f'Monotonic Decrease of Hamm. dist, step {step}')

            #各ステップでフリップしていいのは1つのみであり、かつ答えと合ってないもの
            expr = gp.LinExpr()
            for i,t in enumerate(Target):
                expr += flip[step][i]
                if Y[t] == 1:
                    #合っている場合は遷移しない (合っているならstate[step][i]==1なので)
                    grb_model.addLConstr(flip[step][i] + state[step][i] <= 1,name=f'flip consistency step{step} idx{t}') 
                else:
                    #合っている場合は遷移しない (合っているならstate[step][i]==0なので)
                    grb_model.addLConstr(flip[step][i] + (1-state[step][i]) <= 1,name=f'flip consistency step{step} idx{t}') 

                #flip[step][i] == 1となるとき、またその時に鍵ってstep間のstateは異なっていてOK
                grb_model.addLConstr(state[step][i]-state[step-1][i] <= 0.5 + flip[step-1][i],name=f'flip position step{step} pos at{t}') 
                grb_model.addLConstr(state[step][i]-state[step-1][i] >= -0.5 - flip[step-1][i],name=f'flip position step{step} neg at{t}') 
            grb_model.addLConstr(expr == 1,name=f'single flip condition, step {step}')#stepごとの合計flip数は1

        #================Directionality Constraint=====================
        #Non-Target Reactions to flip
        for step in range(len(Target)):
            for i,r in enumerate(RxnList):
                #if r in self.ignore_rxn:
                #    continue
                if X[i] == 0:
                    continue
                
                K = max(self.reactions[r+'_f'].v.val,self.reactions[r+'_b'].v.val)
                Jp, Jm = gp.QuadExpr(), gp.QuadExpr()
                Jp += self.reactions[r+'_f'].v.val
                Jm += self.reactions[r+'_b'].v.val
                for m in self.reactions[r+'_f'].subs:
                    Jp *= xini[step][self.getIndex(m)]
                for m in self.reactions[r+'_b'].subs:
                    Jm *= xini[step][self.getIndex(m)]

                if i in Target:
                    #flipするならそこはゼロ
                    if step > 0:
                        grb_model.addQConstr(Jp - Jm >= -BigM*(1-flip[step-1][Target.index(i)]),name=f'flip step{step} idx{i} min')
                        grb_model.addQConstr(Jp - Jm <=  BigM*(1-flip[step-1][Target.index(i)]),name=f'flip step{step} idx{i} max')

                    grb_model.addQConstr(Jp - Jm >= -BigM*(1-state[step][Target.index(i)]),name=f'Direction step{step} idx{i} min')
                    grb_model.addQConstr(Jp - Jm <= BigM*state[step][Target.index(i)],name=f'Direction step{step} idx{i} max')
                    
                    grb_model.addLConstr(Flux[step][i] >= -BigM*(1-state[step][Target.index(i)]),name=f'Direction Flux variable step{step} idx{i} min')
                    grb_model.addLConstr(Flux[step][i] <= BigM*state[step][Target.index(i)],name=f'Direction Flux variable step{step} idx{i} max')
                else:
                        
                    if X[i] == 1:
                        grb_model.addQConstr(Jp - Jm >= 0,name=f'Non-Target Direction{i} step{step} {X[i]}')
                        grb_model.addLConstr(Flux[step][i] >= 0,name=f'Non-Target Direction Flux variable step{step} idx{i}')
                    elif X[i] == 0:
                        grb_model.addQConstr(Jp - Jm <= 1e-3,name=f'Non-Target Direction{i} step{step} {X[i]}')
                        grb_model.addQConstr(Jp - Jm >= -1e-3,name=f'Non-Target Direction{i} step{step} {X[i]}')
                        grb_model.addLConstr(Flux[step][i] <= 1e-3,name=f'Non-Target Direction Flux variable step{step} idx{i}')
                        grb_model.addLConstr(Flux[step][i] >= -1e-3,name=f'Non-Target Direction Flux variable step{step} idx{i}')
                    else:            
                        grb_model.addQConstr(Jp - Jm <= 0,name=f'Non-Target Direction{i} step{step} {X[i]}')
                        grb_model.addLConstr(Flux[step][i] <= 0,name=f'Non-Target Direction Flux variable step{step} idx{i}')


        #==============軌道がつながっているConstraint================
        #最初のstepは初期状態
        for i in range(N):
            grb_model.addLConstr(xini[0][i] == C[i])
        
        #step>0について、初期状態はひとつ前のstepの終状態
        for step in range(1,len(Target)):
            for i in range(N):
                grb_model.addLConstr(xini[step][i] == x[step-1][i],name=f'Connectedness idx{i} step{step}')

        #================Kineticに到達可能=========================
        for step in range(len(Target)):
            for i,m in enumerate(self.MetNameList):
                #if m in cofactors:
                #    continue
                F = gp.LinExpr()
                F += xini[step][i] - x[step][i]
                for r in self.metabolites[m].asc_rxn:
                    if r[-2:] == '_f':
                        F += S[i,self.getIndex(r)]*Flux[step][RxnList.index(r[:-2])]
                grb_model.addLConstr(F == 0,name = f'Kinetic Constraint {i} step {step}')

        #=================Set Objective Function============
        grb_model.setObjective(z,GRB.MINIMIZE)

        grb_model.optimize()
        
        if grb_model.status == 15: #BestBdStop
            print('BestBdStop')
            return False
        
        if grb_model.status == 2:
            print('Optimzal Found')
            return True

        grb_model.computeIIS()
        grb_model.write('test.ilp')
    #XからYの遷移
    def Reachable_NoGrad_NonMonotonic(self,X,Y,C):
        cofactors = ['atp_c','adp_c','amp_c','nad_c','nadh_c','nadp_c','nadph_c']
        S = self.stoich
        N, R = len(self.MetNameList), len(self.RxnNameList)
        RxnList = []
        for r in self.RxnNameList:
            rxn = r[:-2]
            if rxn not in RxnList:
                RxnList.append(rxn)
        R = len(RxnList)
        
        #=======List up the auxility variables for the cubic objective function======#
        grb_model = gp.Model()
        grb_model.params.OutputFlag = 1
        grb_model.params.NonConvex = 2
        grb_model.params.NumericFocus = 1
        
        #=======Add variables to grb_model======#
        Target = [r for x,y,r in zip(X,Y,RxnList) if r not in self.ignore_rxn and x!=y and y!=0]
        MaxStep = int(np.ceil(1.5*len(Target)))
        BigM = 1e+4
        x, xini = [], []
        state, flip = [], []
        Flux = []
        FinishFlag = []
        for step in range(MaxStep):
            tmp = []
            for i,m in enumerate(self.MetNameList):
                tmp.append(grb_model.addVar(lb=0,ub=1e3,vtype=GRB.CONTINUOUS,name = f'x{i}_ step{step}'+m))
            x.append(tmp)

            tmp = []
            for i,m in enumerate(self.MetNameList):
                tmp.append(grb_model.addVar(lb=0,ub=1e3,vtype=GRB.CONTINUOUS,name = f'xini{i}_ step{step}'+m))
            xini.append(tmp)

            tmp1, tmp2 = [], []
            for i in range(R):
                tmp1.append(grb_model.addVar(vtype=GRB.BINARY,name = f'state_({i},{step})'))
                tmp2.append(grb_model.addVar(vtype=GRB.BINARY,name = f'flip_({i},{step})'))
            state.append(tmp1)
            flip.append(tmp2)
            
            tmp = []
            for r in range(R):
                tmp.append(grb_model.addVar(lb=-GRB.INFINITY,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name = f'Flux{r} step{step}'))
            Flux.append(tmp)

            FinishFlag.append(grb_model.addVar(vtype=GRB.BINARY,name = f'FinishFlag {step}'))



        #========================目的関数はFlip回数=========================
        obj = gp.LinExpr()
        for step in range(MaxStep):
            for r in range(R):
                obj += flip[step][r]*(1+step) #flipしない、というstepは最後にまとめるためにflipを重み付きの和にする 
        grb_model.setObjective(obj,GRB.MINIMIZE)

        #===========State同士は必ずハミング距離で隣り合っており、単調にターゲットにたどり着いていなければならない========
        #あってない反応についてはかならずflip
        for r in Target:
            expr = gp.LinExpr()
            for step in range(MaxStep):
                expr += flip[step][RxnList.index(r)]
            grb_model.addLConstr(expr >= 1)

        #ignore reactionについてはflipはしない
        print(f'Target Length = {len(Target)}')
        for i in range(MaxStep):
            for r in self.ignore_rxn:
                grb_model.addLConstr(flip[i][RxnList.index(r)] == 0, name = f'ignore reation{i}'+r)
        
        #初期状態は指定
        expr = gp.LinExpr()
        for i,r in enumerate(RxnList):
            if r in self.ignore_rxn:
                continue
            if X[i] == 1:
                grb_model.addLConstr(state[0][i] == 1)
            else:
                grb_model.addLConstr(state[0][i] == 0)
            expr += flip[0][i]
        grb_model.addLConstr(expr == 1,name=f'single flip condition, step {0}')

        #flip=1になっているところのみ異なっていて良い
        for step in range(1,MaxStep):
            for r in range(R):
                grb_model.addLConstr(state[step-1][r] - state[step][r] >= -BigM*flip[step-1][r],name=f'flip and state relation {step} {r} min')
                grb_model.addLConstr(state[step-1][r] - state[step][r] <= +BigM*flip[step-1][r],name=f'flip and state relation {step} {r} min')

        for step in range(1,MaxStep):
            expr = gp.LinExpr()
            consistent = gp.LinExpr()
            for i,r in enumerate(RxnList):
                expr += flip[step][i]
                if r not in self.ignore_rxn:
                    if Y[i] == 1:
                        consistent += 1 - state[step][i]
                    else:
                        consistent += state[step][i] - 0
            grb_model.addLConstr(consistent <= BigM*(1 - FinishFlag[step])) #FinishFlag=1になるなら全て合っている
            grb_model.addLConstr(expr <= 1,name=f'single flip condition, step {step}')#stepごとの合計flip数は多くて1 (flipしない場合は0も許す)

        #ちゃんと正解に到達する
        expr = gp.LinExpr()
        for step in range(MaxStep):
            expr += FinishFlag[step]
        grb_model.addLConstr(expr == 1,'Existence of Correct State')
       
        #================Directionality Constraint=====================
        #Non-Target Reactions to flip
        for step in range(MaxStep-1):
            for i,r in enumerate(RxnList):
                if r in self.ignore_rxn:
                    continue

                Jp, Jm = gp.QuadExpr(), gp.QuadExpr()
                Jp += self.reactions[r+'_f'].v.val
                Jm += self.reactions[r+'_b'].v.val
                for m in self.reactions[r+'_f'].subs:
                    Jp *= x[step][self.getIndex(m)]
                for m in self.reactions[r+'_b'].subs:
                    Jm *= x[step][self.getIndex(m)]

                
                #xはstep+1のxiniなので、step+1の向きに対応するようにする
                grb_model.addQConstr(Jp - Jm >= -BigM*(1-state[step+1][i]),name=f'Direction step{step} idx{i} min')
                grb_model.addQConstr(Jp - Jm <= BigM*state[step+1][i],name=f'Direction step{step} idx{i} max')

                #Fluxの方は現在のstateに準じたフラックス (xini + S@F = x の形なので)
                grb_model.addLConstr(Flux[step][i] >= -BigM*(1-state[step][i]),name=f'Direction Flux variable step{step} idx{i} min')
                grb_model.addLConstr(Flux[step][i] <= BigM*state[step][i],name=f'Direction Flux variable step{step} idx{i} max')

        #==============軌道がつながっているConstraint================
        #最初のstepは初期状態
        for i in range(N):
            grb_model.addLConstr(xini[0][i] == C[i])
        
        #step>0について、初期状態はひとつ前のstepの終状態
        for step in range(1,MaxStep):
            for i in range(N):
                grb_model.addLConstr(xini[step][i] == x[step-1][i],name=f'Connectedness idx{i} step{step}')

        #================Kineticに到達可能=========================
        for step in range(MaxStep):
            for i,m in enumerate(self.MetNameList):
                #if m in cofactors:
                #    continue
                F = gp.LinExpr()
                F += xini[step][i] - x[step][i]
                for r in self.metabolites[m].asc_rxn:
                    if r[-2:] == '_f':
                        F += S[i,self.getIndex(r)]*Flux[step][RxnList.index(r[:-2])]
                grb_model.addLConstr(F == 0,name = f'Kinetic Constraint {i} step {step}')

        
        grb_model.optimize()
        
        if grb_model.status == 15: #BestBdStop
            print('BestBdStop')
            return False
        
        if grb_model.status == 2:
            l = [np.argmax([flip[i][j].x for j in range(R)]) for i in range(MaxStep) if np.sum([flip[i][j].x for j in range(R)]) == 1], [np.sum([flip[i][j].x for j in range(R)]) for i in range(MaxStep)]
            print('Optimzal Found')
            return l

        grb_model.computeIIS()
        grb_model.write('test.ilp')

    '''
    def Flippable(self,X,Y):
        S = self.stoich
        N, R = len(self.MetNameList), len(self.RxnNameList)
        RxnList = []
        for r in self.RxnNameList:
            rxn = r[:-2]
            if rxn not in RxnList:
                RxnList.append(rxn)

        H, idx = HammingDist(X,Y)
        if H != 1:
            print('Hamming Distance between X and Y should be unity')
            exit()

        tgt_rxn = self.RxnNameList[2*idx[0]][:-2]
        
        cubic = True
        if len(self.reactions[tgt_rxn+'_f'].subs) <= 1 and len(self.reactions[tgt_rxn+'_b'].subs) <= 1:
            cubic = False


        #=======Prepare the Gradient Vector of the Hyperplane======#
        GradVector = {}
        for m in self.reactions[tgt_rxn+'_f'].subs:
            if len(self.reactions[tgt_rxn+'_f'].subs) == 1:
                GradVector[self.getIndex(m)] = {'rate':self.reactions[tgt_rxn+'_f'].v.val,'conc':None}
            else: #2体反応
                l = self.reactions[tgt_rxn+'_f'].subs[:]
                l.remove(m)
                GradVector[self.getIndex(m)] = {'rate':self.reactions[tgt_rxn+'_f'].v.val,'conc':self.getIndex(l[0])}
        
        for m in self.reactions[tgt_rxn+'_b'].subs:
            if len(self.reactions[tgt_rxn+'_b'].subs) == 1:
                GradVector[self.getIndex(m)] = {'rate':-self.reactions[tgt_rxn+'_b'].v.val,'conc':None}
            else:
                l = self.reactions[tgt_rxn+'_b'].subs[:]
                l.remove(m)
                GradVector[self.getIndex(m)] = {'rate':-self.reactions[tgt_rxn+'_b'].v.val,'conc':self.getIndex(l[0])}
        
        #=======List up the auxility variables for the cubic objective function======#
        Aux_Var = {}
        if cubic:
            count = 0
            Asc_Rxns = list(set(itertools.chain.from_iterable([self.metabolites[self.MetNameList[m]].asc_rxn for m in GradVector.keys()])))
            for rxn in Asc_Rxns:
                tmp = tuple(sorted([self.getIndex(m) for m in self.reactions[rxn].subs]))
                if len(tmp) == 2 and tmp not in Aux_Var.keys():
                    Aux_Var[tmp] = count
                    count += 1
        

        #=======List up the auxility variables for the cubic objective function======#
        grb_model = gp.Model()
        grb_model.params.OutputFlag = 1
        grb_model.params.NonConvex = 2
        #grb_model.params.TimeLimit = 600    
        grb_model.params.BestBdStop = 1e-8  
        grb_model.params.NumericFocus = 1
        
        C = [self.metabolites[m].x.val for m in self.MetNameList]
        #=======Add variables to grb_model======#
        x = []
        for i,m in enumerate(self.MetNameList):
            x.append(grb_model.addVar(lb=0,ub=1e3,vtype=GRB.CONTINUOUS,name = f'x{i}_'+m))

        Flux = []
        for i,r in enumerate(RxnList):
            if r in self.ignore_rxn:
                Flux.append(grb_model.addVar(lb=-GRB.INFINITY,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name = f'v{i}_'+r))
            else:
                if X[i] == 1:
                    Flux.append(grb_model.addVar(lb=0,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name = f'v{i}_'+r))
                else:
                    Flux.append(grb_model.addVar(lb=-GRB.INFINITY,ub=0,vtype=GRB.CONTINUOUS,name = f'v{i}_'+r))
        if cubic:
            y = []
            for i,m in enumerate(Aux_Var.keys()):
                y.append(grb_model.addVar(lb=0,ub=1e6,vtype=GRB.CONTINUOUS,name = f'y{i}_aux_{m[0]}_{m[1]}'))
        
        zp = grb_model.addVar(lb=0,ub=1.0,vtype=GRB.CONTINUOUS,name = f'zp')
        zm = grb_model.addVar(lb=0,ub=1.0,vtype=GRB.CONTINUOUS,name = f'zm')

        #================Objective function==========
        #Stoichiometryの絶対値が0か1だということを暗に仮定して書いているので、そこは要注意
        obj = gp.QuadExpr()
        d = []
        if cubic:
            for i in GradVector.keys():
                v = gp.LinExpr()
                if GradVector[i]['conc'] == None:
                    v += GradVector[i]['rate']
                else:
                    v += GradVector[i]['rate']*x[GradVector[i]['conc']]
                    
                m = self.MetNameList[i]
                for r in self.metabolites[m].asc_rxn:
                    if tgt_rxn in r:
                        continue
                    
                    if len(self.reactions[r].subs) == 2:
                        tmp = tuple(sorted([self.getIndex(mm) for mm in self.reactions[r].subs]))
                        obj += S[i,self.getIndex(r)]*v*self.reactions[r].v.val*y[Aux_Var[tmp]]
                    else:
                        met = self.reactions[r].subs[0]
                        obj += S[i,self.getIndex(r)]*v*self.reactions[r].v.val*x[self.getIndex(met)]

               
        else:
            d = []
            for i in GradVector.keys():
                v = gp.LinExpr()
                if GradVector[i]['conc'] == None:
                    v += GradVector[i]['rate']
                else:
                    v += GradVector[i]['rate']*x[GradVector[i]['conc']]
                    
                m = self.MetNameList[i]
                for r in self.metabolites[m].asc_rxn:
                    if tgt_rxn in r:
                        continue
                    if len(self.reactions[r].subs) == 2:
                        tmp = sorted([self.getIndex(mm) for mm in self.reactions[r].subs])
                        obj += S[i,self.getIndex(r)]*v*self.reactions[r].v.val*x[tmp[0]]*x[tmp[1]]
                        #d.append([r,m,S[i,self.getIndex(r)],self.reactions[r].v.val,self.reactions[r].subs])     
                    else:
                        met = self.reactions[r].subs[0]
                        obj += S[i,self.getIndex(r)]*v*self.reactions[r].v.val*x[self.getIndex(met)]
                        #d.append([r,m,S[i,self.getIndex(r)],self.reactions[r].v.val,met])
        #print(d)
        #================Directionality Constraint=====================
        for i,r in enumerate(RxnList):
            if r in self.ignore_rxn:
                continue
            Jp, Jm = gp.QuadExpr(), gp.QuadExpr()
            Jp += self.reactions[r+'_f'].v.val
            Jm += self.reactions[r+'_b'].v.val
            for m in self.reactions[r+'_f'].subs:
                Jp *= x[self.getIndex(m)]
            for m in self.reactions[r+'_b'].subs:
                Jm *= x[self.getIndex(m)]

            if r == tgt_rxn:
                grb_model.addQConstr(Jp - Jm == 0,name=f'Direction Target')
            else:
                if X[i] == 1:
                    grb_model.addQConstr(Jp - Jm >= 0,name=f'Direction{i}')
                else:            
                    grb_model.addQConstr(Jp - Jm <= 0,name=f'Direction{i}')
                    

        #================Auxility Variables Constraint==================
        for i,m in enumerate(Aux_Var.keys()):
            grb_model.addQConstr(y[i] - x[m[0]]*x[m[1]] == 0,name=f'Auxility{i}')

        #================Total Ennzyme Amount Constraint================
        for _,eq in enumerate(self.Conservation):
            expr = gp.LinExpr()
            for i in range(len(x)):
                expr += eq['weight'][i]*x[i]
            grb_model.addLConstr(expr == eq['tot'],name=f'Conservation{_}')
        
        #=============Stoichiometric Compatibility======================
        for i,m in enumerate(self.MetNameList):
            expr = gp.LinExpr()
            for j,r in enumerate(RxnList):
                expr += S[i,self.getIndex(r+'_f')]*Flux[j]
            grb_model.addLConstr(x[i] == C[i] + expr,name = f'Soitch_Comp_x{i}_m')
            
        #=================Set Objective Function============
        grb_model.setObjective(zp + zm,GRB.MINIMIZE)

        #===============Solve The Problem==============
        if X[RxnList.index(tgt_rxn)] == 1:
            grb_model.addQConstr(obj + zp - zm <= -1e-4,name='Grad')
        else:
            grb_model.addQConstr(obj + zp - zm >= 1e-4,name='Grad')
        # Solve
        grb_model.optimize()
        if grb_model.status == 15: #BestBdStop
            return False
            
        if 2 < grb_model.status <= 6:
            #return False
            return False

        if grb_model.status == 2:
            #result.append([const_val.x,zp.x,zm.x])
            if zp.x + zm.x > 1e-8:
                return False
            else:
                for i,m in enumerate(self.MetNameList):
                    self.metabolites[m].x.val = x[i].x
                return True
        #TimeUp
        return None
    '''

    '''

    def Flippable(self,X,Y):
        S = self.stoich
        N, R = len(self.MetNameList), len(self.RxnNameList)
        RxnList = []
        for r in self.RxnNameList:
            rxn = r[:-2]
            if rxn not in RxnList:
                RxnList.append(rxn)

        H, idx = HammingDist(X,Y)
        if H != 1:
            print('Hamming Distance between X and Y should be unity')
            exit()

        tgt_rxn = self.RxnNameList[2*idx[0]][:-2]
        

        #=======List up the auxility variables for the cubic objective function======#
        grb_model = gp.Model()
        grb_model.params.OutputFlag = 1
        grb_model.params.NonConvex = 2
        #grb_model.params.TimeLimit = 600    
        #grb_model.params.BestBdStop = 1e-8/N
        grb_model.params.NumericFocus = 1
        
        C = [self.metabolites[m].x.val for m in self.MetNameList]
        #=======Add variables to grb_model======#
        x = []
        for i,m in enumerate(self.MetNameList):
            x.append(grb_model.addVar(lb=0,ub=1e3,vtype=GRB.CONTINUOUS,name = f'x{i}_'+m))

        Flux = []
        for i,r in enumerate(RxnList):
            if r in self.ignore_rxn:
                Flux.append(grb_model.addVar(lb=-GRB.INFINITY,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name = f'v{i}_'+r))
            else:
                if X[i] == 1:
                    Flux.append(grb_model.addVar(lb=0,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name = f'v{i}_'+r))
                else:
                    Flux.append(grb_model.addVar(lb=-GRB.INFINITY,ub=0,vtype=GRB.CONTINUOUS,name = f'v{i}_'+r))
        zp = grb_model.addVar(lb=0,ub=1.0,vtype=GRB.CONTINUOUS,name = f'zp')
        zm = grb_model.addVar(lb=0,ub=1.0,vtype=GRB.CONTINUOUS,name = f'zm')

        
        #================Directionality Constraint=====================
        for i,r in enumerate(RxnList):
            if r in self.ignore_rxn:
                continue
            Jp, Jm = gp.QuadExpr(), gp.QuadExpr()
            Jp += self.reactions[r+'_f'].v.val
            Jm += self.reactions[r+'_b'].v.val
            for m in self.reactions[r+'_f'].subs:
                Jp *= x[self.getIndex(m)]
            for m in self.reactions[r+'_b'].subs:
                Jm *= x[self.getIndex(m)]

            if r == tgt_rxn:
                grb_model.addQConstr(Jp - Jm == 0,name=f'Direction Target')
            else:
                if X[i] == 1:
                    grb_model.addQConstr(Jp - Jm >= 0,name=f'Direction{i}')
                else:            
                    grb_model.addQConstr(Jp - Jm <= 0,name=f'Direction{i}')
        
        #================Total Enzyme Amount Constraint================
        for _,eq in enumerate(self.Conservation):
            expr = gp.LinExpr()
            for i in range(len(x)):
                expr += eq['weight'][i]*x[i]
            grb_model.addLConstr(expr == eq['tot'],name=f'Conservation{_}')
        
        #=============Stoichiometric Compatibility======================
        obj = gp.LinExpr()
        for i,m in enumerate(self.MetNameList):
            expr = gp.LinExpr()
            for r in self.metabolites[m].asc_rxn:
                if r[-2:] == '_f':
                    expr += S[i,self.getIndex(r)]*Flux[RxnList.index(r[:-2])]
            grb_model.addLConstr(x[i] == C[i] + expr,name = f'Soitch_Comp_x{i}_m')
            
        #=================Set Objective Function============
        
        grb_model.setObjective(zp+zm,GRB.MINIMIZE)

        # Solve
        grb_model.optimize()
        if grb_model.status == 15: #BestBdStop
            return False
            
        if 2 < grb_model.status <= 6:
            #return False
            return False

        if grb_model.status == 2:
            for i,m in enumerate(self.MetNameList):
                self.metabolites[m].x.val = x[i].x
            return True
        #TimeUp
        return None
    '''
    

    def KineticSubspace(self,X,Vatt):
        S = self.stoich
        N, R = len(self.MetNameList), len(self.RxnNameList)
        RxnList = []
        for r in self.RxnNameList:
            rxn = r[:-2]
            if rxn not in RxnList:
                RxnList.append(rxn)

        #=======List up the auxility variables for the cubic objective function======#
        grb_model = gp.Model()
        grb_model.params.OutputFlag = 0
        grb_model.params.NonConvex = 2
        grb_model.params.NumericFocus = 1
        grb_model.params.MIPGap = 2.5e-2
        
        C = [self.metabolites[m].x.val for m in self.MetNameList]
        #=======Add variables to grb_model======#
        x = []
        for i,m in enumerate(self.MetNameList):
            x.append(grb_model.addVar(lb=0,ub=1e3,vtype=GRB.CONTINUOUS,name = f'x{i}_'+m))

        Flux = []
        for i,r in enumerate(RxnList):
            if r in self.ignore_rxn:
                Flux.append(grb_model.addVar(lb=0,ub=1e2,vtype=GRB.CONTINUOUS,name = f'v{i}_'+r))
            else:
                if X[i] == 1:
                    Flux.append(grb_model.addVar(lb=0,ub=1e2,vtype=GRB.CONTINUOUS,name = f'v{i}_'+r))
                else:
                    Flux.append(grb_model.addVar(lb=-1e2,ub=0,vtype=GRB.CONTINUOUS,name = f'v{i}_'+r))
        
        #================Directionality Constraint=====================
        for i,r in enumerate(RxnList):
            if r in self.ignore_rxn:
                continue
            Jp, Jm = gp.QuadExpr(), gp.QuadExpr()
            Jp += self.reactions[r+'_f'].v.val
            Jm += self.reactions[r+'_b'].v.val
            for m in self.reactions[r+'_f'].subs:
                Jp *= x[self.getIndex(m)]
            for m in self.reactions[r+'_b'].subs:
                Jm *= x[self.getIndex(m)]

            if X[i] == 1:
                grb_model.addQConstr(Jp - Jm >= 0,name=f'Direction{i}')
            else:            
                grb_model.addQConstr(Jp - Jm <= 0,name=f'Direction{i}')
                    
       
        #=============Stoichiometric Compatibility======================
        obj = gp.LinExpr()
        norm = gp.QuadExpr()
        for i,m in enumerate(self.MetNameList):
            expr = gp.LinExpr()
            for r in self.metabolites[m].asc_rxn:
                if r[-2:] == '_f':
                    expr += S[i,self.getIndex(r)]*Flux[RxnList.index(r[:-2])]
            grb_model.addLConstr(x[i] == C[i] + expr,name = f'Soitch_Comp_x{i}_m')
            #norm += expr*expr
            obj += Vatt[i]*expr
            #L-infinity norm
            grb_model.addLConstr(expr >= -1,name = f'norm{i}_min')
            grb_model.addLConstr(expr <= 1,name = f'norm{i}_max')
            
        #=================Set Objective Function============
        grb_model.setObjective(obj,GRB.MAXIMIZE)

        # Solve
        grb_model.optimize()
            
        #if grb_model.status == 2:
        F = np.array([f.x for f in Flux])
        return S[:,[self.getIndex(r+'_f') for r in RxnList]]@F
        
#####################################
# Non-ClassModule 
#####################################
def ImportData(filename,timescale=1):

    rxn = pd.read_excel(filename,sheet_name='Elementary_Kinetic_Parameters',usecols=[1,2])
    chem = pd.read_excel(filename,sheet_name='Normalized_Enz._Met._Conc.',usecols=[1,2])
    stoich = pd.read_excel(filename,sheet_name='Decomposed_Stoichiometric_Mat.',usecols=range(1,1475))

    alph = ['','_A','_B']
    N, R = len(chem), len(rxn)
    col = chem.columns
    Concentrations = {}
    Chemicals = []
    for n in range(N):
        name = chem[col[0]][n].replace('-','_').replace('(e)','_ext')
        name += alph[Chemicals.count(name)]
        Chemicals.append(name)
        Concentrations[name] = float(chem[col[1]][n])

    col = rxn.columns
    Rates = {}
    Reactions = [x.replace('-','_').replace('(e)','_ext') for x in rxn[col[0]]]
    for r in range(R):
        Rates[rxn[col[0]][r].replace('-','_').replace('(e)','_ext')] = float(rxn[col[1]][r])*timescale
    
    S = stoich.to_numpy()

    #Duplication Check
    for x in Chemicals:
        if Chemicals.count(x) > 1:
            print('Duplicate',x,Chemicals.count(x))
    
    for x in Reactions:
        if Reactions.count(x) > 1:
            print('Duplicate',x,Reactions.count(x))
    
    return Chemicals,Concentrations,Reactions,Rates,S



def HammingDist(X,Y):
    l = [x!=y for x,y in zip(X,Y)]
    return sum(l), [i for i,n in enumerate(l) if n == True]

def ToBinaty(a):
    return int((a+1)/2)