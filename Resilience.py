#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Touki @DUT

2021/06/24

"""
# %% Import Class
from Wave import Wave
from Structure import MDoFNolinear as modelNL
from Solver import MDoFNewmBNL as simulator
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import datetime
import random
from scipy import linalg
import scipy.integrate as integrate
# %% def functions
def count(ngm,EDP,Thre):
    PDS=np.zeros(5)
    for i in range (ngm):
        if EDP[i]<=Thre[0]: # None
            PDS[0]+=1
        if EDP[i]>Thre[0] and EDP[i]<=Thre[1]: # Slight
            PDS[1]+=1
        if EDP[i]>Thre[1] and EDP[i]<=Thre[2]: # Moderate
            PDS[2]+=1
        if EDP[i]>Thre[2] and EDP[i]<=Thre[3]: # Severe
            PDS[3]+=1
        if EDP[i]>=Thre[3]: # Complete
            PDS[4]+=1
    return PDS/ngm

def countwR(ngm,EDP,Thre,MRIDR,Rlim=0.005):
    PDS=np.zeros(5)
    for i in range (ngm):
        if MRIDR[i]>0.005:
            PDS[4]+=1
        else:
            if EDP[i]<=Thre[0]: # None
                PDS[0]+=1
            if EDP[i]>Thre[0] and EDP[i]<=Thre[1]: # Slight
                PDS[1]+=1
            if EDP[i]>Thre[1] and EDP[i]<=Thre[2]: # Moderate
                PDS[2]+=1
            if EDP[i]>Thre[2] and EDP[i]<=Thre[3]: # Severe
                PDS[3]+=1
            if EDP[i]>=Thre[3]: # Complete
                PDS[4]+=1
    return PDS/ngm 
# %% Create files
# hisf='./data/history/'
# if not os.path.exists(hisf):
#     os.makedirs('./data/history/')
mldataf='./data/mldata/'
if not os.path.exists(mldataf):
    os.makedirs('./data/mldata/')
figf='./data/fig/'
if not os.path.exists(figf):
    os.makedirs('./data/fig/')
timef='./data/time/'
if not os.path.exists(timef):
    os.makedirs('./data/time/')
# %% def
Time=[]
nbldg=1000 #number of bldgs
mldata=np.zeros((nbldg,32)) #record
beta=1/6 #linear acceleration method-newmark
damp=0.05 #damping ratio
harden=0.02 #hardenning ratio
h0=3.6 # story height
Ast=400 #floor area in square meter
DL=4 #kN/m2
LL=2 #kN/m2
W0=Ast*(DL+0.5*LL) # story effective seismic weight kN
m0=W0*1000/9.8 # story seismic mass kg --no influence
aData=[0.05,0.10,0.15,0.20,0.30,0.40]
TgData=np.array([[0.20,0.25,0.35,0.45,0.65],[0.25,0.30,0.40,0.55,0.75],[0.30,0.35,0.45,0.65,0.90]])
alphamaxMCEData=[0.28,0.50,0.72,0.90,1.20,1.40]
gammaSpectr=0.9+(0.05-damp)/(0.3+6*damp)
eta1Spectr=0.02+(0.05-damp)/(4+32*damp)
if eta1Spectr<0:
    eta1Spectr=0
eta2Spectr=1+(0.05-damp)/(0.08+1.6*damp)
if eta2Spectr<0.55:
    eta1Spectr=0.55
ThreSData=np.array([[[0.0048,0.0076,0.0162,0.04],[0.0032,0.0051,0.0108,0.0267]],  # VI
                    [[0.006,0.0096,0.0203,0.05],[0.004,0.0064,0.0135,0.0333]],    # VII
                    [[0.006,0.0096,0.0203,0.05],[0.004,0.0064,0.0135,0.0333]],
                    [[0.006,0.0104,0.0235,0.06],[0.004,0.0069,0.0157,0.04]],      # VIII
                    [[0.006,0.0096,0.0203,0.05],[0.004,0.0064,0.0135,0.0333]],
                    [[0.006,0.012,0.03,0.08],[0.004,0.008,0.02,0.0533]]])         #IX
ThreNSAData=np.array([[0.2,0.4,0.8,1.6],  # VI
                    [0.2,0.4,0.8,1.6],    # VII
                    [0.2,0.4,0.8,1.6],
                    [0.25,0.5,1,2],    # VIII
                    [0.25,0.5,1,2],
                    [0.3,0.6,1.2,2.4]])      #IX
RCS=np.array([0.4,1.9,9.6,19.2])
RCD=np.array([0.7,3.3,16.4,32.9])
RCA=np.array([0.9,4.8,14.4,47.9])
CD=np.array([1,5,25,50])
BRC=1897.57*Ast
CRV=BRC
Rtot=BRC+0.5*CRV
BRT=np.array([20,90,360,480]) # days
TLC=365*50
    # %% Generate models
for bldg in range (nbldg):
    starttime=datetime.datetime.now()
    #%% generate random variables
    intensity=random.randint(1,6) #seismic design intensity
    group=random.randint(1,3) #design earthquake group
    siteClass=random.randint(1,5) #site class
    n=random.randint(2,6) #number of storys
    Ct=random.uniform(0.1,0.2) # building period coefficient
    Cy1=random.uniform(0.3,0.7)
    #%% site parameters
    a=aData[intensity-1]
    Tg=TgData[group-1,siteClass-1]
    #%% stiffness 
    mdiag=np.ones(n)
    MMat=np.diag(mdiag)
    elediag=2*np.ones(n)
    elediag[n-1]=1
    eleshift=np.ones(n-1)
    AMat=np.diag(elediag)-np.diagflat(eleshift,-1)-np.diagflat(eleshift,1)
    eigs,v=linalg.eig(AMat,MMat)
    phit=np.mat([v[:,np.argmin(eigs)]])
    phi=phit.transpose()
    IMat=np.mat(np.identity(n))
    aa=phit*AMat*phi
    bb=phit*IMat*phi
    lamda=(aa[0,0])/(bb[0,0])
    T=n*Ct #fundamental period
    k0=4*m0*np.pi**2/(T**2*lamda) #lateral stiffness N/m
    #%% strength
    alphamaxMCE=alphamaxMCEData[intensity-1]
    TgMCE=Tg+0.05
    if T<0.1:
        alpha1=((eta2Spectr-0.45)/0.1*T+0.45)*alphamaxMCE
    if T>=0.1 and T<TgMCE:
        alpha1=alphamaxMCE
    if T>=TgMCE and T<5*TgMCE:
        alpha1=(TgMCE/T)**gammaSpectr*eta2Spectr*alphamaxMCE
    if T>=5*Tg:
        alpha1=(eta2Spectr*0.2**gammaSpectr-eta1Spectr*(T-5*TgMCE))*alphamaxMCE
    FEK=alpha1*0.85*n*W0 # base shear kN
    deltaN=0
    if T>1.4*TgMCE:
        if TgMCE<=0.35:
            deltaN=0.08*T+0.07
        if TgMCE>0.35 and TgMCE<=0.55:
            deltaN=0.08*T+0.01
        if TgMCE>0.55:
            deltaN=0.08*T-0.02
    deltaFN=deltaN*FEK
    F=np.ones(n) # seismic force induced at level i kN
    for i in range (n):
        H1=h0
        Hi=(i+1)*h0
        Hn=n*h0
        F[i]=Hi/((H1+Hn)*n/2)*FEK*(1-deltaN)
    F[n-1]+=deltaFN
    Fflip=np.flip(F)
    Vflip=np.cumsum(Fflip)
    V=np.flip(Vflip) # seismic design story shear in story i kN
    Cy=0.7*Cy1*np.ones(n)
    Cy[0]=Cy1
    Vy=np.multiply(Cy,V)
    #%% create structural model
    m=np.ones(n)*m0
    k=np.ones(n)*k0
    h=np.ones(n)*damp
    alf=np.ones(n)*harden
    ay=Vy/W0 # kN/kN
    mdof=modelNL(m,k,h,alf,ay)
    print("Bldg %d : DBE=%1.2f g, Tg=%1.2f s, n=%d,  Ct=%1.2f, T1=%1.2f s, k=%6.2f kN/m, Sa(T1)=%1.2f g, Cy1=%6.2f, FEK=%6.2f kN, Vy1=%6.2f kN" % (bldg+1,a,Tg,n,Ct,T,k0/1000,alpha1,Cy1,FEK,Vy[0]))
    # %% input wavedata               
    gm=0
    ngm=100
    IDR=np.zeros((ngm,n)) # 2d array: ngm*n
    ACC=np.zeros((ngm,n))
    RIDR=np.zeros((ngm,n))
    MIDR=np.zeros(ngm)
    PFA=np.zeros(ngm)
    MRIDR=np.zeros(ngm)
    WName=[]
    for root, dirs, files in os.walk ('./Wave100/'):   # unit:m/s/s
        for name in files:
            if name.endswith(".txt"):
#                print('gm=%d'%(gm+1))
                wn=name
                wnn=wn.split('.')[0]
                WName.append(wnn)
                wdir=os.path.join(root, name)  
                wave=Wave()
                wave.readbyname(wn,wdir)
                wave.N=len(wave.a)
                wave.scaleWaveSa(alpha1,T,fvtime=10) # add 10s free vibration
                sim=simulator(mdof,wave,beta)
                A,V,D,R,U,F=sim.solver() # 2d array: n*steps 
                # hispath=hisf+str(bldg+1)+'-'+wnn+'.txt'
                # sim.save(hispath)
                for i in range (n):
                    IDR[gm,i]=max(max(U[i,:]),abs(min(U[i,:])))/h0 # unit: rad
                    ACC[gm,i]=max(max(A[i,:]),abs(min(A[i,:])))/9.8 # unit: g
                    RIDR[gm,i]=abs(U[i,-1])/h0 # unit: rad
                MIDR[gm]=max(IDR[gm,:])
                PFA[gm]=max(ACC[gm,:])
                MRIDR[gm]=max(RIDR[gm,:])
            gm+=1
    medianMIDR=np.median(MIDR)
    medianPFA=np.median(PFA)
    medianMRIDR=np.median(MRIDR)    
    # %% damage state
    Nflag=0
    if n>3:
        Nflag=1
    ThreS=ThreSData[intensity-1,Nflag,:]
    ThreNSD=[0.004,0.008,0.025,0.05]
    ThreNSA=ThreNSAData[intensity-1,:]
    POSTR=count(ngm,MIDR,ThreS)
    PONSD=count(ngm,MIDR,ThreNSD)
    PONSA=count(ngm,PFA,ThreNSA)
    # %% damage state with MRIDR
    Rlim=0.005 # MRIDR limit 
    POSTRwR=countwR(ngm,MIDR,ThreS,MRIDR,Rlim)
    PONSDwR=countwR(ngm,MIDR,ThreNSD,MRIDR,Rlim)
    PONSAwR=countwR(ngm,PFA,ThreNSA,MRIDR,Rlim)
    # %% loss
    LS=np.dot(POSTR[1:],RCS)/100*BRC
    LNSD=np.dot(PONSD[1:],RCD)/100*BRC
    LNSA=np.dot(PONSA[1:],RCA)/100*BRC
    LC=np.dot(PONSA[1:],CD)/100*CRV
    Ltot=LS+LNSD+LNSA+LC
    Lf=Ltot/Rtot  # loss function
    # %% loss with MRIDR
    LSwR=np.dot(POSTRwR[1:],RCS)/100*BRC
    LNSDwR=np.dot(PONSDwR[1:],RCD)/100*BRC
    LNSAwR=np.dot(PONSAwR[1:],RCA)/100*BRC
    LCwR=np.dot(PONSAwR[1:],CD)/100*CRV
    LtotwR=LSwR+LNSDwR+LNSAwR+LCwR
    LfwR=LtotwR/Rtot  # loss function
    # %% recovery time and resilience
    TRE=np.dot(POSTR[1:],BRT)
    def func(x):
        return 1-Lf*np.exp(-x*np.log(200)/TRE)
    RES1=integrate.quad(func, 0, TRE)[0]
    RES2=TLC-TRE
    RLC=(RES1+RES2)/TLC
    RRE=(RES1+RES2)/480
    # %% recovery time and resilience with MRIDR
    TREwR=np.dot(POSTRwR[1:],BRT)
    def funcwR(x):
        return 1-LfwR*np.exp(-x*np.log(200)/TREwR)
    RES1wR=integrate.quad(funcwR, 0, TRE)[0]
    RES2wR=TLC-TREwR
    RLCwR=(RES1wR+RES2wR)/TLC
    RREwR=RES1wR/480
    # %% plot functionality curve
    t=np.arange(0,500,1)
    tOE=20
    Qt=np.zeros(len(t))
    QtwR=np.zeros(len(t))
    for i in range(len(t)):
        if t[i]<tOE:
            Qt[i]=1
        elif t[i]>TRE+tOE:
            Qt[i]=1
        else:
            Qt[i]=1-Lf*np.exp(-(t[i]-tOE)*np.log(200)/TRE)
    for i in range(len(t)):
        if t[i]<tOE:
            QtwR[i]=1
        elif t[i]>TREwR+tOE:
            QtwR[i]=1
        else:
            QtwR[i]=1-LfwR*np.exp(-(t[i]-tOE)*np.log(200)/TREwR)      
    fig=plt.figure()
    plt.title("functionality curve",fontsize=16)
    plt.xlabel("t(days)",fontsize=14)
    plt.ylabel("Q(t)",fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.axis([0, 500, 0, 1.2])
    plt.plot(t,Qt,label='without Δr')
    plt.plot(t,QtwR,label='with Δr')
    plt.fill_between(t, QtwR, 0, where=(t>tOE)&(t<tOE+TREwR), facecolor='lightgray', alpha=0.2)
    plt.fill_between(t, Qt, 0, where=(t>tOE)&(t<tOE+TRE), facecolor='lightblue', alpha=0.3)
    plt.legend(loc=4,fontsize=14)
    plt.savefig(figf+'bldg'+str(bldg+1)+'.png')
    plt.show()
    #%% record
    mldata[bldg,:]=[a,Tg,n,alpha1,Ct,Cy1,RLC,RLCwR,
          intensity,group,siteClass,T,k0,FEK,Vy[0],
          medianMIDR,medianPFA,medianMRIDR,
          LS,LNSD,LNSA,LC,Lf,TRE,RES1,
          LSwR,LNSDwR,LNSAwR,LCwR,LfwR,TREwR,RES1wR]
    print('Bldg %d: Lf=%2.1f %%, TRE =%d days, R=%1.6f' %(bldg+1,Lf*100,TRE,RLC))
    print('Bldg %d: LfwR=%2.1f %%, TREwR =%d days, RwR=%1.6f' %(bldg+1,LfwR*100,TREwR,RLCwR))
    # %%
    endtime=datetime.datetime.now()
    time=(endtime-starttime).seconds
    print("Bldg "+str(bldg+1)+" takes "+str(time)+" seconds")
    Time.append(time)
    bldg+=1
# %% data for ml
MLData=pd.DataFrame(mldata,columns=['a','Tg','n','alpha1','Ct','Cy1','RLC','RLCwR',
                                    'intensity','group','siteClass','T1','k','VB','Vy1',
                                    'MIDR','PFA','MRIDR',
                                    'LS','LNSD','LNSA','LC','Lf','TRE','RES',
                                    'LSwR','LNSDwR','LNSAwR','LCwR','LfwR','TREwR','RESwR'])
MLDatapath=mldataf+'mldata.xlsx'
with pd.ExcelWriter(MLDatapath) as writer:
    MLData.to_excel(writer)
# %% save time
timepath=timef+'time.csv'
np.savetxt(timepath,Time, delimiter=',')