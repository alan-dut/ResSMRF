#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 09:24:07 2019

@author: touki
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft,ifft

class Wave:
    def __init__(self,fs=100,a=[]):
        self.a=a
        self.fs=fs
        self.N=len(a)
        self.PGA=0.0
 
    def readbyname(self,ww,fn):
        #% OpenFile
#        try:
        with open(fn,'r',encoding='shift-JIS') as f:
            lines=f.readlines()
        #print(lines)
        # now the data was read as strings in thouse lines
        
            # let's split them
            a=[]
            i=0
            j=0
            for l in lines:
                i+=1
                words=l.split()
                for s in words:
                    #if s=="date:":
                        #N=float(words[-1])
                    if s=='dt:':
                        dt=float(words[-1])
                        j=i
            for l in lines[j:]:
                words=l.split()
                a.append(float(words[0]))
            self.fs=1/dt
            self.a=a
            self.a0=a
            PGA=max(max(a),abs(min(a)))/9.8  # m/ss to g
            self.PGA=PGA
            self.PGA0=PGA
#            print("wave:%s, len=%d, PGA=%2.2f g" % (ww,len(a),PGA))

#    def scaleWave(self,PGAlevel):
#        a=np.array(self.a0)
#        pga0=self.PGA0
#        sf=PGAlevel/pga0
#        scaleda=a*sf
#        self.a=scaleda
    
    def scaleWaveSa(self,alpha1,T,fvtime=0):
        a=np.array(self.a0)
        w=2*np.pi/T
        Sa0=self.SpectrumSolver(w,0.05,1/6) # m/s/s
        sf=alpha1*9.8/Sa0
        scaleda=a*sf
#        self.a=scaleda
        fvt=np.zeros(int(fvtime*self.fs))
        self.a=np.concatenate([scaleda,fvt])
    
    
    def WavePGA(self):
        a=np.array(self.a)
        PGA=max(max(a),abs(min(a)))/9.8  # m/ss to g
        self.PGA=PGA
        return PGA
    
    def plotwave(self):
        a=self.a
        fs=self.fs
        N=len(a)
        t=np.arange(N)/fs
        plt.plot(t,a)
        plt.show
        
    def writebyname(self,fn):
        a=self.a
        fs=self.fs
        N=len(a)
        t=np.arange(N)/fs
        with open(fn,"w") as f:
            # then the head
            f.write("time(s), acc(g)\n")
            # then for loop
            for ti,ai in zip(t,a): # here zip make zipped varable so that we can use v,t same time
                f.write("%6.2f, %6.2f\n" % (ti,ai))# forgot to change lines
        
        print("finish writing file")
# %% Sperctra

    def SpectrumSolver(self,w,h=0.05,beta=1/6):
        Ag=np.array(self.a[:1000])
        dt=1/self.fs
        w2=w*w
        dt2=dt*dt
        Mh=1+h*w*dt+beta*w2*dt2
        Aa=h*w*dt+(0.5-beta)*w2*dt2
        Av=2*h*w+w2*dt
        an=0
        vn=0
        dn=0
        A=[]
        V=[]
        D=[]
        for ag in Ag:
            Fh=ag-Aa*an-Av*vn-w2*dn
            a=Fh/Mh
            da=a-an
            v=vn+an*dt+0.5*da*dt
            d=dn+vn*dt+0.5*an*dt2+beta*da*dt2
            A.append(a)
            V.append(v)
            D.append(d)
            an=a
            vn=v
            dn=d
        Z=Ag-np.array(A)
        Sa=max(abs(Z))
        return Sa        
    def responseSpectrum(self,T,Ts,Te,h=0.05,beta=1/6):
        Sa=[]
        for t in T:
            sa=0
            if t>Ts and t<Te:
                f=1/t
                w=f*2*np.pi
                sa=self.SpectrumSolver(w,h,beta)
            Sa.append(sa)
        return Sa
    
    
    def designcodeSa(self,T):
        Sa=20
        if T<=0.3:
            Sa=44.63*T**(2/3)
        elif T>=0.7:
            Sa=11.04*T**(-5/3)
        return Sa
    def designSpactrum(self,T,Ts,Te):
        Sa=[]
        
        for t in T:
            sa=0
            if t>Ts and t<Te:
                sa=self.designcodeSa(t)
            Sa.append(sa)
        return Sa
            
# %% Setting for fft
    def forFFT(self):
        n=self.N
        dt=1/self.fs
        
        Twave=dt*n
        f0=1/Twave
        
        w0=2*np.pi*f0
        nf=round(n/2)
        w=np.arange(nf)*w0
        f=np.arange(nf)*f0
        T=np.zeros_like(f)
        T[1:]=1/f[1:]
        T[0]=Twave*10
        return nf,w,f,T
    def SpectrumFitting(self,Ts,Te,h,beta,err):
        n,w,f,T=self.forFFT()
        Sa=self.responseSpectrum(T,Ts,Te,h,beta)
        SaT=self.designSpactrum(T,Ts,Te)
        
        # %%
        Sa=np.array(Sa)
        SaT=np.array(SaT)
        Sa0=Sa
        
        Y=fft(self.a)
        
        # %%
        
        while True:
            R=np.zeros_like(T)
            for i in range(len(T)):
                t=T[i]
                if t>=Ts and t<=Te:
                    R[i]=SaT[i]/Sa[i]
                    Y[i]=R[i]*Y[i]
                    j=n-i
                    Y[j]=np.conj(Y[i])
            
            ag=ifft(Y)
            self.a=ag
            Sa=self.responseSpectrum(T,Ts,Te,h=0.05,beta=1/6)
        
        #    
            R1=max(Sa-SaT)
            R2=min(Sa-SaT)
            Err=max(R1,abs(R2))
            print("Err= %6.4f" %Err)
            if Err<err:
                break
        return T,Sa0,Sa,SaT
    def plotMachSa(self,T,Ts,Te,SaT,Sa0,Sa,Index=1):
        plt.figure(Index)
        plt.semilogx(T,Sa0,label="Sa0")
        plt.semilogx(T,Sa,label="Sa")
        plt.semilogx(T,SaT,label="SaT")
        plt.xlim(Ts,Te)
        plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0, fontsize=12)