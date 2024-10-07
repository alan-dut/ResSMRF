#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 09:31:42 2019

@author: touki
"""
import numpy as np
import matplotlib.pyplot as plt
from Structure import SDoF as model
from Wave import Wave


class NewmarkBetaMethod:
    def __init__(self,md=model(),wv=Wave(),beta=1/6):
        if (wv.N<1):
            print("autoload defualt Wave")
            fn='KMMH161604160125.EW2.txt'
            wv.readbyname(fn)
        else:
            print('using setted wave')
        self.dt=1/wv.fs
        self.N=len(wv.a)
        self.md=md
        self.ag=np.array(wv.a)
        self.ag=self.ag# IS unit m/ss
        self.beta=beta
        self.a=[]
        self.v=[]
        self.d=[]
        
    def settings(self):
        dt=self.dt
        m=self.md.m
        c=self.md.c
        k=self.md.k
        beta=self.beta
        Mh=m+0.5*c*dt+beta*k*dt*dt
        Aa=0.5*c*dt+(0.5-beta)*k*dt*dt
        Av=c+k*dt
        self.Mh=Mh
        self.Aa=Aa
        self.Av=Av
    
    def stepintegration(self,an,vn,dn,ag):
        m=self.md.m
        k=self.md.k
        dt=self.dt
        Aa=self.Aa
        Av=self.Av
        Mh=self.Mh
        beta=self.beta
        Fh=m*ag-Aa*an-Av*vn-k*dn
        a=Fh/Mh
        v=vn+0.5*an*dt+0.5*a*dt
        d=dn+vn*dt+0.5*an*dt*dt+beta*(a-an)*dt*dt
        return a,v,d
    
    def solver(self):
        self.settings()
        an=0
        vn=0
        dn=0
        A=[]
        V=[]
        D=[]
        for ag in self.ag:
            a,v,d=self.stepintegration(an,vn,dn,ag)
            A.append(a)
            V.append(v)
            D.append(d)
            an=a
            vn=v
            dn=d
        self.a=A
        self.v=V
        self.d=D
        print("Solved")

    
    def save(self,fn="result.txt"):
        t=np.arange(self.N)*self.dt
        with open(fn,"w") as f:
            # then the head
            f.write("time(s), ag(m/ss), a(m/ss), v(m/s), d(m)\n")
            # then for loop
            for ti,agi,ai,vi,di in zip(t,self.ag,self.a,self.v,self.d): # here zip make zipped varable so that we can use v,t same time
                f.write("%6.2f, %6.2f, %6.2f, %6.2f, %6.2f\n" % (ti,agi,ai,vi,di))# forgot to change lines
        print("finish writing file")
    def plot(self,x,y,title="fig",xlabel="x",ylabel="y",c="k",fn=""):
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
#        plt.subplot(2, 2, 1)
        plt.plot(x,y)
        path="./data/Figs/"
        if fn!="":
            plt.savefig(path+fn)
        plt.show()
    def plots(self):
        fs=1/self.dt
        N=len(self.ag)
        t=np.arange(N)/fs
        print("Input Earthquake")
        self.plot(t,self.ag,"Input Earthquake",'time(s)','ag(m/ss)','k','ag.jpg')
        self.plot(t,self.a,"Response Acceleration",'time(s)','a(m/ss)','k','a.jpg')
        self.plot(t,self.v,"Response Velosity",'time(s)','v(m/s)','k','v.jpg')
        self.plot(t,self.d,"Response Displacement",'time(s)','d(m)','k','d.jpg')

# %% Nonlinear
from Structure import SDoFNL as nonlinearmodel
class PredictorCorrectorMethod(NewmarkBetaMethod):
    def __init__(self,md=nonlinearmodel(),wv=Wave(),beta=1/6):
        #md=model(nlmd.m,nlmd.k,nlmd.h)
        super().__init__(md,wv,beta)
        self.f=[]
        self.settings()
        
    def settings(self):
        super().settings()
        self.Aa=self.md.c*self.dt+0.5*self.md.k*self.dt*self.dt
        self.Av=self.md.k*self.dt
    def stepIncrementalIntegration(self,an,vn,dn,dag):
        dFh=self.md.m*dag-self.Aa*an-self.Av*vn
        da=dFh/self.Mh
        a=an+da
        v=vn+an*self.dt+0.5*da*self.dt
        d=dn+vn*self.dt+0.5*an*self.dt*self.dt+self.beta*da*self.dt*self.dt
        return a,v,d
    def corrector(self,ap,vp,dp,f,fn,k,dn):
        dd=dp-dn
        df=f-fn
        dQ=df-k*dd
        dac=-dQ/self.Mh
        a=ap+dac
        v=vp+0.5*dac*self.dt
        d=dp+self.beta*dac*self.dt*self.dt
        return a,v,d
    def solverNL(self):
        an=0
        vn=0
        dn=0
        agn=0
        fn=0
        A=[]
        V=[]
        D=[]
        F=[]
        for ag in self.ag:
            dag=ag-agn
            a,v,d=self.stepIncrementalIntegration(an,vn,dn,dag)
            f=self.md.force(d)
            a,v,d=self.corrector(a,v,d,f,fn,self.md.k,dn)
            A.append(a)
            V.append(v)
            D.append(d)
            F.append(f)
            an=a
            vn=v
            dn=d
            agn=ag
            fn=f
        self.a=A
        self.v=V
        self.d=D
        self.f=F
        
        print("Solved")
        return A,V,D,F
    def plots(self):
        super().plots()
        d=np.array(self.d)/self.md.dy
        f=np.array(self.f)/self.md.fy
        self.plot(d,f,"HysteresisCurve",'d/dy','f/fy','k','f2d.jpg')
# %% Mdof solver
from Structure import MDoF as mdof
from numpy.linalg import inv

class MdofNewmarBetaMethod():
    def __init__(self,md=mdof(),wv=Wave(),beta=1/6):
        self.dt=1/wv.fs
        self.N=len(wv.a)
        self.n=md.dof
        self.md=md
        self.ag=np.array(wv.a)
        self.ag=self.ag# IS unit m/ss
        self.beta=beta
        self.a=[]
        self.v=[]
        self.d=[]
        
        
    def settings(self):
        dt=self.dt
        M=self.md.M
        C=self.md.C
        K=self.md.K
        beta=self.beta
        Mh=M+0.5*dt*C+beta*dt*dt*K
        Aa=0.5*dt*C+(0.5-beta)*dt*dt*K
        Av=C+dt*K
        self.Mh=Mh
        self.Mhi=inv(Mh)
        self.Aa=Aa
        self.Av=Av
        
#        print(Mh)

    def stepintegration(self,an,vn,dn,ag):
        M=self.md.M
        K=self.md.K
        dt=self.dt
        Aa=self.Aa
        Av=self.Av
        Mhi=self.Mhi
        beta=self.beta
        fh=M.dot(ag)-Aa.dot(an)-Av.dot(vn)-K.dot(dn)
        a=Mhi.dot(fh)
        da=a-an
        v=vn+an*dt+0.5*da*dt
        d=dn+vn*dt+0.5*an*dt*dt+beta*da*dt*dt
        return a,v,d
    def getavd(self):
        n=self.n
        N=self.N
        an=np.zeros(n)
        vn=np.zeros(n)
        dn=np.zeros(n)
        A=np.zeros((n,N))
        V=np.zeros((n,N))
        D=np.zeros((n,N))
        return an,vn,dn,A,V,D
    
    def solver(self):
        self.settings()
        an,vn,dn,A,V,D=self.getavd()
        
        i=0
        for ag in self.ag:
            a,v,d=self.stepintegration(an,vn,dn,ag*np.ones(self.n))
            
            A[:,i]=a
            V[:,i]=v
            D[:,i]=d
            an=a
            vn=v
            dn=d
            i=i+1
        self.a=A
        self.v=V
        self.d=D
        return A,V,D
        print("Solved")

    
    def save(self,fn="resultmdof.txt"):
        t=np.arange(self.N)*self.dt
        with open(fn,"w") as f:
            # then the head
            f.write("time(s),ag(m/ss)")
            for i in range(self.n):
                f.write(",a%d(m/ss),v%d(m/s), d%d(m)" %(i+1,i+1,i+1))
            f.write("\n")
            # then for loop
            for i in range(self.N):
                f.write("%6.4f,%6.4f" %(t[i],self.ag[i]))
                for j in range(self.n):
                    f.write(",%6.4f,%6.4f,%6.4f" %(self.a[j,i],self.v[j,i],self.d[j,i]))
                f.write("\n")
        #print("finish writing file")
    def plot(self,x,y,title="fig",xlabel="x",ylabel="y",c="k",fn=""):      
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
#        plt.subplot(2, 2, 1)
        plt.plot(x,y)
        path="./data/Figs/"
        if fn!="":
            plt.savefig(path+fn)
        plt.show()
    def plotmdof(self,x,y,title="fig",xlabel="x",ylabel="y",c="k",fn=""):
        n=self.n
        for i in range(n):
            plt.subplot(n,1,i+1)
            yi=y[i,:]
            plt.plot(x,yi)
            plt.title(title+str(i+1))
            plt.xlabel(xlabel)
            plt.ylabel(ylabel+str(i+1))
        path="./data/Figs/"
        if fn!="":
            plt.savefig(path+fn)
        plt.show()
  
    def plots(self):
        fs=1/self.dt
        N=len(self.ag)
        t=np.arange(N)/fs
        print("Input Earthquake")
        plt.figure(1)
        self.plot(t,self.ag,"Input Earthquake",'time(s)','ag(m/ss)','k','ag.jpg')
        plt.figure(2)
        self.plotmdof(t,self.a,"Response accleration(m/ss)",'time(s)','a','k','a.jpg')
        plt.figure(3)
        self.plotmdof(t,self.v,"Response Velosity(m/s)",'time(s)','v','k','v.jpg')
        plt.figure(4)
        self.plotmdof(t,self.d,"Response Displacement(m)",'time(s)','d','k','d.jpg')     
# %% MDoF Nonlinear
from Structure import MDoFNolinear as mdofNL
class MDoFNewmBNL(MdofNewmarBetaMethod):
    def __init__(self,md=mdofNL(),wv=Wave(),beta=1/6):
        super().__init__(md,wv,beta)
        dt=self.dt
        M=md.M
        C=md.C
        K=self.md.K
        Mhf=M+0.5*dt*C
        Aaf=0.5*dt*C
        Avi=K*dt
        Aai=C*dt+K*0.5*dt*dt
        self.Mhfi=inv(Mhf)
        self.Aaf=Aaf
        self.Aai=Aai
        self.Avi=Avi
        
    def getavd(self):
        an,vn,dn,A,V,D=super().getavd()
        fn=np.zeros(self.n)
        rn=np.zeros(self.n)
        un=np.zeros(self.n)
        R=np.zeros((self.n,self.N))
        F=np.zeros((self.n,self.N))
        U=np.zeros((self.n,self.N))
        return an,vn,dn,un,rn,fn,A,V,D,U,F,R
    def incrementalMethod(self,an,vn,dn,dag):
        M=self.md.M
        dt=self.dt
        Aa=self.Aai
        Av=self.Avi
        Mhi=self.Mhi
        beta=self.beta
        dfh=M.dot(dag)-Aa.dot(an)-Av.dot(vn)
        da=Mhi.dot(dfh)
        a=an+da
        v=vn+an*dt+0.5*da*dt
        d=dn+vn*dt+0.5*an*dt*dt+beta*da*dt*dt
        return a,v,d
    def corrector(self,a,v,d,r,rn,dn):
        dr=r-rn
        dd=d-dn
        K=self.md.K
        dq=dr-K.dot(dd)
        dt=self.dt
        Mhi=self.Mhi
        dac=Mhi.dot(dq)
        beta=self.beta
        a=a-dac
        v=v-dac*dt/2
        d=d-beta*dac*dt*dt
        return a,v,d
    def forceMethod(self,ag,an,vn,dn,r):
        M=self.md.M
        Aa=self.Aaf
        C=self.md.C
        Mhi=self.Mhfi
        fh=M.dot(ag)-C.dot(vn)-Aa.dot(an)-r
        a=Mhi.dot(fh)
        beta=self.beta
        dt=self.dt
        da=a-an
        v=vn+an*dt+0.5*da*dt
        d=dn+vn*dt+0.5*an*dt*dt+beta*da*dt*dt
        return a,v,d
    def solver(self):
        self.settings()
        an,vn,dn,un,rn,fn,A,V,D,U,F,R=self.getavd()
        agn=0
        i=0
        for ag in self.ag:
            dag=(ag-agn)*np.ones(self.n)
#            a,v,d=self.stepintegration(an,vn,dn,ag*np.ones(self.n))
            a,v,d=self.incrementalMethod(an,vn,dn,dag)
            r,f,u=self.md.restoringForce(d,dn,fn)
            a,v,d= self.corrector(a,v,d,r,rn,dn)
#            a,v,d=self.forceMethod(ag*np.ones(self.n),an,vn,dn,r)
            A[:,i]=a
            V[:,i]=v
            D[:,i]=d
            F[:,i]=f
            R[:,i]=r
            U[:,i]=u
            an=a
            vn=v
            dn=d
            fn=f
            rn=r
            agn=ag
            i=i+1
        self.A=A
        self.V=V
        self.D=D
        self.U=U
        self.F=F
        #print("Solved")
        return A,V,D,R,U,F
        
    def plotmdofUF(self,x,y,title="fig",xlabel="x",ylabel="y",c="k",fn=""):
        n=self.n
        for i in range(n):
            plt.subplot(n,1,i+1)
            xi=x[i,:]
            yi=y[i,:]
            plt.plot(xi,yi)
            plt.title(title+str(i+1))
            plt.xlabel(xlabel)
            plt.ylabel(ylabel+str(i+1))
        path="./data/Figs/"
        if fn!="":
            plt.savefig(path+fn)
        plt.show()   
    
    def plots(self):
        fs=1/self.dt
        N=len(self.ag)
        t=np.arange(N)/fs
        print("Input Earthquake")
        plt.figure(1)
        self.plot(t,self.ag,"Input Earthquake",'time(s)','ag(m/ss)','k','ag.jpg')
        plt.figure(2)
        self.plotmdof(t,self.A,"Response accleration(m/ss)",'time(s)','a','k','a.jpg')
        plt.figure(3)
        self.plotmdof(t,self.V,"Response Velosity(m/s)",'time(s)','v','k','v.jpg')
        plt.figure(4)
        self.plotmdof(t,self.D,"Response Displacement(m)",'time(s)','d','k','d.jpg')
        plt.figure(5)
        self.plotmdofUF(self.U,self.F,"HysteresisCurves",'Disp(m)','Force(N)','k','ag.jpg')

    def save(self,fn="resultmdofNL.txt"):
        t=np.arange(self.N)*self.dt
        with open(fn,"w") as f:
            # then the head
            f.write("time(s),ag(m/ss)")
            for i in range(self.n):
                f.write(",a%d(m/ss),v%d(m/s), d%d(m), u%d(m), f%d(m)" %(i+1,i+1,i+1,i+1,i+1))
            f.write("\n")
            # then for loop
            for i in range(self.N):
                f.write("%6.4f,%6.4f" %(t[i],self.ag[i]))
                for j in range(self.n):
                    f.write(",%6.4f,%6.4f,%6.4f,%6.4f,%6.4f" %(self.A[j,i],self.V[j,i],self.D[j,i],self.U[j,i],self.F[j,i]))
                f.write("\n")     