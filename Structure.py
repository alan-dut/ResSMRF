#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 09:14:51 2019

@author: touki
"""
import numpy as np
import os
from numpy.linalg import inv
from numpy.linalg import eig
# %% Nonlinear
class SDoF:
    def __init__(self,m=1,k=100,h=0.05):
        self.m=m
        self.k=k
        self.h=h
        self.w=np.sqrt(k/m)
        self.f=self.w/2/np.pi
        self.c=2*h*np.sqrt(m*k)
    
    def writebypathandname(self,path='./data/Models/',name='model.txt'):
        if not os.path.exists(path):
            os.makedirs(path)
        fn=path+name
        with open(fn,"w") as f:
            f.write("strucure model\n")
            # then the head
            f.write(" m(kg), k(N/m),   h\n")
            # then for loop
            f.write("%6.2f, %6.2f, %6.2f\n" % (self.m,self.k,self.h))
        
        print("structural model saved to %s" % fn)

# %% Nonlinear
class SDoFNL(SDoF):
    
    def __init__(self,m=1,k=100,h=0.05,ay=0.3,alf=0.1):
        super().__init__(m,k,h)
        self.ay=ay
        self.alf=alf
        self.updateparameters()
        
    def updateparameters(self):
        g=9.8
        self.fy=self.m*self.ay*g
        self.dy=self.fy/self.k
        self.k1=self.k
        self.k2=self.k*self.alf
        self.qc=(1-self.alf)*self.fy
        self.f=0
        self.d=0

        
    def force(self,d):
        dn=self.d
        fn=self.f
        dd=d-dn
        ke=self.k2
        kp=self.k1-self.k2
        fpn=fn-ke*dn
        fe=ke*d
        fp=fpn+kp*dd
        if fp>self.qc:
            fp=self.qc
        if fp<-self.qc:
            fp=-self.qc
        f=fe+fp
        self.f=f
        self.d=d
        return f
# %% Mdof Model
class MDoF(SDoF):
    def __init__(self,m=np.array([1,1,1]),k=np.array([100,100,100]),h=np.array([0.05,0.05,0.05])):
        self.setmodel(m,k,h)

    def setmodel(self,m,k,h):
        self.m=m
        self.k=k
        self.h=h
        self.dof=len(m)
        self.make()
        self.saveMdofMatrix()
    
    def writebypathandname(self,path='./data/Models/',name='Mdof.txt'):
        if not os.path.exists(path):
            os.makedirs(path)
        fn=path+name
        with open(fn,"w") as f:
            f.write("strucure model\n")
            # then the head
            f.write(" m(kg), k(N/m),   h\n")
            # then for loop
            for m,k,h in zip(self.m,self.k,self.h):
                f.write("%6.2f, %6.2f, %6.2f\n" % (m,k,h))

    def readbypathandname(self,path='./data/Models/',name='Mdof.txt'):
        fn=path+name
        print(path,name)
        with open(fn,"r") as f:
            lines=f.readlines()
        m=[]
        k=[]
        h=[]
        for l in lines[2:]:
            words=l.split(',')
            m.append(float(words[0]))
            k.append(float(words[1]))
            h.append(float(words[2]))
        print("----reset----")
        self.setmodel(m,k,h)

    def saveMdofMatrix(self,path='./data/Models/',name='MdofMatrix.txt'):
        if not os.path.exists(path):
            os.makedirs(path)
        fn=path+name
        n=self.dof
        M=self.M
        K=self.K
        C=self.C
        with open(fn,"w") as f:
            f.write("strucure model\n")
            # then the head
            f.write(" M(kg):\n")
            # then for loop
            for i in range(n):
                for j in range(n):
                    f.write("%6.2f" % M[i,j])
                    if j<n-1:
                        f.write(",")
                    else:
                        f.write("\n")
                        
            f.write(" K(N/m):\n")
            # then for loop
            for i in range(n):
                for j in range(n):
                    f.write("%6.2f" % K[i,j])
                    if j<n-1:
                        f.write(",")
                    else:
                        f.write("\n")
                        
            f.write(" C(N/m*s):\n")
            # then for loop            
            for i in range(n):
                for j in range(n):
                    f.write("%6.2f" % C[i,j])
                    if j<n-1:
                        f.write(",")
                    else:
                        f.write("\n")
                        
    def makeM(self,m):
        n=len(m)
#        print(n)
        M=np.zeros((n,n))
        for i in range(n):
            M[i,i]=m[i]
#        print(M)
        return M
    def makeK(self,k):
        n=len(k)
#        print(n)
        K=np.zeros((n,n))
        for i in range(n):
            K[i,i]=k[i]
            if i<n-1:
                K[i,i]=K[i,i]+k[i+1]
                K[i,i+1]=-k[i+1]
            if i>0:
                K[i,i-1]=-k[i]
#        print(K)
        return K
    def makeC(self,M,K,h):
        n=len(h)
#        print(n)
        C=np.zeros((n,n))
        MK=np.matmul(inv(M),K)
        
#        print(MK)
        ww,v=eig(MK)
        w=np.sqrt(ww)
#        print(ww,w)
        w.argsort()
        self.w=w
#        print("w=",self.w)
        self.f=w/2/np.pi
#        print("f=",self.f)
        self.T=1/self.f
#        print("T=",self.T)
#        print(w)
        w1=w[n-1]
        w2=w[n-2]
#        print(w1,w2)
        h1=h[0]
        h2=h[1]
        a0=2*w1*w2*(w2*h1-w1*h2)/(w2*w2-w1*w1)
        a1=2*(w2*h2-w1*h1)/(w2*w2-w1*w1)
#        print(a0,a1)
        C=a0*M+a1*K
#        print(C)
        return C
    def make(self):
        self.M=self.makeM(self.m)
        self.K=self.makeK(self.k)
        self.C=self.makeC(self.M,self.K,self.h)
#        print('model marix builded')
  # %% Mdof Nonlinear Model
class MDoFNolinear(MDoF):      
    def __init__(self,m=np.array([1,1,1]),k=np.array([100,100,100]),h=np.array([0.05,0.05,0.05]),alf=np.array([0.1,0.1,0.1]),ay=np.array([0.3,0.3,0.3])):
        self.setmodel(m,k,h,alf,ay)

    def setmodel(self,m,k,h,alf,ay):
        super().setmodel(m,k,h)
        self.alf=alf
        self.ay=ay # kN/KN
        self.fy=m*ay*9.8 # N
        self.qc=(1-alf)*self.fy
        
    def readbypathandname(self,path='./data/Models/',name='MdofNL.txt'):
        fn=path+name
        print(path,name)
        with open(fn,"r") as f:
            lines=f.readlines()
        m=np.array([])
        k=np.array([])
        h=np.array([])
        alf=np.array([])
        ay=np.array([])
        for l in lines[2:]:
            words=l.split(',')
            np.append(m,float(words[0]))
            np.append(k,float(words[1]))
            np.append(h,float(words[2]))
            np.append(alf,float(words[3]))
            np.append(ay,float(words[4]))
        print("----reset----")
        print(h)
        self.setmodel(m,k,h,alf,ay)
        
    def d2u(self,d):
        n=self.dof
        u=np.zeros(self.dof)
        for i in range(n):
            if i==0:
                u[i]=d[i]
            else:
                u[i]=d[i]-d[i-1]
        return u
    
    def f2r(self,f):
        n=self.dof
        r=np.zeros(self.dof)
        for i in range(n):
            if i==n-1:
                r[i]=f[i]
            else:
                r[i]=f[i]-f[i+1]
        return r
    
    def force(self,d,dn,fn,k1,alf,qc):
        dd=d-dn
        ke=alf*k1
        kp=(1-alf)*k1
        
        fe=ke*d
        fen=ke*dn
        fpn=fn-fen
        fp=fpn+kp*dd
        if fp>qc:
            fp=qc
        if fp<-qc:
            fp=-qc
        f=fe+fp
        
        return f

    def restoringForce(self,d,dn,fn):
        u=self.d2u(d)
        un=self.d2u(dn)
        f=np.zeros(self.dof)
        for i in range(self.dof):
            f[i]=self.force(u[i],un[i],fn[i],self.k[i],self.alf[i],self.qc[i])
        r=self.f2r(f)
#        print(u[0],f[0])
        return r,f,u
                    
    def writebypathandname(self,path='./data/Models/',name='Mdof.txt'):
        if not os.path.exists(path):
            os.makedirs(path)
        fn=path+name
        with open(fn,"w") as f:
            f.write("strucure model\n")
            # then the head
            f.write(" m(kg), k(N/m),   h,    alf,    ay(g) \n")
            # then for loop
            for m,k,h,alf,ay in zip(self.m,self.k,self.h,self.alf,self.ay):
                f.write("%6.2f, %6.2f, %6.2f, %6.4f, %6.2f\n" % (m,k,h,alf,ay))
    def DamageLevel(self,U,Ulimit,A,V):
        UMAX=[]
        AccMAX=[]
        VelMAX=[]
        for i in range(self.dof):
            u=U[i,:]
            umax=max(u)
            umin=(min(u))
            um=max(umax,abs(umin))
            UMAX.append(um)
            acc=A[i,:]
            accmax=max(acc)
            accmin=min(acc)
            accm=max(accmax,abs(accmin))
            AccMAX.append(accm)
            vel=V[i,:]
            velmax=max(vel)
            velmin=min(vel)
            velm=max(velmax,abs(velmin))
            VelMAX.append(velm)
        isDamage=False
        self.umax=max(UMAX)
        self.amax=max(AccMAX)
        self.vmax=max(VelMAX)
        for um,ul in zip(UMAX,Ulimit):
            if um>ul:
                isDamage=True
        return isDamage
  
