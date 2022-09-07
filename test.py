# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 14:22:05 2021

@author: simon
"""

import numpy as np
import math as mt
import matplotlib.pyplot as plt
import cmath as cmt
"""variables"""
ni = 1

a_c=29.2

a_i=0

no = 1.6599
ne = 1.5452

go=0
ge=-1.92*go

thick= 3E-3
lamda= 853*10**-9
sens=1E-8
vnorm=np.zeros((3,1))
vnorm[2]=1

Ei=np.zeros((3,1),dtype=complex)
Ei[0]=1
Ei[1]=0
Ei[2]=0

Ei=Ei/np.linalg.norm(Ei)




def tensor(no,ne,go,ge,vnorm,a_c,a_i):
    ee=ne * ne
    eo=no * no
    
    a_c=a_c * np.pi / 180.
    eje=np.zeros((3,1))
    eje[1]=np.sin(a_c)
    eje[2]=np.cos(a_c)

    z=np.zeros((3,1))
    z[2]=1.
    
    x=np.zeros((3,1))
    x[0]=1.

    ed=np.zeros((3,3))
    ed[0][0]=eo
    ed[1][1]=eo
    ed[2][2]=ee
    
    gd=np.zeros((3,3))
    gd[0][0]=go
    gd[1][1]=go
    gd[2][2]=ge
    
    if eje[2]!=1:
        phi=mt.acos( np.vdot(eje,z) )

        theta=mt.acos(np.vdot(eje,x) / mt.sin(phi))
        R1=np.zeros((3,3))
        R1[0][0]=mt.cos(theta)
        R1[0][1]=mt.sin(theta)
        R1[1][0]=-mt.sin(theta)
        R1[1][1]=mt.cos(theta)
        R1[2][2]=1.
    
        R2=np.zeros((3,3))
        R2[0][0]=mt.cos(phi)
        R2[0][2]=mt.sin(phi)
        R2[1][1]=1.
        R2[2][0]=-mt.sin(phi)
        R2[2][2]=mt.cos(phi)
    
        R=np.matmul(R1,R2)
        
    
        
        e=np.matmul(np.matmul(np.linalg.inv(R),ed),R)
        g=np.matmul(np.matmul(np.linalg.inv(R),gd),R)

    else:
        e=ed
        g=gd
    if a_i==0:
        a_i=0.001
    
    a_i=a_i * np.pi / 180
    ki=np.zeros((3,1))
    ki[1]=np.sin(a_i)
    ki[2]=np.cos(a_i)
    
    return e,g,eje,ki


e,g,eje,ki=tensor(no,ne,go,ge,vnorm,a_c,a_i)

# Ep=np.cross(Ei.transpose(),ki.transpose())[0]
# Ei=np.zeros((3,1),dtype=complex)
# Ei[0]=Ep[0]
# Ei[1]=Ep[1]
# Ei[2]=Ep[2]
# Hi=np.cross(ni*ki.transpose()[0],Ei.transpose()[0])
def k_t1(no,ne,go,ge,eje,ki,ni,vnorm,prop):
    n=no
    k=0
    ee=ne*ne
    eo=no*no
    while k<5:
        gammat=np.vdot(-ni*ki,vnorm)+mt.sqrt(ni*ni*np.vdot(ki,vnorm)**2+(n*n-ni*ni))
        kt=ni*ki+gammat*vnorm
        kt=kt/np.linalg.norm(kt)
        
        theta=mt.acos(np.vdot(kt,eje))
        eop=eo-go*go
        eep=ee-ge*ge
        e=(mt.cos(theta)**2)*eep+(mt.sin(theta)**2)*eop
        v=4*eep*go*go*mt.cos(theta)**2+eop*(go+ge)*(go+ge)*mt.sin(theta)**2
        w=(go-ge)*(go-ge)*(mt.sin(theta)**2)*mt.cos(theta)**2

        n1=eop*(eep+e)+v
        n2=cmt.sqrt(eop*eop*(eep+e)**2+2*eop*v*(eep+e)+v*v-4*(e+w)*eop*eop*eep)
        
        if prop==0:
            no_n=cmt.sqrt((n1-n2)/(2*(e+w)))

        elif prop==1:
            no_n=cmt.sqrt((n1+n2)/(2*(e+w)))
#            print(k)
        n=no_n.real
        k=k+1
    return n,kt
nto,kto=k_t1(no,ne,go,ge,eje,ki,ni,vnorm,0)
nte,kte=k_t1(no,ne,go,ge,eje,ki,ni,vnorm,1)

def k_r1(ni,ki,vnorm,a_i):
    kr=ni*ki-2*ni*np.vdot(ki,vnorm)*vnorm
    kr=kr/np.linalg.norm(kr)

    Ers=np.cross(np.transpose(kr),np.transpose(vnorm))[0]
    Ers=Ers/np.linalg.norm(Ers)
    Erp=np.cross(np.transpose(kr),Ers)[0]
    Erp=Erp/np.linalg.norm(Erp)
    Hrs=ni*np.cross(np.transpose(kr),Ers)[0]
    Hrp=ni*np.cross(np.transpose(kr),Erp)[0]
    
    return(kr,Ers,Erp,Hrs,Hrp)

kr,Ers,Erp,Hrs,Hrp=k_r1(ni,ki,vnorm,a_i)

def pol(kt,ed,gd,n):
    
    KT=np.zeros((3,3))
    KT[0][1]=-kt[2]
    KT[0][2]=kt[1]
    KT[1][0]=kt[2]
    KT[1][2]=-kt[0]
    KT[2][0]=-kt[1]
    KT[2][1]=kt[0]
    
    M=ed+np.dot((n*KT+1j*gd),(n*KT+1j*gd))
    w,vh = np.linalg.eig(M)
    
    wa=sorted(abs(w))
    v=vh
    Et=np.zeros((3,1),dtype=complex)
    Et_=np.zeros((3,1),dtype=complex)
    for a in range(len(Et)):
        if wa[0]==abs(w[a]):
            Et=v[:,a]
        if wa[1]==abs(w[a]):
            Et_=v[:,a]
    
    # u, s, vh = np.linalg.svd(M, full_matrices=True)
    # v=vh.transpose().conjugate()
    # Et=v[:,2]
    # Et_=v[:,1]
    Et=Et/np.linalg.norm(Et)
    Et_=Et_/np.linalg.norm(Et_)
    Ht=np.dot((n*KT+1j*gd),Et)
    Ht_=np.dot((n*KT+1j*gd),Et_)
    


    return Et,Ht,Et_,Ht_

def frho(E,H):
    
    rho=np.cross(np.transpose(E),np.transpose(np.conjugate(H)))
    rho=rho.real
    rho=rho/np.linalg.norm(rho)
    
    return rho


def Fresnel1(ki,vnorm,Eto,Ete,Hto,Hte,Ers,Erp,Hrs,Hrp,Ei):
    q=ki
    Hi=np.cross(ni*ki.transpose()[0],Ei.transpose()[0])
    if np.linalg.norm(np.cross(np.transpose(q),np.transpose(vnorm))[0])==0:
        q=np.zeros((3,1),dtype=complex)
        q[0]=1

    s1=np.cross(q.transpose(),vnorm.transpose())[0]
    s2=np.cross(vnorm.transpose(),s1)[0]
    
    F=np.zeros((4,4),dtype=complex)
    
    F[0][0]=np.dot(s1,Eto)
    F[0][1]=np.dot(s1,Ete)
    F[0][2]=np.dot(-s1,Ers)
    F[0][3]=np.dot(-s1,Erp)

    F[1][0]=np.dot(s2,Eto)
    F[1][1]=np.dot(s2,Ete)
    F[1][2]=np.dot(-s2,Ers)
    F[1][3]=np.dot(-s2,Erp)
    
    F[2][0]=np.dot(s1,Hto)
    F[2][1]=np.dot(s1,Hte)
    F[2][2]=np.dot(-s1,Hrs)
    F[2][3]=np.dot(-s1,Hrp)
    
    F[3][0]=np.dot(s2,Hto)
    F[3][1]=np.dot(s2,Hte)
    F[3][2]=np.dot(-s2,Hrs)
    F[3][3]=np.dot(-s2,Hrp)

    F_1=np.linalg.inv(F)
    F_c=np.zeros((4,1),dtype=complex)
    F_c[0]=np.dot(s1,Ei)
    F_c[1]=np.dot(s2,Ei)
    F_c[2]=np.dot(s1,Hi)
    F_c[3]=np.dot(s2,Hi) 
    Fi=np.dot(F_1,F_c)
    
    
    ato=Fi[0]
    ate=Fi[1]
    aro=Fi[2]
    are=Fi[3]
    
    return ato,ate,aro,are

#ato,ate,ars,arp=Fresnel1(ki,vnorm,Eto,Ete,Hto,Hte,Ers,Erp,Hrs,Hrp,Ei)
#
#
#
#Er=ars*Ers+arp*Erp
#
##Er=Er.real
#R=np.dot(Er,Er.conjugate(Er))
#
##Et=Eto*ato+Ete*ate
##T=np.dot(Et,Et.conjugate(Et))
#ai=a_i*np.pi/180
#cto=np.dot(kto.transpose(),vnorm)
#To=nto*cto*ato*ato/(ni*np.cos(ai))
#cte=np.dot(kte.transpose(),vnorm)
#Te=nte*cte*ate*ate/(ni*np.cos(ai))  

#Total=T+R
#print(Total)

#
def primera_capa(a_i,no,ne,a_c,go,ge,vnorm,ni,Ei):

    e,g,eje,ki=tensor(no,ne,go,ge,vnorm,a_c,a_i)

    Ep=np.cross(Ei.transpose(),ki.transpose())[0]
    Ep=Ep/np.linalg.norm(Ep)
#    print(Ep)
    Ei=np.zeros((3,1),dtype=complex)
    Ei[0]=Ep[0]
    Ei[1]=Ep[1]
    Ei[2]=Ep[2]
    nto,kto=k_t1(no,ne,go,ge,eje,ki,ni,vnorm,0)
    nte,kte=k_t1(no,ne,go,ge,eje,ki,ni,vnorm,1)
    kr,Ers,Erp,Hrs,Hrp=k_r1(ni,ki,vnorm,a_i)

    Eto,Hto,Eto_,Hto_=pol(kto,e,g,nto)
    Ete,Hte,Ete_,Hte_=pol(kte,e,g,nte)
    if np.linalg.norm(abs(Eto)-abs(Ete))<1E-1:
        Eto=Eto_
        Hto=Hto_
    if np.linalg.norm(Eto-Ete)<1E-2:
        Ete=Ete_
        Hte=Hte_
    
    rho_o=frho(Eto,Hto)
    rho_e=frho(Ete,Hte)
    
    ato,ate,ars,arp=Fresnel1(ki,vnorm,Eto,Ete,Hto,Hte,Ers,Erp,Hrs,Hrp,Ei)
    
    Er=arp*Erp+Ers*ars
    R=np.dot(Er,Er.conjugate(Er))
    ai=a_i*np.pi/180
    cto=np.dot(kto.transpose(),vnorm)
    To=no*cto*ato*ato/(ni*np.cos(ai))
    To=To[0]
    cte=np.dot(kte.transpose(),vnorm)
    Te=nte*cte*ate*ate/(ni*np.cos(ai))
    Te=Te[0]
    return R,To,Te

a_i=np.linspace(0,90,num=1000)
Re=[]
To_=[]
Te_=[]
Total=[]
for a in range(len(a_i)):
    R,To,Te=primera_capa(a_i[a],no,ne,a_c,go,ge,vnorm,ni,Ei)
    Re.append(R.real)
    To_.append(To.real)
    Te_.append(Te.real)
    Total.append((R+To+Te).real)
    
# plt.plot(a_i,To_,label='To')
# plt.plot(a_i,Te_,label='Te')
plt.plot(a_i,Re,label='R')
# plt.plot(a_i,Total,label='Total')
print(np.argmin(Re)*90/1000)
plt.ylabel('R')
plt.xlabel('angulo incidente [°]')
plt.legend(loc='center', shadow=True, fontsize='large')
plt.show()