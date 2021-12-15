# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 00:59:05 2021

@author: simon
"""

import numpy as np
import math as mt
import cmath as cmt



"""function used to rotate the standard diagonal form of the refractive index
    matrix and the gyrotropic tensor""" 
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
    
        R=np.matmul(R2,R1)
        
    
        
        e=np.matmul(np.matmul(np.linalg.inv(R),ed),R)
        g=np.matmul(np.matmul(np.linalg.inv(R),gd),R)

    else:
        e=ed
        g=gd
    if a_i==0:
        a_i=0.0000000000001
    
    a_i=a_i * np.pi / 180
    ki=np.zeros((3,1))
    ki[1]=np.sin(a_i)
    ki[2]=np.cos(a_i)
    
    return e,g,eje,ki



""" Refraction at Nonbirefringent-to-Birefringent Interfaces""" 
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

        n=no_n.real
        k=k+1


    return n,kt


""" standard form of the law of reflection used for isotropyc material"""
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



"""function used to get the Electric field and the Magnetic field using the 
    eigenvectors of the matrix M"""
def pol(kt,ed,gd,n):
    
    KT=np.zeros((3,3))
    KT[0][1]=-kt[2]
    KT[0][2]=kt[1]
    KT[1][0]=kt[2]
    KT[1][2]=-kt[0]
    KT[2][0]=-kt[1]
    KT[2][1]=kt[0]
    
    M=ed+np.dot((n*KT+1j*gd),(n*KT+1j*gd))
    w,v = np.linalg.eig(M)
    
    wa=sorted(abs(w))
    v=v
    Et=np.zeros((3,1),dtype=complex)
    Et_=np.zeros((3,1),dtype=complex)
    for a in range(len(Et)):
        if wa[0]==abs(w[a]):
            Et=v[:,a]
        if wa[1]==abs(w[a]):
            Et_=v[:,a]
    

    Et=Et/np.linalg.norm(Et)
    Et_=Et_/np.linalg.norm(Et_)
    Ht=np.dot((n*KT+1j*gd),Et)
    Ht_=np.dot((n*KT+1j*gd),Et_)


    return Et,Ht,Et_,Ht_

""" standard form of the Poynting vector"""
def frho(E,H):
    
    rho=np.cross(np.transpose(E),np.transpose(np.conjugate(H)))
    rho=rho.real
    rho=rho/np.linalg.norm(rho)
    
    return rho

""" Fresnel equations for transmition and reflection of an isotropyc-unixial
    interface"""

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

""" Ray transfer and Optical path lenghts for a ray in the y-z plane"""
def OPL(rho_o,rho_e,thick,kto,kte,ang_i,lamda,no,ne):
    ang_lo=mt.asin(rho_o[1])
    ang_le=mt.asin(rho_e[1])

    yto=thick*mt.tan(ang_lo)
    yte=thick*mt.tan(ang_le)
    

    lo=mt.sqrt(yto**2+thick**2)
    le=mt.sqrt(yte**2+thick**2)
    
    

    OPLo=no*lo*np.vdot(kto,rho_o)

    OPLe=ne*le*np.vdot(kte,rho_e)

    ang_i=ang_i*np.pi/180

    OPD=OPLe-OPLo+(yto-yte)*mt.sin(ang_i)
    desfase=(OPD*2*mt.pi)/(lamda)
    
    return desfase,abs(yto-yte)

"""Second Interface birrefringente-isotropyc"""
"""standard form of the law of refraction used for isotropyc material"""

def kt_bir_nbir(vnorm,ki,ni,n):
    gamma_t=-np.vdot(ni*ki,vnorm)+mt.sqrt((ni*ni*np.vdot(ki,vnorm)**2)+(n*n-ni*ni))
    
    kt=ni*ki+gamma_t*vnorm
    kt=kt/np.linalg.norm(kt)
    
    Ets=np.cross(np.transpose(kt),np.transpose(vnorm))[0]
    Ets=Ets/np.linalg.norm(Ets)

    Etp=np.cross(np.transpose(kt),Ets)[0]
    
    Hts=np.cross(np.transpose(ni*kt),Ets)[0]
    Hts=Hts/np.linalg.norm(Hts)
    Htp=np.cross(np.transpose(ni*kt),Etp)[0]
    Htp=Htp/np.linalg.norm(Htp)
    
    return kt,Ets,Etp,Hts,Htp


"""Reflection at Nonbirefringent-to-Birefringent Interfaces"""
def k_r2(no,ne,go,ge,eje,ki,ni,vnorm,prop):
    n=no
    k=0

    ee=ne*ne
    eo=no*no
    while k<5:
        gammat=np.vdot(-ni*ki,vnorm)-mt.sqrt(ni*ni*np.vdot(ki,vnorm)**2+(n*n-ni*ni))
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

"""FRESNEL EQUATIONS FOR TRANSMISSION AND REFLECTION"""
def Fresnel2(ki,vnorm,Ets,Etp,Hts,Htp,Ero,Ere,Hro,Hre,Ei,Hi):
    q=ki
    if np.linalg.norm(np.cross(np.transpose(q),np.transpose(vnorm))[0])==0:
        q=np.zeros((3,1),dtype=complex)
        q[0]=1

    s1=np.cross(q.transpose(),vnorm.transpose())[0]
    s2=np.cross(vnorm.transpose(),s1)[0]

    F=np.zeros((4,4),dtype=complex)
    
    F[0][0]=np.dot(s1,Ets)
    F[0][2]=np.dot(-s1,Ero)
    F[0][3]=np.dot(-s1,Ere)
    

    F[1][1]=np.dot(s2,Etp)
    F[1][2]=np.dot(-s2,Ero)
    F[1][3]=np.dot(-s2,Ere)
    

    F[2][1]=np.dot(s1,Htp)
    F[2][2]=np.dot(-s1,Hro)
    F[2][3]=np.dot(-s1,Hre)
    
    F[3][0]=np.dot(s2,Hts)
    F[3][2]=np.dot(-s2,Hro)
    F[3][3]=np.dot(-s2,Hre)

    F_1=np.linalg.inv(F)
    F_c=np.zeros((4,1),dtype=complex)
    F_c[0]=np.dot(s1,Ei)
    F_c[1]=np.dot(s2,Ei)
    F_c[2]=np.dot(s1,Hi)
    F_c[3]=np.dot(s2,Hi) 
    Fi=np.dot(F_1,F_c)
    
    
    ats=Fi[0]
    atp=Fi[1]
    aro=Fi[2]
    are=Fi[3]
    
    return ats,atp,aro,are



""" general algorithm for the two interfaces"""

def ray_tra(no,ne,thick,go,ge,a_c,a_i,ni,lamda,Ei):
    e,g,eje,ki=tensor(no,ne,go,ge,vnorm,a_c,a_i)
    nto,kto=k_t1(no,ne,go,ge,eje,ki,ni,vnorm,0)
    nte,kte=k_t1(no,ne,go,ge,eje,ki,ni,vnorm,1)
    
    kr,Ers,Erp,Hrs,Hrp=k_r1(ni,ki,vnorm,a_i)
    Eto,Hto,Eto_,Hto_=pol(kto,e,g,nto)
    Ete,Hte,Ete_,Hte_=pol(kte,e,g,nte)

    if np.linalg.norm(abs(Eto)-abs(Ete))<1E-1:
        Eto=Eto_
        Hto=Hto_
    if np.linalg.norm(abs(Eto)-abs(Ete))<1E-1:
        Ete=Ete_
        Hte=Hte_
    rho_o=frho(Eto,Hto)
    rho_e=frho(Ete,Hte)

    ato,ate,ars,arp=Fresnel1(ki,vnorm,Eto,Ete,Hto,Hte,Ers,Erp,Hrs,Hrp,Ei)

    desfase,OPD=OPL(rho_o,rho_e,thick,kto,kte,a_i,lamda,nto,nte)

    ktto,Etos,Etop,Htos,Htop=kt_bir_nbir(vnorm,kto,nto,ni)
    ktte,Etes,Etep,Htes,Htep=kt_bir_nbir(vnorm,kte,nte,ni)
    
    nroo,kroo=k_r2(no,ne,go,ge,eje,kto,nto,vnorm,0)
    nroe,kroe=k_r2(no,ne,go,ge,eje,kto,nto,vnorm,1)
    nreo,kreo=k_r2(no,ne,go,ge,eje,kte,nte,vnorm,0)
    nree,kree=k_r2(no,ne,go,ge,eje,kte,nte,vnorm,1)
    
    Eroo,Hroo,Eroo_,Hroo_=pol(kroo,e,g,nroo)
    Eroe,Hroe,Eroe_,Hroe_=pol(kroe,e,g,nroe)
    Ereo,Hreo,Ereo_,Hreo_=pol(kreo,e,g,nreo)
    Eree,Hree,Eree_,Hree_=pol(kree,e,g,nree)

    if np.linalg.norm(abs(Eroo)-abs(Eroe))<10**-1:
            Eroo=Eroo_
            Hroo=Hroo_
    if np.linalg.norm(abs(Eroo)-abs(Eroe))<10**-1:
            Eroe=Eroe_
            Hroe=Hroe_
    
    if np.linalg.norm(abs(Ereo)-abs(Eree))<10**-1:
            Ereo=Ereo_
            Hreo=Hreo_
    if np.linalg.norm(abs(Ereo)-abs(Eree))<10**-1:
            Eree=Eree_
            Hree=Hree_
 
    atos,atop,aroo,aroe = Fresnel2(kto,vnorm,Etos,Etop,Htos,Htop,Eroo,Eroe,Hroo,Hroe,Eto,Hto)
    ates,atep,areo,aree = Fresnel2(kte,vnorm,Etes,Etep,Htes,Htep,Ereo,Eree,Hreo,Hree,Ete,Hte)
    
    Eout_s=ato*atos+ate*ates*cmt.exp(-desfase*1j)
    Eout_p=ato*atop+ate*atep*cmt.exp(-desfase*1j)

    ang_s=np.angle(Eout_p)
    Eout_s=Eout_s/(np.cos(ang_s)+1j*np.sin(ang_s))
    Eout_p=Eout_p/(np.cos(ang_s)+1j*np.sin(ang_s))
    
    
    return Eout_s,Eout_p,OPD




""" general algorithm for the polarizations, in this case we used the V (vectical)
    and H (horizontal) base to obtained the Jones matrix"""

def all_polarization(no,ne,go,ge,thick,a_c,a_i,ni,lamda):
    Es=np.zeros((3,1))
    Es[0]=1
    
    ai=a_i * np.pi / 180
    ki=np.zeros((3,1))
    ki[1]=np.sin(ai)
    ki[2]=np.cos(ai)
    
    Ep=np.cross(ki.flatten(),Es.flatten())
    Ep_1=np.zeros((3,1),dtype=complex)
    Ep_1[0]=Ep[0]
    Ep_1[1]=Ep[1]
    Ep_1[2]=Ep[2]
    Ep=Ep_1


    Ei=[Es,Ep]
    
    J_in=np.zeros((2,2),dtype=complex)
    J_in[0,0]=1
    J_in[1,1]=1
    
    EH=Es+1*Ep
    
    EH=EH/np.linalg.norm(EH)
    JH=np.zeros((2,1),dtype=complex)
    JH[0]=1/mt.sqrt(2)
    JH[1]=1/mt.sqrt(2)
    Eout_s=[]
    Eout_p=[]
    OPd=[]
    for a in range(len(Ei)):
        A,B,C=ray_tra(no,ne,thick,go,ge,a_c,a_i,ni,lamda,Ei[a])


        Eout_s.append(A)
        Eout_p.append(B)
    
    OPD=C
    J_out=np.zeros((2,2),dtype=complex)
    J_out[0,0]=Eout_s[0]
    J_out[1,0]=Eout_p[0]
    J_out[0,1]=Eout_s[1]
    J_out[1,1]=Eout_p[1]

 


    Jon=np.array([[J_out[0,0],J_out[0,1]],[J_out[1,0],J_out[1,1]]])

    U=np.zeros((4,4),dtype=complex)
    U[0][0]=1
    U[0][3]=1
    U[1][0]=1
    U[1][3]=-1
    U[2][1]=1
    U[2][2]=1
    U[3][1]=1j
    U[3][2]=-1j

    
    
    U=U/mt.sqrt(2)
    Tens_i=np.kron(Jon,np.conjugate(Jon))
    M=np.matmul(np.matmul(U,Tens_i),np.linalg.inv(U))
    M=M.real/np.max(M.real)

    return M,OPD*1E6


"""general algorithm for only the first interface (Nonbirefringent-to-Birefringent)"""
def first_interface(a_i,no,ne,a_c,go,ge,vnorm,ni,Ei):

    e,g,eje,ki=tensor(no,ne,go,ge,vnorm,a_c,a_i)

    Ep=np.cross(Ei.transpose(),ki.transpose())[0]
    Ep=Ep/np.linalg.norm(Ep)

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
        
    if np.linalg.norm(abs(Eto)-abs(Ete))<1E-1:
        Ete=Ete_
        Hte=Hte_
    
   
    ato,ate,ars,arp=Fresnel1(ki,vnorm,Eto,Ete,Hto,Hte,Ers,Erp,Hrs,Hrp,Ei)
    
    Er=arp*Erp+Ers*ars
    R=np.dot(Er,Er.conjugate(Er))


    return R

""" Brewster angle function"""
def Brewster(no,ne,a_c,go,ge,vnorm,ni,Ei):
    a_i=np.linspace(50,90,num=400)
    Re=[]

    for a in range(len(a_i)):
        R=first_interface(a_i[a],no,ne,a_c,go,ge,vnorm,ni,Ei)
        Re.append(R.real)
    Re_=np.array(Re)
    for a in range(len(a_i)):
        if Re_.min()==Re[a]:
            brews=a_i[a]
    return brews
