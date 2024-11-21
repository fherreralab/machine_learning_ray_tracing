import jax
import jax.numpy as jnp
from jax import lax, jit, vmap, random, value_and_grad, jacrev, jacfwd, grad
from jax.experimental import optimizers

import jaxopt
from jaxopt import implicit_diff

from jax.config import config
config.update("jax_debug_nans", True)
config.update("jax_enable_x64", True)

def tensor(no,ne,go,ge,vnorm,a_c,a_i):
    ee = ne[0] * ne[0]
    eo = no[0] * no[0]

    go = go
    ge = ge    


    a_c = a_c * jnp.pi / 180
    eje = jnp.zeros((3,1))
    eje = eje.at[1,0].set(jnp.sin(a_c)) # eje[1] =jnp.sin(a_c)
    eje = eje.at[2,0].set(jnp.cos(a_c)) # eje[2] = jnp.cos(a_c)
       
    z = jnp.zeros((3,1))
    z = z.at[2].set(1.) # z[2]=1
 
    x = jnp.zeros((3,1))
    x = x.at[0].set(1.) # y[1]=1

    ed = jnp.diag(jnp.array([eo,eo,ee]))
    
    gd = jnp.diag(jnp.array([go,go,ge]))

    def f_matrix_R(eje,z):
      eje=eje.at[2,0].set((eje[2,0]!=1)*eje[2,0] + (eje[2,0]==1)*0.000000000001)
      phi = jnp.arccos(jnp.vdot(eje,z))
      theta = jnp.arccos(jnp.vdot(eje,x)/jnp.sin(phi))
      
      cos_phi = jnp.cos(phi)
      sin_phi = jnp.sin(phi)
      cos_theta = jnp.cos(theta)
      sin_theta = jnp.sin(theta)

      R1 = jnp.zeros((3,3))
      R1 = R1.at[0,0].set(cos_theta)
      R1 = R1.at[0,1].set(sin_theta)
      R1 = R1.at[1,0].set(-sin_theta)
      R1 = R1.at[1,1].set(cos_theta)
      R1 = R1.at[2,2].set(1.)    

      R2 = jnp.zeros((3,3))
      R2 = R2.at[0,0].set(cos_phi)
      R2 = R2.at[0,2].set(sin_phi)
      R2 = R2.at[1,1].set(1.)
      R2 = R2.at[2,0].set(-sin_phi)
      R2 = R2.at[2,2].set(cos_phi)    

      R = jnp.matmul(R2,R1)
      return R

    R = f_matrix_R(eje,z)
    e = jnp.matmul(jnp.matmul(jnp.linalg.inv(R),ed),R)
    g = jnp.matmul(jnp.matmul(jnp.linalg.inv(R),gd),R)

    e = (eje[2,0]!=1)*e + (eje[2,0]==1)*ed
    g = (eje[2,0]!=1)*g + (eje[2,0]==1)*gd   

    a_i = (a_i==0)*1E-10 + (a_i!=0)*a_i
    
    a_i = a_i * jnp.pi / 180
    ki = jnp.zeros((3,1))
    ki = ki.at[1,0].set(jnp.sin(a_i))
    ki = ki.at[2,0].set(jnp.cos(a_i))    
    return e,g,eje,ki

def almost_zero(t):
    return jax.lax.cond(t > 1.e-10, 
                        lambda _: t,
                        lambda _: equal_zero(t),#1.e-10
                        operand=None)

def equal_zero(t):
    return jax.lax.cond(t < 1.e-10, 
                        lambda _: 1.e-20,
                        lambda _: t,#1.e-10
                        operand=None)

def k_t1(no,ne,go,ge,eje,ki,ni,vnorm,prop):
    n=no[0]
    k=0
    k1=0
    rate=1
    sens=1E-8
    
    ee=ne[0]**2
    eo=no[0]**2

    def f(n_and_kt):
      n,kt = n_and_kt
      gammat = jnp.vdot(-ni*ki,vnorm) + jnp.sqrt(ni*ni*jnp.vdot(ki,vnorm)**2+(n*n-ni*ni))
      kt = ni*ki+gammat*vnorm
      kt = kt/jnp.linalg.norm(kt)
    
      theta = jnp.arccos(jnp.vdot(kt,eje))
      eop = eo-go*go
      eep = ee-ge*ge
      e = (jnp.cos(theta)**2)*eep + (jnp.sin(theta)**2)*eop
      v = 4*eep*go*go*jnp.cos(theta)**2+eop*(go+ge)*(go+ge)*jnp.sin(theta)**2
      w = (go-ge)*(go-ge)*(jnp.sin(theta)**2)*jnp.cos(theta)**2
    
      n1 = eop*(eep+e)+v
      n2 = (eop*eop*(eep+e)**2+2*eop*v*(eep+e)+v*v-4*(e+w)*eop*eop*eep)
      n2 = almost_zero(n2)
      n2 = jnp.sqrt(n2)
      n2 = almost_zero(n2)
      
      no_n = (prop==0)*jnp.sqrt((n1-n2)/(2*(e+w))) + (prop==1)*jnp.sqrt((n1+n2)/(2*(e+w)))
      no_n_and_kt = (no_n,kt)
      return no_n_and_kt

    kt = jnp.ones_like(vnorm)
    n_and_kt = (n,kt)
    for i in range(6):
      n_and_kt_prev, n_and_kt = n_and_kt, f(n_and_kt)

    n,kt = n_and_kt
    return n,kt

def k_r1(ni,ki,vnorm,a_i):
    kr = ni*ki-2*ni*jnp.vdot(ki,vnorm)*vnorm
    kr = kr/jnp.linalg.norm(kr)

    Ers = jnp.cross(jnp.transpose(kr),jnp.transpose(vnorm))[0]
    Ers = Ers/jnp.linalg.norm(Ers)
    Erp = jnp.cross(jnp.transpose(kr),Ers)[0]
    Erp = Erp/jnp.linalg.norm(Erp)
    Hrs = ni*jnp.cross(jnp.transpose(kr),Ers)[0]
    Hrp = ni*jnp.cross(jnp.transpose(kr),Erp)[0]
    
    return kr,Ers,Erp,Hrs,Hrp

def pol(kt,ed,gd,n,prop):
  KT = jnp.zeros((3,3))
  KT=KT.at[0,1].set(-kt[2,0])
  KT=KT.at[0,2].set(kt[1,0])
  KT=KT.at[1,0].set(kt[2,0])
  KT=KT.at[1,2].set(-kt[0,0])
  KT=KT.at[2,0].set(-kt[1,0])
  KT=KT.at[2,1].set(kt[0,0])

  M=ed+jnp.dot((n[0]*KT+1j*gd),(n[0]*KT+1j*gd))

  # u, s, vh = jnp.linalg.svd(M)
  # v=vh.transpose()
  # print(M)
  s, v = jnp.linalg.eigh(M)
  # print(s,v)

  Et=jnp.zeros((3,1),dtype=complex)
  Et_=jnp.zeros((3,1),dtype=complex)

  #s, v=jnp.linalg.eigh(M)
  
  i=jnp.argsort(abs(s))

  #Et=Et.at[0,0].set(v[0,2])
  #Et=Et.at[1,0].set(v[1,2])
  #Et=Et.at[2,0].set(v[2,2])

  #Et_=Et_.at[0,0].set(v[0,1])
  #Et_=Et_.at[1,0].set(v[1,1])
  #Et_=Et_.at[2,0].set(v[2,1])

    
  Et = v[:,i[0]]
  Et = Et[:,None]
  Et_ = v[:,i[1]]
  Et_ = Et_[:,None]


  Et=Et/jnp.linalg.norm(Et)
  Et_=Et_/jnp.linalg.norm(Et_)


  Ht=jnp.dot((n[0]*KT+1j*gd),Et)
  Ht_=jnp.dot((n[0]*KT+1j*gd),Et_)
   
  return Et,Ht,Et_,Ht_

def frho(E,H):
    rho=jnp.cross(jnp.transpose(E),jnp.transpose(jnp.conjugate(H)))
    rho=rho.real
    rho=rho/jnp.linalg.norm(rho)
    return rho

def Fresnel1(ki,vnorm,Eto,Ete,Hto,Hte,Ers,Erp,Hrs,Hrp,Ei,ni):
  q=ki
  Hi=jnp.cross(ni[0]*ki.transpose()[0],Ei.transpose()[0])
  s1=jnp.cross(q.transpose(),vnorm.transpose())[0]
  s1=s1+1j*jnp.zeros_like(s1,dtype=complex)
  s2=jnp.cross(vnorm.transpose(),s1)[0]

  F=jnp.zeros((4,4),dtype=complex)
  
  F=F.at[0,0].set(jnp.dot(s1,Eto.ravel()))
  F=F.at[0,1].set(jnp.dot(s1,Ete.ravel()))
  F=F.at[0,2].set(jnp.dot(-s1,Ers.ravel()))
  F=F.at[0,3].set(jnp.dot(-s1,Erp.ravel()))

  F=F.at[1,0].set(jnp.dot(s2,Eto.ravel()))
  F=F.at[1,1].set(jnp.dot(s2,Ete.ravel()))
  F=F.at[1,2].set(jnp.dot(-s2,Ers.ravel()))
  F=F.at[1,3].set(jnp.dot(-s2,Erp.ravel()))

  F=F.at[2,0].set(jnp.dot(s1,Hto.ravel()))
  F=F.at[2,1].set(jnp.dot(s1,Hte.ravel()))
  F=F.at[2,2].set(jnp.dot(-s1,Hrs.ravel()))
  F=F.at[2,3].set(jnp.dot(-s1,Hrp.ravel()))

  F=F.at[3,0].set(jnp.dot(s2,Hto.ravel()))
  F=F.at[3,1].set(jnp.dot(s2,Hte.ravel()))
  F=F.at[3,2].set(jnp.dot(-s2,Hrs.ravel()))
  F=F.at[3,3].set(jnp.dot(-s2,Hrp.ravel()))

  F_1=jnp.linalg.inv(F)
  F_c=jnp.zeros((4,1),dtype=complex)
  F_c=F_c.at[0].set(jnp.dot(s1,Ei.ravel()))
  F_c=F_c.at[1].set(jnp.dot(s2,Ei.ravel()))
  F_c=F_c.at[2].set(jnp.dot(s1,Hi.ravel()))
  F_c=F_c.at[3].set(jnp.dot(s2,Hi.ravel()))

  Fi=jnp.dot(F_1,F_c)

  ato=Fi[0]
  ate=Fi[1]
  aro=Fi[2]
  are=Fi[3]

  return ato,ate,aro,are

def OPL(rho_o,rho_e,thick,kto,kte,ang_i,lamda,no,ne):
  ang_lo=jnp.arcsin(rho_o[0][1])
  ang_le=jnp.arcsin(rho_e[0][1])

  yto=thick*jnp.tan(ang_lo)
  yte=thick*jnp.tan(ang_le)
  
  lo=jnp.sqrt(yto**2+thick**2)
  le=jnp.sqrt(yte**2+thick**2)

  OPLo=no[0]*lo*jnp.vdot(kto,rho_o)
  OPLe=ne[0]*le*jnp.vdot(kte,rho_e)

  ang_i=ang_i*jnp.pi/180

  OPD=OPLe-OPLo+(yto-yte)*jnp.sin(ang_i)
  desfase=(OPD*2*jnp.pi)/lamda
  return desfase,abs(yto-yte)

def kt_bir_nbir(vnorm,ki,ni,n):
  ni=ni[0]
  gamma_t=-jnp.vdot(ni*ki,vnorm)+jnp.sqrt((ni*ni*jnp.vdot(ki,vnorm)**2)+(n*n-ni*ni))
  
  kt=ni*ki+gamma_t*vnorm
  kt=kt/jnp.linalg.norm(kt)

  Ets=jnp.cross(jnp.transpose(kt),jnp.transpose(vnorm))[0]
  Ets=Ets/jnp.linalg.norm(Ets)

  Etp=jnp.cross(jnp.transpose(kt),Ets)[0]

  Hts=jnp.cross(jnp.transpose(ni*kt),Ets)[0]
  Hts=Hts/jnp.linalg.norm(Hts)

  Htp=jnp.cross(jnp.transpose(ni*kt),Etp)[0]
  Htp=Htp/jnp.linalg.norm(Htp)
  return kt,Ets,Etp,Hts,Htp

def k_r2(no,ne,go,ge,eje,ki,ni,vnorm,prop):
    n=no[0]
    k=0
    k1=0
    rate=1
    sens=1E-8
    
    ee=ne[0]**2
    eo=no[0]**2

    def f(n_and_kt):
      n,kt = n_and_kt
      gammat = jnp.vdot(-ni*ki,vnorm) - jnp.sqrt(ni*ni*jnp.vdot(ki,vnorm)**2+(n*n-ni*ni))
      kt = ni*ki+gammat*vnorm
      kt = kt/jnp.linalg.norm(kt)
    
      theta = jnp.arccos(jnp.vdot(kt,eje))
      eop = eo-go*go
      eep = ee-ge*ge
      e = (jnp.cos(theta)**2)*eep + (jnp.sin(theta)**2)*eop
      v = 4*eep*go*go*jnp.cos(theta)**2+eop*(go+ge)*(go+ge)*jnp.sin(theta)**2
      w = (go-ge)*(go-ge)*(jnp.sin(theta)**2)*jnp.cos(theta)**2
    
      n1 = eop*(eep+e)+v
      n2 = (eop*eop*(eep+e)**2+2*eop*v*(eep+e)+v*v-4*(e+w)*eop*eop*eep)
      n2 = almost_zero(n2)
      n2 = jnp.sqrt(n2)
      no_n = (prop==0)*jnp.sqrt((n1-n2)/(2*(e+w))) + (prop==1)*jnp.sqrt((n1+n2)/(2*(e+w)))
      no_n_and_kt = (no_n,kt)
      return no_n_and_kt

    kt = jnp.ones_like(vnorm)
    n_and_kt = (n,kt)
    for i in range(5):
      n_and_kt_prev, n_and_kt = n_and_kt, f(n_and_kt)

    n,kt = n_and_kt
    return n,kt

def Fresnel2(ki,vnorm,Ets,Etp,Hts,Htp,Ero,Ere,Hro,Hre,Ei,Hi):
  q=ki

  s1=jnp.cross(q.transpose(),vnorm.transpose())[0]
  s2=jnp.cross(vnorm.transpose(),s1)[0]

  F=jnp.zeros((4,4),dtype=complex)

  F=F.at[0,0].set(jnp.dot(s1,Ets.ravel()))
  F=F.at[0,1].set(jnp.dot(s1,Etp.ravel()))
  F=F.at[0,2].set(jnp.dot(-s1,Ero.ravel()))
  F=F.at[0,3].set(jnp.dot(-s1,Ere.ravel()))

  F=F.at[1,0].set(jnp.dot(s2,Ets.ravel()))
  F=F.at[1,1].set(jnp.dot(s2,Etp.ravel()))
  F=F.at[1,2].set(jnp.dot(-s2,Ero.ravel()))
  F=F.at[1,3].set(jnp.dot(-s2,Ere.ravel()))

  F=F.at[2,0].set(jnp.dot(s1,Hts.ravel()))
  F=F.at[2,1].set(jnp.dot(s1,Htp.ravel()))
  F=F.at[2,2].set(jnp.dot(-s1,Hro.ravel()))
  F=F.at[2,3].set(jnp.dot(-s1,Hre.ravel()))

  F=F.at[3,0].set(jnp.dot(s2,Hts.ravel()))
  F=F.at[3,1].set(jnp.dot(s2,Htp.ravel()))
  F=F.at[3,2].set(jnp.dot(-s2,Hro.ravel()))
  F=F.at[3,3].set(jnp.dot(-s2,Hre.ravel()))

  F_1=jnp.linalg.inv(F)
  F_c=jnp.zeros((4,1),dtype=complex)
  F_c=F_c.at[0].set(jnp.dot(s1,Ei.ravel()))
  F_c=F_c.at[1].set(jnp.dot(s2,Ei.ravel()))
  F_c=F_c.at[2].set(jnp.dot(s1,Hi.ravel()))
  F_c=F_c.at[3].set(jnp.dot(s2,Hi.ravel()))

  Fi=jnp.dot(F_1,F_c)

  ats=Fi[0]
  atp=Fi[1]
  aro=Fi[2]
  are=Fi[3]

  return ats,atp,aro,are

def ray_tra(no,ne,thick,go,ge,vnorm,a_c,a_i,ni,lamda,Ei):
  e,g,eje,ki=tensor(no,ne,go,ge,vnorm,a_c,a_i)

  nto,kto=k_t1(no,ne,go,ge,eje,ki,ni,vnorm,0)
  nte,kte=k_t1(no,ne,go,ge,eje,ki,ni,vnorm,1)
  kr,Ers,Erp,Hrs,Hrp=k_r1(ni,ki,vnorm,a_i)

  Eto,Hto,Eto_,Hto_=pol(kto,e,g,nto*jnp.ones(1),0)
  Ete,Hte,Ete_,Hte_=pol(kte,e,g,nte*jnp.ones(1),1)
    
  # Hto=Hto+(jnp.linalg.norm(abs(Eto)-abs(Ete))<1E-3)*(-Hto+Hto_)
  # Eto=Eto+(jnp.linalg.norm(abs(Eto)-abs(Ete))<1E-3)*(-Eto+Eto_)
  
  Hto=Hto+(jnp.linalg.norm(jnp.absolute(Eto)-jnp.absolute(Ete))<1E-1)*(-Hto+Hto_)
  Eto=Eto+(jnp.linalg.norm(jnp.absolute(Eto)-jnp.absolute(Ete))<1E-1)*(-Eto+Eto_)
  
  Hte=Hte+(jnp.linalg.norm(jnp.absolute(Eto)-jnp.absolute(Ete))<1E-1)*(-Hte+Hte_)
  Ete=Ete+(jnp.linalg.norm(jnp.absolute(Eto)-jnp.absolute(Ete))<1E-1)*(-Ete+Ete_)


  rho_o=frho(Eto,Hto)
  rho_e=frho(Ete,Hte)

  ato,ate,ars,arp=Fresnel1(ki,vnorm,Eto,Ete,Hto,Hte,Ers,Erp,Hrs,Hrp,Ei,ni*jnp.ones(1))
  desfase,OPD=OPL(rho_o,rho_e,thick,kto,kte,a_i,lamda,nto*jnp.ones(1),nte*jnp.ones(1))

  ktto,Etos,Etop,Htos,Htop=kt_bir_nbir(vnorm,kto,nto*jnp.ones(1),ni)
  ktte,Etes,Etep,Htes,Htep=kt_bir_nbir(vnorm,kte,nte*jnp.ones(1),ni)

  nroo,kroo=k_r2(no,ne,go,ge,eje,kto,nto,vnorm,0)
  nroe,kroe=k_r2(no,ne,go,ge,eje,kto,nto,vnorm,1)
  nreo,kreo=k_r2(no,ne,go,ge,eje,kte,nte,vnorm,0)
  nree,kree=k_r2(no,ne,go,ge,eje,kte,nte,vnorm,1)
    
  Eroo,Hroo,Eroo_,Hroo_=pol(kroo,e,g,nroo*jnp.ones(1),0)
  Eroe,Hroe,Eroe_,Hroe_=pol(kroe,e,g,nroe*jnp.ones(1),0)
  Ereo,Hreo,Ereo_,Hreo_=pol(kreo,e,g,nreo*jnp.ones(1),1)
  Eree,Hree,Eree_,Hree_=pol(kree,e,g,nree*jnp.ones(1),1)

  Hroo=Hroo+(jnp.linalg.norm(jnp.absolute(Eroo)-jnp.absolute(Eroe))<1E-1)*(-Hroo+Hroo_)
  Eroo=Eroo+(jnp.linalg.norm(jnp.absolute(Eroo)-jnp.absolute(Eroe))<1E-1)*(-Eroo+Eroo_)

  Hroe=Hroe+(jnp.linalg.norm(jnp.absolute(Eroo)-jnp.absolute(Eroe))<1E-1)*(-Hroe+Hroe_)
  Eroe=Eroe+(jnp.linalg.norm(jnp.absolute(Eroo)-jnp.absolute(Eroe))<1E-1)*(-Eroe+Eroe_)

  
  Hreo=Hreo+(jnp.linalg.norm(jnp.absolute(Ereo)-jnp.absolute(Eree))<1E-1)*(-Hreo+Hreo_)
  Ereo=Ereo+(jnp.linalg.norm(jnp.absolute(Ereo)-jnp.absolute(Eree))<1E-1)*(-Ereo+Ereo_)

    
  Hree=Hree+(jnp.linalg.norm(jnp.absolute(Ereo)-jnp.absolute(Eree))<1E-1)*(-Hree+Hree_)
  Eree=Eree+(jnp.linalg.norm(jnp.absolute(Ereo)-jnp.absolute(Eree))<1E-1)*(-Eree+Eree_)
  

  atos,atop,aroo,aroe = Fresnel2(kto,vnorm,Etos,Etop,Htos,Htop,Eroo,Eroe,Hroo,Hroe,Eto,Hto)
  ates,atep,areo,aree = Fresnel2(kte,vnorm,Etes,Etep,Htes,Htep,Ereo,Eree,Hreo,Hree,Ete,Hte)
  #ato=ato*-1
  #ate=ate*-1j
  #ates=ates*1j
  #atep=atep*1j
  #atos=atos*-1
  #ates=ates*-1
  Eout_s=ato*atos+ate*ates*jax.lax.exp(-desfase*1j)
  Eout_p=ato*atop+ate*atep*jax.lax.exp(-desfase*1j)

  return Eout_s,Eout_p,OPD
    
def al_polarization(no,ne,go,ge,vnorm,thick,a_c,a_i,ni,lamda):
  Es=jnp.zeros((3,1))
  Es=Es.at[0].set(1)

  an_i=a_i * jnp.pi / 180
  ki=jnp.zeros((3,1))
  ki=ki.at[1].set(jnp.sin(an_i))
  ki=ki.at[2].set(jnp.cos(an_i))

  Ep=jnp.cross(ki.flatten(),Es.flatten())
  Ep=Ep/jnp.linalg.norm(Ep)
  Ep_1=jnp.zeros((3,1),dtype=complex)
  Ep_1=Ep_1.at[0].set(Ep[0])
  Ep_1=Ep_1.at[1].set(Ep[1])
  Ep_1=Ep_1.at[2].set(Ep[2])
  Ep=Ep_1

  Ei=[Es,Ep] # (RAVH CHECK)
  # print('caca',jnp.vstack((Ep,Es)))

  J_in=jnp.zeros((2,2),dtype=complex)
  J_in=J_in.at[0,0].set(1)
  J_in=J_in.at[1,1].set(1)

  Eout_s=[]
  Eout_p=[]

  for a in range(len(Ei)):
    A,B,C=ray_tra(no,ne,thick,go,ge,vnorm,a_c,a_i,ni,lamda,Ei[a])
    Eout_s.append(A)
    Eout_p.append(B)

  J_out=jnp.zeros((2,2),dtype=complex)
  J_out=J_out.at[0,0].set(Eout_s[0][0])
  J_out=J_out.at[1,0].set(Eout_p[0][0])
  J_out=J_out.at[0,1].set(Eout_s[1][0])
  J_out=J_out.at[1,1].set(Eout_p[1][0])

  # Jon=jnp.array([[J_out[0,0],J_out[0,1]],[J_out[1,0],J_out[1,1]]]) #(Simon)
  Jon = jnp.array([[Eout_s[0][0],Eout_s[1][0]],[Eout_p[0][0],Eout_p[1][0]]])

  U=jnp.zeros((4,4),dtype=complex)
  U=U.at[0,0].set(1.)
  U=U.at[0,3].set(1.)
  U=U.at[1,0].set(1.)
  U=U.at[1,3].set(-1.)
  U=U.at[2,1].set(1.)
  U=U.at[2,2].set(1.)
  U=U.at[3,1].set(1j)
  U=U.at[3,2].set(-1j)
  
  U=U/jnp.sqrt(2)
  Tens_i=jnp.kron(Jon,jnp.conjugate(Jon))
  M=jnp.dot(jnp.dot(U,Tens_i),jnp.linalg.inv(U))
  M=M.real/jnp.max(M.real)

  return M,C


# ----------------------
# Brewster angle

def primera_capa(a_i,no,ne,a_c,go,ge,vnorm,ni):

    a_i = a_i[0] #check this notation
    Ei = jnp.zeros((3,1),dtype=complex)
    Ei = Ei.at[0,0].set(1.)

    e,g,eje,ki=tensor(no,ne,go,ge,vnorm,a_c,a_i)

    Ep = jnp.cross(Ei.transpose(),ki.transpose())[0]
    Ep = Ep/jnp.linalg.norm(Ep)
    Ei = jnp.zeros((3,1),dtype=complex)
    Ei = Ei.at[0,0].set(Ep[0])
    Ei = Ei.at[1,0].set(Ep[1])
    Ei = Ei.at[2,0].set(Ep[2])

    nto,kto = k_t1(no,ne,go,ge,eje,ki,ni,vnorm,0)
    nte,kte = k_t1(no,ne,go,ge,eje,ki,ni,vnorm,1)
    kr,Ers,Erp,Hrs,Hrp = k_r1(ni,ki,vnorm,a_i)

    Eto,Hto,Eto_,Hto_ = pol(kto,e,g,nto*jnp.ones(1),0) # CHECK
    Ete,Hte,Ete_,Hte_ = pol(kte,e,g,nte*jnp.ones(1),1) # CHECK

    if jnp.linalg.norm(jnp.absolute(Eto)-jnp.absolute(Ete))<1E-1:
        Eto=Eto_
        Hto=Hto_
    if jnp.linalg.norm(Eto-Ete)<1E-2:
        Ete=Ete_
        Hte=Hte_
    
    
    ato,ate,ars,arp = Fresnel1(ki,vnorm,Eto,Ete,Hto,Hte,Ers,Erp,Hrs,Hrp,Ei,ni*jnp.ones(1))
    
    Er = arp*Erp+Ers*ars
    R = jnp.dot(Er,jnp.conjugate(Er))
    return jnp.real(R)

#search for the brewster angle using a grid (Simon)
def Brewster_brute_search(no,ne,a_c,go,ge,vnorm,ni):
    a_i = jnp.linspace(0,90,num=100)
    Re = []

    a_temp = 1E+6
    re_temp = 1E+6
    # # (NOTE) vmap later
    for a in a_i:
      R = primera_capa(a,no,ne,a_c,go,ge,vnorm,ni)
      Re.append(jnp.real(R))
    Re_=jnp.array(Re)
    for a in range(len(a_i)):
      if Re_.min()==Re_[a]:
        brews=a_i[a]
    return brews

# search for brewster angle using Adam descent (Rodrigo)
def Brewster_sgd(no,ne,a_c,go,ge,vnorm,ni):

  f_primera_capa = lambda a: primera_capa(jnp.exp(a),no,ne,a_c,go,ge,vnorm,ni)
  a_init = jnp.log(55*jnp.ones(1))

  learning_rate = 1.
  opt_init, opt_update, get_params = optimizers.sgd(learning_rate)
  opt_state = opt_init(a_init)
  
  def step(itr,opt_state):
    params = get_params(opt_state)
    value, grads = jax.value_and_grad(f_primera_capa)(params)
    opt_state = opt_update(itr, grads, opt_state)
    # v_and_g = (value, grads)
    return opt_state#v_and_g,

  num_steps = 15
  opt_params_itr = []
  loss_params_itr = []

  def body_fun(itr,opt_state):
    opt_state = step(itr,opt_state)#value_and_grads, 
    return opt_state

  for itr in range(num_steps):
    opt_state = body_fun(itr,opt_state)
    # print(itr,jnp.exp(get_params(opt_state)))

  params = get_params(opt_state)
  return jnp.exp(params)[0]

# search for brewster angle using JAXOPT and Implicit diff (Rodrigo)
def Brewster_jaxopt(no,ne,a_c,go,ge,vnorm,ni):

  f_primera_capa = lambda a: primera_capa(jnp.exp(a),no,ne,a_c,go,ge,vnorm,ni)
  a_init = jnp.log(55*jnp.ones(1))

  f_ang_brew = lambda a: f_primera_capa()

  learning_rate = 1.
  opt_init, opt_update, get_params = optimizers.sgd(learning_rate)
  opt_state = opt_init(a_init)
  
  def step(itr,opt_state):
    params = get_params(opt_state)
    value, grads = jax.value_and_grad(f_primera_capa)(params)
    opt_state = opt_update(itr, grads, opt_state)
    # v_and_g = (value, grads)
    return opt_state#v_and_g,

  num_steps = 15
  opt_params_itr = []
  loss_params_itr = []

  def body_fun(itr,opt_state):
    opt_state = step(itr,opt_state)#value_and_grads, 
    return opt_state

  for itr in range(num_steps+1):
    opt_state = body_fun(itr,opt_state)
    # print(itr,jnp.exp(get_params(opt_state)))

  params = get_params(opt_state)
  return jnp.exp(params)[0]