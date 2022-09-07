# -*- coding: utf-8 -*-
"""
Created on Tues Aug 2 10:02:05 2022

@author: ricky
"""

import numpy as np
import matplotlib.pyplot as plt
from ray_tracing_algorithm import Brewster

# number of points (angles) to plot (between 0 and 90 degrees)
l = 100
# refractive index 1st material (using air)
n1 = 1
# incident angle spectra
a_i = np.linspace(0, np.pi/2, num=l)

def reflect_params(no, ne, a_c):
    # indices of refraction
    eo = no**2
    ee = ne**2
    delta_eps = ee - eo
    # transformation params (alpha**2 + beta**2 + gamma**2 = 1)
    alpha = np.cos(a_c)
    beta = 0 # literature convention, optic axis in the plane of incidence 
    gamma = np.sin(a_c)
    return (eo, ee, delta_eps, alpha, beta, gamma)

def brewster_angle(no, ne, a_c):
    eo, ee, delta_eps, alpha, beta, gamma = reflect_params(no, ne, a_c)
    eg = eo + gamma**2*delta_eps
    e1 = n1**2
    eag = ee - beta**2*delta_eps
    # assumes optic axis lies in plane of incidence (beta = 0)
    brew_act = np.arctan(np.sqrt((eo*ee - e1*eg)/(e1*(eg - e1))))*(180/np.pi)
    # no assumptions, 1st order approximation for stranger geometries
    brew_fo_approx = np.arctan(np.sqrt(eo/e1 + ((alpha**2*eo - gamma**2*e1)*delta_eps)/(e1*(eo - e1))))*(180/np.pi)
    # no assumptions, better approximation
    brew_approx = np.arctan(np.sqrt((eo*eag - e1*eg)/(e1*(eg - e1))))*(180/np.pi)
    return brew_act, brew_fo_approx, brew_approx

def electric_field(lamb, eo, ee, delta_eps, alpha, beta, gamma, ang):
    # wavenumber
    k = 2*np.pi/lamb
    # wave vector components
    K = k*np.sin(ang)
    q1 = k*np.cos(ang)
    e1 = n1**2
    # normal mode solutions
    d = eo*(ee*(eo + gamma**2*delta_eps)*k**2 - (ee - beta**2*delta_eps)*K**2)
    qo_p = np.sqrt(eo*k**2 - K**2)
    qo_m = -np.sqrt(eo*k**2 - K**2)
    qe_p = (np.sqrt(d) - alpha*gamma*K*delta_eps)/(eo + gamma**2*delta_eps)
    qe_m = (-np.sqrt(d) - alpha*gamma*K*delta_eps)/(eo + gamma**2*delta_eps)
    # E-field eigenvectors
    Eo = np.array([-beta*qo_p, alpha*qo_p - gamma*K, beta*K])
    Eo /= np.linalg.norm(Eo)
    Ee = np.array([alpha*qo_p**2 - gamma*qe_p*K, beta*eo*K**2, gamma*(eo*k**2 - qe_p**2) - alpha*qe_p*K])
    Ee /= np.linalg.norm(Ee)
    return Eo, Ee, e1, qo_p, qe_p, q1, k, K

def electric_field_vec(lamb, eo, ee, delta_eps, alpha, beta, gamma):
    # wavenumber
    k = 2*np.pi/lamb
    # wave vector components
    K = k*np.sin(a_i)
    q1 = k*np.cos(a_i)
    e1 = n1**2
    # normal mode solutions
    d = eo*(ee*(eo + gamma**2*delta_eps)*k**2 - (ee - beta**2*delta_eps)*K**2)
    qo_p = np.sqrt(eo*k**2 - K**2)
    qo_m = -np.sqrt(eo*k**2 - K**2)
    qe_p = (np.sqrt(d) - alpha*gamma*K*delta_eps)/(eo + gamma**2*delta_eps)
    qe_m = (-np.sqrt(d) - alpha*gamma*K*delta_eps)/(eo + gamma**2*delta_eps)
    # E-field eigenvectors
    Eo = np.stack((-beta*qo_p, alpha*qo_p - gamma*K, beta*K), axis=-1)
    Eo /= np.linalg.norm(Eo, axis=1)[:,np.newaxis]
    Ee = np.stack((alpha*qo_p**2 - gamma*qe_p*K, beta*eo*K**2, gamma*(eo*k**2 - qe_p**2) - alpha*qe_p*K), axis=-1)
    Ee /= np.linalg.norm(Ee, axis=1)[:,np.newaxis]
    return Eo, Ee, e1, qo_p, qe_p, q1, k, K

def fresnel_coeffs(Eo, Ee, e1, qo_p, qe_p, q1, k, K, ang):
    # angle-dependent constants
    A = (qo_p + q1 + K*np.tan(ang))*Eo[0] - K*Eo[2]
    B = (qe_p + q1 + K*np.tan(ang))*Ee[0] - K*Ee[2]
    D = (q1 + qe_p)*A*Ee[1] - (q1 + qo_p)*B*Eo[1]
    qt = e1*k**2/q1
    # reflectivity values
    # for beta=0, reduces to (q1 - qo_p) / (q1 + qo_p)
    r_ss = ((q1 - qe_p)*Ee[1]*A - (q1 - qo_p)*Eo[1]*B)/D 
    r_sp = 2*n1*k*(Ee[0]*A - Eo[0]*B)/D
    t_so = -2*q1*B/D
    t_se = -2*q1*A/D
    # for beta=0, reduces to (K*(qe_p*qt - eo*k**2) - alpha*delta_eps*(gamma*(qe_p - qt) + alpha*K)*k**2) / (K*(qe_p*qt + eo*k**2) + alpha*delta_eps*(gamma*(qe_p + qt) + alpha*K)*k**2)
    r_pp = (2*qt/D)*((q1 + qe_p)*Eo[0]*Ee[1] - (q1 + qo_p)*Ee[0]*Eo[1]) - 1
    r_ps = 2*n1*k*(qe_p - qo_p)*Eo[1]*Ee[1]/D
    t_po = 2*n1*k*(q1 + qe_p)*Ee[1]/D
    t_pe = -2*n1*k*(q1 + qo_p)*Eo[1]/D
    return r_ss, r_sp, t_so, t_se, r_pp, r_ps, t_po, t_pe

def fresnel_coeffs_vec(Eo, Ee, e1, qo_p, qe_p, q1, k, K):
    # angle-dependent constants
    A = (qo_p + q1 + K*np.tan(a_i))*Eo[:,0] - K*Eo[:,2]
    B = (qe_p + q1 + K*np.tan(a_i))*Ee[:,0] - K*Ee[:,2]
    D = (q1 + qe_p)*A*Ee[:,1] - (q1 + qo_p)*B*Eo[:,1]
    qt = e1*k**2/q1
    # reflectivity values
    # for beta=0, reduces to (q1 - qo_p) / (q1 + qo_p)
    r_ss = ((q1 - qe_p)*Ee[:,1]*A - (q1 - qo_p)*Eo[:,1]*B)/D 
    r_sp = 2*n1*k*(Ee[:,0]*A - Eo[:,0]*B)/D
    t_so = -2*q1*B/D
    t_se = -2*q1*A/D
    # for beta=0, reduces to (K*(qe_p*qt - eo*k**2) - alpha*delta_eps*(gamma*(qe_p - qt) + alpha*K)*k**2) / (K*(qe_p*qt + eo*k**2) + alpha*delta_eps*(gamma*(qe_p + qt) + alpha*K)*k**2)
    r_pp = (2*qt/D)*((q1 + qe_p)*Eo[:,0]*Ee[:,1] - (q1 + qo_p)*Ee[:,0]*Eo[:,1]) - 1
    r_ps = 2*n1*k*(qe_p - qo_p)*Eo[:,1]*Ee[:,1]/D
    t_po = 2*n1*k*(q1 + qe_p)*Ee[:,1]/D
    t_pe = -2*n1*k*(q1 + qo_p)*Eo[:,1]/D
    return r_ss, r_sp, t_so, t_se, r_pp, r_ps, t_po, t_pe

def jones(r_ss, r_sp, r_pp, r_ps):
    return np.matrix([[r_ss, r_ps], [r_sp, r_pp]])

def mueller(jon):
    A = np.matrix([[1, 0,  0,  1],
                   [1, 0,  0, -1],
                   [0, 1,  1,  0],
                   [0, 1, -1,  0]])
    mul = A @ np.kron(jon, np.conjugate(jon)) @ np.linalg.inv(A)
    return mul / mul[0,0]

def reflectivity_curve(no, ne, a_c, lamb):
    eo, ee, delta_eps, alpha, beta, gamma = reflect_params(no, ne, a_c)
    Eo, Ee, e1, qo_p, qe_p, q1, k, K = electric_field_vec(lamb, eo, ee, delta_eps, alpha, beta, gamma)
    r_ss, r_sp, t_so, t_se, r_pp, r_ps, t_po, t_pe = fresnel_coeffs_vec(Eo, Ee, e1, qo_p, qe_p, q1, k, K)
    # initial E-fields (s and p polarized)
    Ers = np.zeros((l, 3))
    Ers[:,1] = 1
    Erp = np.zeros((l, 3))
    ki = np.stack((K, np.zeros(l), q1), axis=-1)
    Erp = np.cross(Ers, ki)
    Erp /= np.linalg.norm(Erp, axis=1)[:,np.newaxis]
    Es = r_sp.reshape(l, 1)*Erp + r_ss.reshape(l, 1)*Ers
    Ep = r_ps.reshape(l, 1)*Ers + r_pp.reshape(l, 1)*Erp
    Rs = (Es*Es.conjugate(Es)).sum(axis=-1)
    Rp = (Ep*Ep.conjugate(Ep)).sum(axis=-1)

    brew = np.argmin(Rp)
    return (brew, Rs, Rp)

def main():
    brew_info = 57.091 # for lambda = 853e-9, Eimerl refractiveindex.info et al 1987
    no = 1.6599
    ne = 1.5452
    # crystal angle
    a_c = 29.2*(np.pi/180)
    # using lambda = 853e-9
    lamb = 853e-9
    brew_act, brew_fo_approx, brew_approx = brewster_angle(no, ne, a_c)
    brew_graph, Rs, Rp = reflectivity_curve(no, ne, a_c, lamb) 
    print("measured BBO brewster angle (graph): ", brew_graph*90/l)
    print("BBO brewster angle (analytic): ", brew_act)
    print("BBO brewster angle (first order approximation): ", brew_fo_approx)
    print("BBO brewster angle (better approximation): ", brew_approx)
    print("actual BBO brewster angle (refractiveindex.info): ", brew_info)
    print("error (analytic/source): ", (np.abs(brew_info - brew_act)/brew_act)*100)

    Es = np.zeros((3, 1))
    Es[0] = 1 # raytracing code uses yz plane
    vnorm = np.zeros((3, 1))
    vnorm[2] = 1
    # BBO params:        no  ne  a_c  go ge vnorm  ni  Ei (s polarization)
    BBO_angle = Brewster(no, ne, a_c, 0, 0, vnorm, n1, Es) 
    print("experimental BBO brewster angle (raytracing): ", BBO_angle)

    plt.plot(a_i*(180/np.pi), Rs, label='S-polarized')
    plt.plot(a_i*(180/np.pi), Rp, label='P-polarized')
    plt.plot(brew_graph*90/l, Rp[brew_graph], marker="o", label="Brewster's angle")
    plt.legend()
    plt.xlabel('angle of incidence, deg')
    plt.ylabel('Reflectance')
    plt.show()

    ang = 5*(np.pi/180)
    eo, ee, delta_eps, alpha, beta, gamma = reflect_params(no, ne, a_c)
    Eo, Ee, e1, qo_p, qe_p, q1, k, K = electric_field(lamb, eo, ee, delta_eps, alpha, beta, gamma, ang)
    r_ss, r_sp, t_so, t_se, r_pp, r_ps, t_po, t_pe = fresnel_coeffs(Eo, Ee, e1, qo_p, qe_p, q1, k, K, ang)
    jon = jones(r_ss, r_sp, r_pp, r_ps)
    mul = mueller(jon)
    Ers = np.zeros((1, 3))
    Ers[0,1] = 1
    ki = np.array([K, 0, q1])
    Erp = np.cross(Ers, ki)
    Erp /= np.linalg.norm(Erp)
    print("Mueller matrix:\n", mul)

    # verifying calcite brewster angle (Figure 1 Lekner et al. 1993)
    no = 1.655 
    ne = 1.485
    lamb = 633e-9

    cal_zs = np.linspace(0, 1, l)
    brewsters = np.zeros(l)
    brewsters_fo = np.zeros(l)
    brewsters_approx = np.zeros(l)
    for i in range(l):
        a_c = np.arcsin(np.sqrt(cal_zs[i]))
        brewsters[i], brewsters_fo[i], brewsters_approx[i] = brewster_angle(no, ne, a_c)

    plt.plot(cal_zs, brewsters, label=r'$\beta=0$')
    plt.plot(cal_zs, brewsters_fo, '--', label=r'$1^{st}$ order approximation')
    plt.plot(cal_zs, brewsters_approx, '-', label=r'better approximation')
    plt.legend()
    plt.xlabel(r'$\gamma^2$')
    plt.ylabel(r'$\theta_{pp}$')
    plt.show()

if __name__ == "__main__":
    main()