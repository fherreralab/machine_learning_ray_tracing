import os
import argparse

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm

import numpy as np

import jax
import jax.numpy as jnp

from ray_tracing import al_polarization
from utils import get_exp_parameters

def _grid(no_exp,ne_exp,material_):
    ni = 1.
    a_c=29.2
    a_i=1E-6
    go=0
    ge=-1.92*go

    thick= 3E-3
    lamda= 853*10**-9
    sens=1E-8
    vnorm=np.zeros((3,1))
    vnorm=vnorm.at[2].set(1)

    def f_MMuller(no,ne):
        M,C = al_polarization(no,ne,go,ge,vnorm,thick,a_c,a_i,ni,lamda)
        return C    

    def f_loss_single(params,MMuller_exp):
        no,ne = params
        MMuller_pred = f_MMuller(no,ne)
        return jnp.linalg.norm(MMuller_exp - MMuller_pred)

    def f_loss_stoch(params,MMuller_exp,key):
        std_no = 0.01
        std_ne = 0.01
        mu_no,mu_ne = params
        eps = jax.random.normal(key,shape=(10,))
        no_ = eps*std_no + mu_no
        _,key = jax.random.split(key)
        eps = jax.random.normal(key,shape=(10,))
        ne_ = eps*std_ne + mu_ne
        _,key = jax.random.split(key)

        loss_ = jnp.zeros(1)    
        for no,ne in zip(no_,ne_):
            params = (no*jnp.ones(1),ne*jnp.ones(1))
            loss = f_loss_single(params,MMuller_exp)
            loss_ = jnp.append(loss_,loss)
            # MMuller_pred = f_MMuller(no,ne)
        return jnp.mean(loss_[1:]),key,jnp.array(loss_[1:])

    MMuller_exp = f_MMuller(no_exp,ne_exp)   
 

    no_ = no_exp + jnp.linspace(-0.1,0.1,50)
    ne_ = ne_exp + jnp.linspace(-0.1,0.1,50)

    X, Y = jnp.meshgrid(no_, ne_)
    z_=[]
    Brew=[]
    
    key = jax.random.PRNGKey(0)
    loss_sample = jnp.ones(10)
    for x,y in zip(X.ravel(),Y.ravel()):
        params = (x*jnp.ones(1),y*jnp.ones(1))

        z_temp = f_MMuller(x*jnp.ones(1),y*jnp.ones(1)) # OPD
        z_.append(z_temp)


    z_ = jnp.array(z_)
    Z = jnp.reshape(z_,jnp.shape(X))
    D = {'X':X,
        'Y': Y,
        'Z': Z,
        'no': no_exp,
        'ne': ne_exp,
        'material': material_,
        }
    return D

def main_surface_data(no_exp,ne_exp,material_):

    D = _grid(no_exp,ne_exp,material_)
    jnp.save('{}_OPD_n_50_red.npy'.format(material_),D,allow_pickle=True)


def main_data(material):
    print(material)
    # "BBO"
    if material == 'BBO':
        no_exp =  1.6589*jnp.ones(1)#1.6626
        ne_exp = 1.5446*jnp.ones(1)#1.5470
        main_surface_data(no_exp,ne_exp,'BBO')
    elif material == 'Calcite':
        # "calcite" "CaCO_3"
        no_exp = 1.6473*jnp.ones(1)
        ne_exp =  1.4813*jnp.ones(1)
        main_surface_data(no_exp,ne_exp,'Calcite')
    elif material == 'KDP':
        # "KDP"
        no_exp = 1.4999*jnp.ones(1)
        ne_exp = 1.4625*jnp.ones(1)
        main_surface_data(no_exp,ne_exp,'KDP')   
    elif material == 'Quartz':
        # "Quartz"
        no_exp = 1.5373*jnp.ones(1)
        ne_exp = 1.5462*jnp.ones(1)
        main_surface_data(no_exp,ne_exp,'Quartz')

def _surface_plot(material):
    def get_data(materia='BBO',type_='MMuller'):
        D = jnp.load('{}_{}_n_50_red.npy'.format(material,type_),allow_pickle=True)


        X = D.item()['X']
        Y = D.item()['Y']
        Z = D.item()['Z']
        if type_ == 'OPD':
            Z = jnp.abs(Z - 188.2E-6)
        no = D.item()['no']
        ne = D.item()['ne']
        return X,Y,Z,no,ne
        
    X,Y,Z_MM,no,ne = get_data(material,'MMuller')
    Z = Z_MM



    norm = cm.colors.Normalize(vmax=Z.max(), vmin=Z.min())

    fig, ax = plt.subplots(1,)
    cset1 = ax.contourf(
        X, Y, Z, 40,
        norm=norm)

    ax.scatter(no,ne,c='w',marker='x',s=25)

    cbar = plt.colorbar(cset1)
    cbar.set_label(r'$\mathcal{L}_{MMuller}$',fontsize=15) 


    ax.set_xlim(jnp.min(X), jnp.max(X))
    ax.set_ylim(jnp.min(Y), jnp.max(Y))

    ax.set_xlabel('no',fontsize=15)
    ax.set_ylabel('ne',fontsize=15)


    plt.title(material)
    plt.savefig('fig_{}_MMuller.png'.format(material),dpi=150)

def _surface_plot_opt(material):
    D = np.load('{}_MMuller_n_50_red.npy'.format(material),allow_pickle=True)
    X = D.item()['X']
    Y = D.item()['Y']
    Z = D.item()['Z']
    no = D.item()['no']
    ne = D.item()['ne']

    D_opt = np.load('results_opt_{}.npy'.format(material),allow_pickle=True)
    print(D_opt)
    params = D_opt.item()['params']

    norm = cm.colors.Normalize(vmax=Z.max(), vmin=Z.min())

    fig, ax = plt.subplots(1,)
    cset1 = ax.contourf(
        X, Y, Z, 40,
        norm=norm)

    ax.scatter(no,ne,c='w',marker='x',s=25)
    ax.scatter(params[:,0],params[:,1],c='w',marker='o',s=25)

    cbar = plt.colorbar(cset1)
    cbar.set_label(r'$\mathcal{L}_{MMuller}$',fontsize=15) 

    ax.set_xlim(jnp.min(X), jnp.max(X))
    ax.set_ylim(jnp.min(Y), jnp.max(Y))

    ax.set_xlabel('no',fontsize=15)
    ax.set_ylabel('ne',fontsize=15)


    plt.title(material)
    plt.savefig('fig_{}_MMuller_opt.png'.format(material),dpi=150)

def _plot_opt_params(material,opt='all'):
    r_dir = 'Results_opt_{}/'.format(material)
    arr = os.listdir(r_dir)

    w_opt = np.ones((1,2))    
    l_opt = np.ones(1)
    for file in arr:
        if material in file and '.npy' in file and opt in file:
            D = np.load(r_dir + file,allow_pickle=True)
            l = D.item()['loss']
            w = D.item()['params']
            li = np.argmin(l[:,0])
            w_tmp = w[li]
            w_opt = np.vstack((w_opt,w_tmp))
            l_opt = np.append(l_opt,l[li,0])
    
    l_opt = l_opt[1:].ravel()
    w_opt = w_opt[1:]

    def scatter_opt_params(l_opt,w_opt):
        fig, ax = plt.subplots(1,)
        cbar = ax.scatter(w_opt[:,0],w_opt[:,1],c=l_opt,marker='o',s=15,vmin=np.min(l_opt),vmax=np.max(l_opt))
        cbar = plt.colorbar(cbar)
        if opt == 'all':
            cbar.set_label(r'$\mathcal{L}_{MM} + \mathcal{L}_{OPD} + \mathcal{L}_{Brew}$',fontsize=15)
        elif opt == 'mm_opd':
            cbar.set_label(r'$\mathcal{L}_{MM} + \mathcal{L}_{OPD}$',fontsize=15)
        elif opt == 'mm_brew':
            cbar.set_label(r'$\mathcal{L}_{MM} + \mathcal{L}_{Brew}$',fontsize=15)
        no,ne = get_exp_parameters(material)
        ax.scatter(no,ne,color='red',marker='*',s=70,label='Exact')
        ax.set_xlabel(r'$n_{o}$',fontsize=15)
        ax.set_ylabel(r'$n_{e}$',fontsize=15)

        plt.title('{}'.format(material),fontsize=12)
        fig.tight_layout()
        plt.savefig('Figures_new/fig_{}_{}_parameters.png'.format(material,opt),dpi=150)
    
    def histogram_opt_params(l_opt,w_opt):
        no,ne = get_exp_parameters(material)
        diff_no = np.abs(w_opt[:,0] - no)
        diff_ne = np.abs(w_opt[:,1] - ne)


        labels_ = ['no','ne']
        num_bins = 10
        fig, ax = plt.subplots()


        ax.hist(diff_no, num_bins, density=True, histtype="stepfilled", alpha=0.8, label=r'$n_{o}$')
        ax.hist(diff_ne, num_bins, density=True, histtype="stepfilled", alpha=0.8, label=r'$n_{e}$')

        ax.legend(prop={'size': 12})
        ax.set_title('{}'.format(material),fontsize=15)
        ax.set_xlabel(r'$|n_{i} - \hat{n}_{i}|$',fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)

        fig.tight_layout()
        plt.savefig('Figures_new/fig_{}_{}_hist.png'.format(material,opt),dpi=150)

    def histogram_opt_params_subplots(l_opt,w_opt):
        no,ne = get_exp_parameters(material)
        diff_no = np.abs(w_opt[:,0] - no)
        diff_ne = np.abs(w_opt[:,1] - ne)




        if opt == 'all':
            l = r'$\mathcal{L}_{MM} + \mathcal{L}_{OPD} + \mathcal{L}_{Brew}$'
        elif opt == 'mm_opd':
            l  = r'$\mathcal{L}_{MM} + \mathcal{L}_{OPD}$'
        elif opt == 'mm_brew':
            l = r'$\mathcal{L}_{MM} + \mathcal{L}_{Brew}$'

        labels_ = ['no','ne']
        num_bins = 10


        fig, (ax0,ax1) = plt.subplots(2)


        ax0.hist(diff_no, num_bins, density=True, histtype="stepfilled", alpha=0.8, color="tab:blue")

        ax0.set_title(material)
        ax0.set_xlabel(r'$|\Delta n_{o}|$',fontsize=15)


        ax1.hist(diff_ne, num_bins, density=True, histtype="stepfilled", alpha=0.8, color="tab:orange")
        ax1.set_xlabel(r'$|\Delta n_{e}|$',fontsize=15)

        fig.tight_layout()
        plt.savefig('Figures_new/fig_{}_{}_hist.png'.format(material,opt),dpi=150)
    
    # histogram_opt_params(l_opt,w_opt)
    histogram_opt_params_subplots(l_opt,w_opt)
    scatter_opt_params(l_opt,w_opt)

def _learning_curve(material,opt):
    r_dir = 'Results_opt_lr_5em3_run0/'
    arr = os.listdir(r_dir)

    w_opt = np.ones((1,2))    
    l_curve = np.ones(1)
    if0 = 0
    for file in arr:
        if material in file and '.npy' in file and opt in file:
            D = np.load(r_dir + file,allow_pickle=True)
            l = D.item()['loss']
            w = D.item()['params']
            print(if0,l.shape)


    print(l_curve.shape)

def main():

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--mat',  type=str, default='BBO',
                    help='material')

    args = parser.parse_args()
    material = args.mat
    main_data(material)

if __name__== "__main__":
    main()
    _surface_plot('BBO')
    _surface_plot_opt('BBO')

    _plot_opt_params('BBO','mm_brew')
    _plot_opt_params('BBO','mm_opd')
    _plot_opt_params('BBO','all')

    materials_ = ['BBO','Calcite','KDP','Quartz']
    for m in materials_:
        _plot_opt_params(m,'mm_brew')
        _plot_opt_params(m,'mm_opd')
        _plot_opt_params(m,'all')

    _learning_curve('BBO','all') 



    materials_ = ['BBO','Calcite','KDP','Quartz']
    for m in materials_:
        _surface_plot(m)
    # main_data()
    print(jax.__version__)