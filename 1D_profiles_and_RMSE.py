# -*- coding: utf-8 -*-
"""

This code is part of the supplementary materials of the Journal of Glaciology article titled:
 -----------------------------------------------------------------------------------------------------

'Firn densification in two dimensions:
modelling the collapse of snow caves and enhanced densification in ice-stream shear margins'

                            Arrizabalaga-Iriarte J, Lejonagoitia-Garmendia L, Hvidberg CS, Grinsted A, Rathmann NM

 ----------------------------------------------------------------------------------------------------

In this paper, we revisit the nonlinear-viscous firn rheology introduced by Gagliardini and Meyssonnier (1997)
that allows posing multi-dimensional firn densification problems subject to arbitrary stress and temperature fields.
In this sample code in particular, we revisit the calibration of the model by solving the 1D steady state densification
problem for different values of K for the six Greenlandic sites analyzed in this study.
We then compute the Root Mean Square Error (RMSE) with respect to the density measurements.

"""

import copy, code # code.interact(local=locals())
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
from dolfin import *
from dolfin_adjoint import * 
import moola
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

#control verbosity of solve function 
# https://fenicsproject.org/qa/810/how-to-disable-message-solving-linear-variational-problem/
set_log_level(21)

def get_rho_data(H,site):
 
    obs= np.loadtxt(r'./../icecores/density-raw/dens_%s.txt'%(site))
    
    z_obs = H - (obs[:,0] - 0*obs[0,0])
    I_obs = np.argsort(z_obs) # must be sorted for interpolation below
    z_obs = z_obs[I_obs]  
    
    rho_obs = obs[I_obs,1] # kg/m^3

    I_rm = np.argwhere(z_obs < 0)
    rho_obs = rho_obs[z_obs >= 0]
    z_obs   = z_obs[z_obs >= 0]
        
    return rho_obs, z_obs

def smooth_and_extrapolate_raw_data(z_raw,rho_raw,H):
    
    #-------------------------------------SMOOTH RAW DATA TO FIT--------------------------------------
    
    zdata=np.copy(z_raw)
    rhodata=np.copy(rho_raw)

    #---------------ignore first 2.5 meters of data to avoid any surface phenomena

    ignoresurface=H-2.5
    
    rhodata=rhodata[zdata<=ignoresurface]
    zdata=zdata[zdata<=ignoresurface]
    
    #---------------smooth the raw data

    Hmodel=180 #200
    Nnodes_model=6 #2.5
    dx_model=Hmodel/Nnodes_model

    dx_rawdata= zdata[20]-zdata[18] #watch out, it goes in steps of two
    print('data',dx_rawdata,'   dx_model',dx_model)
    
    window_length=int(dx_model/dx_rawdata)
    print('window_length',window_length)
    
    
    polyorder=3

    if window_length>polyorder:

        rhodata_smooth=savgol_filter(rhodata,window_length,polyorder=polyorder)
        rhodata=rhodata_smooth
    
    #---------------save smoothed version
    z_smooth=np.copy(zdata)
    rho_smooth=np.copy(rhodata)
    
    #---------------EXTRAPOLATION to the surface density--- max ~5m (grip)
    f_extrapolate_surface=scipy.interpolate.interp1d(zdata[::2],rhodata[::2],kind='cubic',fill_value='extrapolate')
    rho_snow_ext = f_extrapolate_surface(H) #In the code the surface is z=H    
    
    return z_smooth, rho_smooth, rho_snow_ext


def acc_from_iceeqyr_to_snoweqs(acc,rho_snow,verbose=True):

    """Change accumulation units from ice equivalent meters per year to snow equivalent meters per second"""
    
    accum_watereqyr = acc # water equiv. accum, m/yr
    accum_iceeqyr = 1000/rho_ice * accum_watereqyr
    accum_iceeqs = accum_iceeqyr/SecPerYear # ice accum, m/s 
    acc_rate = rho_ice/rho_snow * accum_iceeqs

    if verbose:
        print(f'{acc=} m of water eq per year')
        print(f'{acc_rate*yr2s=} m of snow eq per year')
        print(f'{acc_rate=} m of snow eq per second')
    
    return acc_rate


def get_ab(rho,rho_ice,phi_snow,ab_phi_lim,nglen,K):

    """Evaluate the coefficient functions as introduced by Zwinger et al. 2007 """
    
    rhoh = rho/Constant(rho_ice) # normalized density (rho hat)
    rhohsnow, rhohcrit = Constant(phi_snow), Constant(ab_phi_lim)

    f_a0 = lambda rhoh: (1+2/3*(1-rhoh))*rhoh**(-2*nglen/(nglen+1))
    f_b0 = lambda rhoh: 3/4*((1/nglen*(1-rhoh)**(1/nglen))/(1-(1-rhoh)**(1/nglen)))**(2*nglen/(nglen+1))

    gamma_mu = 20*1
    mu = lambda rhoh: 1/(1+exp(-gamma_mu*(rhohcrit*1-rhoh))) # step function (approximated by logistics function)

    gamma_a = lambda k: (ln(k)-ln(f_a0(rhohcrit)))/(rhohcrit-rhohsnow)
    gamma_b = lambda k: (ln(k)-ln(f_b0(rhohcrit)))/(rhohcrit-rhohsnow)
    f_a1 = lambda rhoh,k: k*exp(-gamma_a(k)*(rhoh-rhohsnow))
    f_b1 = lambda rhoh,k: k*exp(-gamma_b(k)*(rhoh-rhohsnow))

    f_a = lambda rhoh,k: f_a0(rhoh) + mu(rhoh)*f_a1(rhoh,k)
    f_b = lambda rhoh,k: f_b0(rhoh) + mu(rhoh)*f_b1(rhoh,k)
    
    a = f_a(rhoh,K)
    b = f_b(rhoh,K)
    
    return a, b


def get_sigma(v,a,b,Aglen,nglen):

    """Evaluate the (1D) stress tensor (sigma) according to the GM97 rheology"""
    
    eps_dot=v.dx(0)                              #sym(grad(v))
    J1=eps_dot                                   #tr(eps(v))
    J2=eps_dot*eps_dot                           #inner(eps_dot,eps_dot)
    eps_E2=1/a*(J2-J1**2/3) + (3/2)*(1/b)*J1**2
    viscosity = (1/2)**((1-nglen)/(2*nglen)) * Aglen**(-1/nglen) * (eps_E2)**((1-nglen)/(2*nglen))    
    sigma = viscosity * (1/a*(eps_dot-(J1/3))+(3/2)*(1/b)*J1)
    
    return sigma


#Define boundaries of the domain
class bottom_boundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0],0)

class surface_boundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0],H)


def solve_forward_problem(H,Hres,acc_rate,K,n,Aglen,rho_snow,rho_ice=910,phi_snow=0.4,ab_phi_lim=0.81): #steady state da hau

    """Solve the steady state 1D forward problem for a particular value of K and the corresponding site parameters"""

    #-------------------------------------------MESH-----------------------------------------------------#

    mesh = IntervalMesh(Hres-1, 0, H) # Hres nodes on x=[0,H]

    #-----------------------------------MIXED FUNCTION SPACE---------------------------------------------#

    Uele = FiniteElement("CG", mesh.ufl_cell(), 1) #Density
    Vele = FiniteElement("CG", mesh.ufl_cell(), 2) #Velocity
    U, V = FunctionSpace(mesh, Uele), FunctionSpace(mesh, Vele) 
    MixedEle = MixedElement([Uele, Vele]) 
    W = FunctionSpace(mesh, MixedEle)

    (wr,wv) = TestFunctions(W) #weight functions (Galerkin method)
    w = Function(W) #container for solution (rho,v)
    (rho,v) = split(w)

    #Get nodal coordinates
    z_rho = U.tabulate_dof_coordinates()[:,0] #velocity node coordinates
    z_v  = V.tabulate_dof_coordinates()[:,0]  #density node coordinates
    I_rho, I_v = np.argsort(z_rho), np.argsort(z_v)
    z_rho, z_v = z_rho[I_rho], z_v[I_v] #sorted list of coordinates

    #--------------------------------------BOUNDARY SUBDOMAINS--------------------------------------------#

    boundary_subdomains = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundary_subdomains.set_all(0)

    bottom=bottom_boundary()
    bottom.mark(boundary_subdomains, 1)
    surface=surface_boundary()
    surface.mark(boundary_subdomains, 2)

    #--------------------------------------BOUNDARY CONDITIONS--------------------------------------------#
    rho_surf=rho_snow

    v_surf=-acc_rate
    v_bot=-acc_rate*(rho_surf/rho_ice)

    #-----------------------------------TOP
    bc_rho_s=DirichletBC(W.sub(0),rho_surf,boundary_subdomains,2) #Density at the surface
    bc_v_s=DirichletBC(W.sub(1),v_surf,boundary_subdomains,2) #Velocity at the surface

    #-----------------------------------BOTTOM
    bc_v_b=DirichletBC(W.sub(1),v_bot,boundary_subdomains,1) #Velocity at the bottom

    bcs=[bc_rho_s,bc_v_s,bc_v_b]      

    #----------------------------------------INITIAL STATE--------------------------------------------#

    assigner = FunctionAssigner(W, [U, V]) # used to assign values into components of mixed function space

    r_init = Expression("rho0  - (rho0-rhoi)*pow((H-x[0])/H,1/3.0)", H=H, rhoi=rho_ice, rho0=rho_snow, degree=2)
    u_init = Expression("uztop - (uztop-uzbot)*pow((H-x[0])/H,1/3.0)", H=H, uztop=v_surf, uzbot=v_bot,  degree=2)

    rho_init, v_init = interpolate(r_init, U), interpolate(u_init, V)
    w_init = Function(W) # container for solution (uz,rho)
    assigner.assign(w_init, [rho_init,v_init]) # assign initial guesses to w_init #just for clarity
    w.assign(w_init) #assign the initial guesses stored in w_init to our w

    #--------------------------------------SOLVE FEM PROBLEM----------------------------------------------#

    #---------------------------STOKES

    #Get Cs, a, b, and sigma
    Aglen = interpolate(Constant(Aglen), U)
    a_,b_=get_ab(rho,rho_ice,phi_snow,ab_phi_lim,n,K)
    sigma=get_sigma(v,a_,b_,Aglen,n)

    a_sig = sigma * wv.dx(0) * dx # viscous stress divergence
    L_sig = rho*Constant(g)*wv*dx # Body force (gravity)

    #---------------------------DENSITY

    a_rho = (rho*v).dx(0) * wr * dx # d(rho)/dt + div(rho*u) = 0, for d/dt = 0 
    L_rho = 0

    #---------------------------NON-LINEAR WEAK FORM

    F  = a_sig - L_sig # stokes problem 
    F += a_rho - L_rho # density problem

    #---------------------------SOLVE MIXED PROBLEM

    tol, relax, maxiter = 1e-2, 0.35, 50
    solparams = {"newton_solver":  {"relaxation_parameter":relax, "relative_tolerance": tol, "maximum_iterations":maxiter} }
    solve(F==0, w, bcs, solver_parameters=solparams)
    rho_sol, v_sol = w.split()
    v_sol, rho_sol = project(v_sol, V), project(rho_sol, U) # ... but this does
    v, rho = v_sol.vector()[I_v], rho_sol.vector()[I_rho] # solution on nodes, correctly sorted for plotting 
    
    return rho, z_rho

#Define parameters

#Unit conversion
SecPerYear = 31556926;
s2yr = 1/SecPerYear
ms2myr = 31556926
ms2kyr = ms2myr/1000

#Physical constants
g = -9.82  # gravitational accel
rho_ice = 910 # density of glacier ice
ab_phi_lim = 0.81 # critical relative density (\hat{\rho}) at which coefficient function changes from a0 to a1, and b0 to b1
phi_snow = 0.4#rho_snow/rho_ice # relative density of snow, used for calibrating a(rhoh_snow) = b(rhoh_snow) = k 
n = 3 # flow law exponent
A0,Q,R = 3.985e-13, 60e3, 8.314 # Canonical values, same as Zwinger et al. (2007)
T0 = 273.15

#Site related parameters
H0=180
Hres=30#30 #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<tune resolution

sites=['dye3','grip','neem','ngrip','site_2','siteA_crete']

Ks=np.linspace(10,3000,301)
Hs={'dye3':H0, 'grip':H0, 'neem':H0-15, 'ngrip':H0-10, 'site_2':H0, 'siteA_crete':H0}
acc_sites={'dye3':0.50,    'grip':0.21,   'neem':0.20,   'ngrip':0.175,   'site_2':0.36,     'siteA_crete':0.282}
Ts={'dye3':-21, 'grip':-31.7, 'neem':-28.8, 'ngrip':-31.5, 'site_2':-25, 'siteA_crete':-29.5}


#initalize RMSE arrays
SSE=np.zeros((len(sites),len(Ks)))
RMSE=np.zeros((len(sites),len(Ks)))
SSE_08=np.zeros((len(sites),len(Ks)))
SSEpercent_08=np.zeros((len(sites),len(Ks)))
RMSE_08=np.zeros((len(sites),len(Ks)))
RMSEpercent_08=np.zeros((len(sites),len(Ks)))

for i in range(len(sites)):
    
    site=sites[i]
    print(f'\n>>>>>>>>>>>>>>>>>>>>>>>>>>>> computing {site=}<<<<<<<<<<<<\n')
    
    H=Hs[site]
    T=Ts[site]
    Aglen=A0*np.exp(-Q/(R*(T+T0)))
    
    rho_obs_raw, z_obs_raw=get_rho_data(H,site)

    #-------------------------------------SMOOTH RAW DATA TO FIT--------------------------------------
    
    z_obs_smooth,rho_obs_smooth,rho_surf=smooth_and_extrapolate_raw_data(z_obs_raw,rho_obs_raw,H)
    
    #---------------CALCULATE acc according to the surface density
    acc_rate=acc_from_iceeqyr_to_snoweqs(acc_sites[site],rho_surf)
    
    #---------------------------------PERFORM THE FIT WITH THE SMOOTHED DATA--------------------------------------    
    
    for j in range(len(Ks)):
        
        K=Ks[j]
        print(f'------------- {K=}/{np.max(Ks)}')

        #--------------------------SOLVE FORWARD PROBLEM
        
        rho_model,z_rho_model=solve_forward_problem(H,Hres,acc_rate,Ks[j],n,Aglen,rho_surf,rho_ice=916)
        f_rho_interp=scipy.interpolate.interp1d(z_rho_model, rho_model, kind='linear')
        rho_model_interp=f_rho_interp(z_obs_smooth)

        #------------------------------------------------------COMPUTE RMSEs
        #-------------------------------Hole column RMSE

        SSE[i,j]=np.sum((rho_obs_smooth-rho_model_interp)**2)
        RMSE[i,j]=np.sqrt(SSE[i,j]/len(rho_obs_smooth))

        #----------------------------------------Relative densities lower than 0.8
        rho_obs_smooth_08=rho_obs_smooth[rho_obs_smooth/910<=0.8]
        rho_model_interp_08=rho_model_interp[rho_obs_smooth/910<=0.8]
        
        SSE_08[i,j]=np.sum((rho_obs_smooth_08-rho_model_interp_08)**2)
        SSEpercent_08[i,j]=np.sum((100*(rho_obs_smooth_08-rho_model_interp_08)/rho_obs_smooth_08)**2)
        RMSE_08[i,j]=np.sqrt(SSE_08[i,j]/len(rho_obs_smooth_08))
        RMSEpercent_08[i,j]=np.sqrt(SSEpercent_08[i,j]/len(rho_obs_smooth_08))



np.save('RMSE_site_by_site.npy',RMSE)
np.save('RMSE_site_by_site_08.npy',RMSE_08)
np.save('RMSEpercent_site_by_site_08.npy',RMSEpercent_08)
