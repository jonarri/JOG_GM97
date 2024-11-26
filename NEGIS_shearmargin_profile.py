#!/usr/bin/env python
# coding: utf-8
"""

This code is part of the supplementary materials of the Journal of Glaciology article titled:
 -----------------------------------------------------------------------------------------------------

'Firn densification in two dimensions:
modelling the collapse of snow caves and enhanced densification in ice-stream shear margins'

                            Arrizabalaga-Iriarte J, Lejonagoitia-Garmendia L, Hvidberg CS, Grinsted A, Rathmann NM

 ----------------------------------------------------------------------------------------------------

In this paper, we revisit the nonlinear-viscous firn rheology introduced by Gagliardini and Meyssonnier (1997)
that allows posing multi-dimensional firn densification problems subject to arbitrary stress and temperature fields.
In this sample code in particular, we compute the densification (and, thus, surface elevation and Bubble Close-Off
depth) predictions for a transect across the North-East Greenland Ice Stream (NEGIS) to see if the rheology can explain
the increased densification rate and varying BCO depth observed on the shear margins.

"""


import sys
import time
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from fenics import *
from dolfin import *
from scipy import interpolate as interpolatescipy
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter


#control verbosity of solve function 
#https://fenicsproject.org/qa/810/how-to-disable-message-solving-linear-variational-problem/
set_log_level(21)


# ###-------------------------- FUNCTIONS:


def acc_from_iceeqyr_to_snoweqs(acc,rho_obs):

    accum_iceeqs = acc/SecPerYear # ice accum, m/s 
    rho_snow = rho_obs
    acc_rate = rho_ice/rho_snow * accum_iceeqs
    
    print('adot [metre snow/yr] = %e'%(acc_rate*SecPerYear))
    
    return acc_rate



def get_ab(rho,rho_ice,phi_snow,ab_phi_lim,nglen,K):
    
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


def get_sigma(v,a,b,Aglen,nglen,epsl,epss):
    
    eps_dot=sym(grad(v))                     
    J1_e=tr(eps_dot)# + epsl  
    J1_s=tr(eps_dot)
    J2=inner(eps_dot,eps_dot)+2*epss**2#+epsl**2     
    eps_E2=1/a*(J2-J1_e**2/3) + (3/2)*(1/b)*J1_e**2
    viscosity = (1/2)**((1-nglen)/(2*nglen)) * Aglen**(-1/nglen) * (eps_E2)**((1-nglen)/(2*nglen))    
    sigma = viscosity * (1/a*(eps_dot-(J1_s/3)*Identity(2))+(3/2)*(1/b)*J1_s*Identity(2))
    
    return sigma



def get_Aglen(U,heated_margins=False):

    x0Riverman=10.48*1e3

    #shear margins defined as eps_xy >abs(strain rate max peak/4)~3.5yr-1
    x1= - x0Riverman +13.064*1e3 + 5000 #position (from 0) + 5km left side buffer
    x2= - x0Riverman +17.720*1e3 + 5000
    x3= - x0Riverman +34.315*1e3 + 5000
    x4= - x0Riverman +42.279*1e3 + 5000

    T=-28+273.15
    #----------------to impose the hotter margins
    if heated_margins:
        T_shear=T+6
    else:
        T_shear = T

    #------------to properly order the nodes before setting the values--------------------------
    # self.smesh = IntervalMesh(self.RES_L-1, 0, L)
    f_xs = mesh.coordinates()[:, 0] # x coords of mesh *vertices*
    f_ys = mesh.coordinates()[0, :] # y coords of mesh *vertices*

    Ts=Function(U)

    # func spaces
    # self.S  = FunctionSpace(self.smesh, "CG", 1, constrained_domain=pbc_1D)
    scoords = U.tabulate_dof_coordinates()
    xs_dof = scoords[:,0] # x coords of func space NODES
    s_numdofs = len(xs_dof)
    ISx = np.argsort(xs_dof)

    #set temperature values
    for ii in range(s_numdofs):
        xii = xs_dof[ii] # x coord of DOF (node)

        if xii>=x1 and xii<=x2:
            Ts.vector()[ii]=T_shear

        elif xii>=x3 and xii<=x4:
            Ts.vector()[ii]=T_shear

        else:
            Ts.vector()[ii]=T

    Aglen=Function(U)
    Aglen=A0*exp(-Q/(R*Ts))
    Aglen_p=project(Aglen,U)

    return Aglen_p


#----------Surface evolution
class Surface:
    
    def __init__(self, f_z0):
        
        # init profile
        self.RES_L = Lres #<------
        self.xvec = np.linspace(0,L, self.RES_L)
        self.zvec = f_z0(self.xvec) # init profile
        
        # mesh 
        self.smesh = IntervalMesh(self.RES_L-1, 0, L)
        self.xs = self.smesh.coordinates()[:, 0] # x coords of mesh *vertices* 
        
        # func spaces 
        self.S  = FunctionSpace(self.smesh, "CG", 1)#, constrained_domain=pbc_1D)
        scoords = self.S.tabulate_dof_coordinates()
        self.xs_dof = scoords[:,0] # x coords of func space NODES
        self.s_numdofs = len(self.xs_dof)
        self.IS = np.argsort(self.xs_dof)    

        self.deltaH = self.zvec #will be changed after evolving surface
        self.usz = Function(self.S)
        
        self.accumulation=True
        if self.accumulation:
            
            self.accf=Function(self.S)
            
            for ii in range(self.s_numdofs):
            
                xii = self.xs_dof[ii] # x coord of DOF (node)
                self.accf.vector()[ii]=acc_rate
                
            acc_proj=project(self.accf,self.S)
        
        
    def _extendPeriodicVector(self, vec):
        # make the last point the first value too, which is excluded in a Vector()
        return np.append([vec[-1]],vec) 
        
    def evolve(self, u, dt):
        
        # "u" is the full 2D velocity field solution
        u.set_allow_extrapolation(True) 
        
        zvec_intp = interpolatescipy.interp1d(self.xvec, self.zvec, kind='linear') 
        
        # x and z vel. values at surface
        usx_arr, usz_arr = np.zeros(self.s_numdofs), np.zeros(self.s_numdofs)
        
        
        for ii in np.arange(self.s_numdofs): # loop over surface mesh DOFs (nodes)
            xii = self.xs_dof[ii] # x coord of DOF (node)
            zii = zvec_intp(xii) # the surface height of DOF (node)
            usx_arr[ii], usz_arr[ii] = u(xii, zii) # surface x and z velocity components
        
        # Save the sfc vel. values in "S" function spaces for solving the sfc evo problem below
        usx, usz = Function(self.S), Function(self.S) 
        usx.vector()[:] = usx_arr[:]
        usz.vector()[:] = usz_arr[:]
        
        # Function spaces
        v       = TestFunction(self.S) # weight funcs
        strial  = TrialFunction(self.S) # new (unknown) solution
        s0      = Function(self.S) # prev. solution
        s0.vector()[self.IS] = np.copy(self.zvec)#[1:]) # -1 because of periodic domain
        
        # Solve surface equation
        dt_ = Constant(dt)
        a  = strial*v*dx + dt_*usx*strial.dx(0)*v*dx
        L  = s0*v*dx + dt_*(usz+project(self.accf,self.S))*v*dx
        s = Function(self.S)
        solve(a == L, s, [])
     
        # Update surface profile
        self.deltaH = s.vector()[self.IS] - self.zvec
        self.zvec = s.vector()[self.IS]
        self.usz=usz



#Create mesh

def createMesh(xvec, bottomsurf, topsurf, Nx, Nz):
    
    hl = Nx-1 # Number of horizontal layers
    vl = Nz-1 # Number of vertical layers
    
    # generate mesh on unitsquare
    mesh = RectangleMesh(Point(0.0, 0.0), Point(1.0, 1.0), hl, vl)
    
    # Extract x and y coordinates from mesh
    x = mesh.coordinates()[:, 0]
    y = mesh.coordinates()[:, 1]
    
    x0 = min(xvec)
    x1 = max(xvec)
    
    # Map coordinates on unit square to the computational domain
    xnew = x0 + x*(x1-x0)
    zs = np.interp(xnew, xvec, topsurf)
    zb = np.interp(xnew, xvec, bottomsurf)
    ynew = zb + y*(zs-zb)
    
    xynewcoor = np.array([xnew, ynew]).transpose()
    
    mesh.coordinates()[:] = xynewcoor
    
    return mesh


# Define boundaries of the domain

class bottom_boundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

class top_boundary(SubDomain):
    def inside(self,x,on_boundary):
        return on_boundary and x[1]>=50 

class right_boundary(SubDomain):
    def inside(self,x,on_boundary):
        return on_boundary and near(x[0],L)

class left_boundary(SubDomain):
    def inside(self,x,on_boundary):
        return on_boundary and near(x[0],0)  



def convert_to_np(U,ufl_array):
    
    zcoord=U.tabulate_dof_coordinates()[:,0]   #save coordinates, not in order.
    np_array=ufl_array.vector()                #save values of the variable we are interested in, not in order.
    I=np.argsort(zcoord)                       #save index to sort depending on the value of z. 
    sorted_coors=zcoord[I]                     #save coordinates, in order. 
    sorted_np=np_array[I]                      #save values of the variable we are interested in, in order.

    return sorted_coors,sorted_np


# ### SIMULATION PARAMETERS

    
outfolder='results/'    
    
K=1000#2000
heated_margins=True


if heated_margins:
    margintag='marginT'
else:
    margintag='isoT'
    
print(f'-------{K=}    {margintag=}')

n=3

deltaL=5000                        #Buffer on each side to avoid numerical artifacts 
H,L=200,37440 +2*deltaL            #Depth, Length
Hres,Lres=150,300                  #Nodal resolution 

rho_surf=275                       #Density of snow at the surface (kg/m^3)
rho_ice= 910                       #Density of pure ice (kg/m^3)
rho_ice_softened= rho_ice - 0.2    #we use this to avoid numerical errors when reaching the density of pure ice

phi_snow=0.4                       #relative density of snow
ab_phi_lim=0.81                    #relative density at which a=b


day2s=24*60*60
yr2s=365.25*day2s
SecPerYear=yr2s

acc_meas=0.18                                           #Measured acc in m of ice equivalent per year
acc_rate=acc_from_iceeqyr_to_snoweqs(acc_meas,rho_surf) #Unit conversion from ice equivalent per year to snow equivalent per second

print(f'{acc_rate=} m_snow_eq/s {acc_rate*yr2s=} m_snow_eq/yr')

acc_rate= 0.408/yr2s #karlson 2020
print(f'{acc_rate=} m_snow_eq/s {acc_rate*yr2s=} m_snow_eq/yr')

g=Constant((0.0,-9.82)) 

dtf=0.25/0.2 #dt prefactor
dt=dtf*yr2s #lenght of the timestep in seconds



print(f'>>>>>>>>>>{K=}  {heated_margins=}\n {dtf=} ')
print(f'-----------------------------> dt={dt/yr2s} yrs----')

# ###------------------------------------------- ADDITIONAL STRAIN RATE VALUES

#Load strain rate values (units: yr^-1)
epsyy=np.load("NEGIS_epsyy.npy")
epsxy=np.load("NEGIS_epsxy.npy")


#Modify epsyy and epsxy vectors to add a buffer to both sides to avoid numerical instabilities
eps_dens= len(epsyy)/(L-2*deltaL) #number of measurements per lenght unit

Neps_deltaL=int(deltaL*eps_dens)  #lenght of buffer
extension=np.zeros(Neps_deltaL)

epsyy= np.concatenate((extension,epsyy,extension)) #create array with buffer in both sides 
epsxy= np.concatenate((extension,epsxy,extension))


#------------------------------------------Make the strain rate smoother

window_length=5
polyorder=3

epsyy_smooth=savgol_filter(epsyy/yr2s,window_length,polyorder=polyorder)
epsxy_smooth=savgol_filter(epsxy/yr2s,window_length,polyorder=polyorder)

#Cross check plot
fig, ax = plt.subplots(figsize=(15,5))
plt.plot(epsyy,c='tab:blue',ls='--',label='epsyy')
plt.plot(epsxy,c='tab:orange',ls='--',label='epsxy')
plt.plot(epsyy_smooth*yr2s,c='tab:blue',label='epsyy smoothed')
plt.plot(epsxy_smooth*yr2s,c='tab:orange',label='epsxy smoothed')
plt.ylabel('smoothed strain rates $\ \ \ \  (yr^{\ -1})$')
plt.xlabel('X (m)')
plt.legend()
plt.savefig(outfolder+'Strainrates_smooth.png',dpi=150,format='png')


#---------------------------------Create a function for strain rates to be able to add them to the equations later
#Create mesh
mesh=RectangleMesh(Point(0,0),Point(L,H),Lres,Hres)

#Define function space for density
deg=2 #Polinomial degree
U=FunctionSpace(mesh, 'Lagrange', deg); # Polynomial function space of order "deg" (=1 is linear, ...)


#------------to properly order the nodes before setting the values--------------------------
f_xs = mesh.coordinates()[:, 0] # x coords of mesh *vertices* 
f_ys = mesh.coordinates()[0, :] # y coords of mesh *vertices* 


# func spaces 
scoords = U.tabulate_dof_coordinates()
xs_dof = scoords[:,0] # x coords of func space NODES
s_numdofs = len(xs_dof)
ISx = np.argsort(xs_dof)  

fepsyy_smooth = interp1d(np.linspace(0,L,len(epsyy_smooth)), epsyy_smooth, kind='cubic',fill_value='extrapolate')
fepsxy_smooth = interp1d(np.linspace(0,L,len(epsxy_smooth)), epsxy_smooth, kind='cubic',fill_value='extrapolate')

strainl=Function(U)
strains=Function(U)

for ii in range(s_numdofs):
            
    xii = xs_dof[ii] # x coord of DOF (node)
    strainl.vector()[ii]=fepsyy_smooth(xii)
    strains.vector()[ii]=fepsxy_smooth(xii)
    
strainl_p=project(strainl,U)
strains_p=project(strains,U)



#----------------------------------------------------------------
# ### COUPLED PROBLEM: MAIN LOOP
#------------------------------------------INITIAL COORDINATES OF SURFACE AND BEDROCK
#Define horizontal coordinates
xvec= np.linspace(0, L, Lres)

#Define bedrock form (flat in this case)
zbot = np.zeros(len(xvec))

#Define initial surface (flat in this case)   
z0 = H/2
f_z0 = lambda x: H +x*0
sfc = Surface(f_z0)

#------------------------------------------INITIAL DENSITY PROFILE
rho_prev=Expression("rho0  - (rho0-rhoi)*pow((H-x[1])/H,1/3.0)", H=H, rhoi=rho_ice, rho0=rho_surf, degree=2)

#-----------------------------------------INITIALIZE VARIABLES
tstep=0

#-----------------------------------------DEFINE PARAMETERS FOR THE STEADY STATE
dHdt=100             #So that it enters the while
dHdt_tol=0.025 #0.010#0.015 #Change of height per unit time to consider that we have reached the steady state

#-----------------------------------------SOLVE PROBLEM
t0=time.time()
tprev=t0
dHdtprev=100


while dHdt>=dHdt_tol:

    q_degree = 3
    dx = dx(metadata={'quadrature_degree': q_degree})

    #-------------------------------------------MESH-----------------------------------------------------#
    
    mesh = createMesh(sfc.xvec, zbot, sfc.zvec, Lres, Hres)
    
    #--------------------------------------BOUNDARY SUBDOMAINS-------------------------------------------#
    
    #Give a number to each different boundary
    boundary_subdomains = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundary_subdomains.set_all(0)
        
    bottom=bottom_boundary()
    bottom.mark(boundary_subdomains, 2)
    top=top_boundary()
    top.mark(boundary_subdomains, 1)
    left=left_boundary()
    left.mark(boundary_subdomains, 3)
    right=right_boundary()
    right.mark(boundary_subdomains, 4)
    
    #--------------------------------------FUNCTION SPACE------------------------------------------------#
    
    #Define function space for density
    deg_rho=2#Polinomial degree

    U=FunctionSpace(mesh, 'Lagrange', deg_rho) # Polynomial function space of order "deg"

    rho=Function(U) # the unknown function
    wr=TestFunction(U)  # the weight function

    #Define function space for velocity
    deg_v=2
    V=VectorFunctionSpace(mesh, "CG", deg_v)

    v=Function(V) # the unknown function
    wv=TestFunction(V)  # the weight function
    
    #--------------------------------------BOUNDARY CONDITIONS--------------------------------------------#
    
    #-----------------------------------TOP
    bc_rho_s=DirichletBC(U,rho_surf,boundary_subdomains,1) #Density at the surface
    
    #-----------------------------------BOTTOM
    bc_v_b=DirichletBC(V,(0.0,-acc_rate*(rho_surf/rho_ice)),boundary_subdomains,2) #Velocity at the bottom
    
    #-----------------------------------LEFT
    bc_v_l=DirichletBC(V.sub(0),0.0,boundary_subdomains,3) #Velocity at the left boundary
    
    #-----------------------------------RIGHT
    bc_v_r=DirichletBC(V.sub(0),0.0,boundary_subdomains,4) #Velocity at the right boundary

    bcs_rho=[bc_rho_s] 
    bcs_v=[bc_v_b,bc_v_l,bc_v_r]
    
    #--------------------------------------INITIAL CONDITION--------------------------------------------#
    
    if tstep==0:
        
        r_init = Expression("rho0  - (rho0-rhoi)*pow((H-x[1])/H,1/3.0)", H=H, rhoi=rho_ice, rho0=rho_surf, degree=2)
        u_init = Expression(('vx',"uztop - (uztop-uzbot)*pow((H-x[1])/H,0.35)"),vx=Constant(0.0), H=H, uztop=-acc_rate, uzbot=-acc_rate*(rho_surf/rho_ice),  degree=2)
        rho_init, v_init = interpolate(r_init, U), interpolate(u_init, V)
        rho.assign(rho_init)
        v.assign(v_init)
        
    else:
        
        rho_prev.set_allow_extrapolation(True)       
        rho_init = interpolate(rho_prev,U)
        v_init = interpolate(v_sol,V)
        rho.assign(rho_init)
        v.assign(v_init)
        
    #--------------------------------------INTERPOLATE RHO------------------------------------------------#
    
    if tstep >0: 
        
        rho_old=rho_prev.copy()
        rho_old.set_allow_extrapolation(True)

        rho_new=Function(U)
        LagrangeInterpolator.interpolate(rho_new,rho_old)

        rho_prev.assign(rho_new)
        rho_prev.set_allow_extrapolation(True)       
        
        strainl.set_allow_extrapolation(True)       
        strains.set_allow_extrapolation(True)       
        
        #................................KEEP IT BELOW RHO_ICE...........................#
        
        rhovec = rho_prev.vector()[:]
        rhovec[rhovec > rho_ice_softened] = rho_ice_softened
        rho_prev.vector()[:] = rhovec
    
    rho_prev=project(rho_prev,U)
    Aglen=get_Aglen(U,heated_margins=heated_margins)
    
    print('                              Aglen computed')
            
    #-----------------------------------------------------------------------------------------------------#
    #--------------------------------------SOLVE FEM PROBLEM----------------------------------------------#
    #-----------------------------------------------------------------------------------------------------# 
    
    #--------------------------------GET a, b VARIABLES AND SIGMA-----------------------------------------
    
    #----------------------------------- a and b
    a_,b_=get_ab(rho_prev,rho_ice,phi_snow,ab_phi_lim,n,K)
    print('                              a&b computed')
    
    #---------------------- get shear and longitudinal strain rates. If rho=rho_ice*0.99, the straiin value will be 0
    rho_max=rho_ice*0.99
    new_strain_value=Constant(0.0)
    
    strainl_add=project(strainl,U)
    strains_add=project(strains,U)
    csub_array1 = strainl_add.vector().get_local()
    csub_array2 = strains_add.vector().get_local()
    rho_prev_arr = rho_prev.vector().get_local()
    
    np.place(csub_array1, rho_prev_arr>rho_max, new_strain_value)
    strainl_add.vector()[:] = csub_array1
    np.place(csub_array2, rho_prev_arr>rho_max, new_strain_value)
    strains_add.vector()[:] = csub_array2
    
    #----------------------------------- sigma
    sigma=get_sigma(v,a_,b_,Aglen,n,strainl_add,strains_add)
    print('                              sigma computed')
    
    #-----------------------------------SOLVE MOMENTUM EQUATION--------------------------------------------
    a_v = inner(sigma,grad(wv))*dx           
    L_v = rho_prev*inner(g,wv)*dx 
    
    F_v = a_v - L_v
    
    tol, relax, maxiter = 1e-2, 0.35, 50
    solparams = {"newton_solver":  {"relaxation_parameter":relax, "relative_tolerance": tol, "maximum_iterations":maxiter} }
    solve(F_v==0, v, bcs_v, solver_parameters=solparams)
    
    v_sol=project(v,V)
    print('                             solved momentum equation')
    
    #---------------------------------SOLVE MASS BALANCE EQUATION------------------------------------------
    a_rho = Constant(1/dt)*rho*wr*dx + rho*div(v_sol)*wr*dx + dot(v_sol,grad(rho))*wr*dx
    L_rho = wr* Constant(1/dt)*rho_prev *dx 
    
    F_rho = a_rho - L_rho

    tol, relax, maxiter = 1e-2, 0.35, 50
    solparams = {"newton_solver":  {"relaxation_parameter":relax, "relative_tolerance": tol, "maximum_iterations":maxiter} }
    solve(F_rho==0, rho, bcs_rho, solver_parameters=solparams)
    
    rho_prev.assign(rho)  #<-------------UPDATE RHO PROFILE
    print('                             solved mass balance equation')
    
    if tstep==0:
        
        v_init=project(v_sol,V)
        zcoord_initial_np,v_initial_np=convert_to_np(V,v_init)

    #-----------------------------------------------------------------------------------------------------#
    #----------------------------------------CALCULATE NEW H----------------------------------------------#
    #-----------------------------------------------------------------------------------------------------#
   
    #Surface evolution
    sfc.evolve(v_sol, dt)
    # ...new mesh is constructed at next loop entry...
    
    #Calculate rate of height change
    dHdt = max(np.abs(sfc.deltaH))*yr2s/dt
    
    tnow=time.time()
    
    print(f'{tstep=}------t={np.round(tstep*dt/yr2s,3)}yrs------>dHdt={np.round(dHdt,3)}/{dHdt_tol}m')
    print(f'               iter_dt={np.round(tnow-tprev,2)}s \n           (expected_minimum_time={np.round(np.ceil((dHdt-dHdt_tol)/(dHdtprev-dHdt))*(tnow-tprev)/60,2)} mins)\n ')

    #update
    tstep+=1    
    tprev=tnow
    dHdtprev=dHdt    


#----------------------------------some PLOTTING----------------------------------------------------------------------

#-------------------------------------densities

fig, ax=plt.subplots(figsize=(15,5))
rhoplot=plot(rho_prev,cmap='PuBu')
plt.ylim([93,190])
plt.xlim([deltaL,L-deltaL])
plot(rho_prev,mode='contour',levels=[280,830,915],cmap='Pastel2')
clb=plt.colorbar(rhoplot,orientation="vertical",label=r'Density (kg/m$^3$)')  

plt.xlabel('Offset along line (km)')
positions = (9520, 14520,19520,24520,29520,34520,39520)
labels = (15,20,25,30,35,40,45)
plt.xticks(positions, labels)
positions = (173, 163,153,143,133,123,113,103,93)
labels = (0,10,20,30,40,50,60,70,80)
plt.yticks(positions, labels)

ax.set_aspect(aspect=125)
ax.set_ylabel('Depht (m)')
plt.savefig(outfolder+'strain_softening_SS_rho_K_'+str(K)+'_'+margintag+f'dtf{dtf}hres{Hres}deg{deg_rho}tol{dHdt_tol}.png',format='png',dpi=200)

#-------------------------------velocities

fig, ax=plt.subplots(figsize=(15,5))
vplot=plot(v_sol*yr2s,cmap='magma')
plt.ylim([93,190])
plt.xlim([deltaL,L-deltaL])
clb=plt.colorbar(vplot,orientation="vertical",label=r'V (m/a)')  

plt.xlabel('Offset along line (km)')
positions = (9520, 14520,19520,24520,29520,34520,39520)
labels = (15,20,25,30,35,40,45)
plt.xticks(positions, labels)
positions = (173, 163,153,143,133,123,113,103,93)
labels = (0,10,20,30,40,50,60,70,80)
plt.yticks(positions, labels)

ax.set_aspect(aspect=125)
ax.set_ylabel('Depht (m)')
plt.savefig(outfolder+'strain_softening_V_SS_K_'+str(K)+'_'+margintag+f'dtf{dtf}hres{Hres}deg{deg_rho}tol{dHdt_tol}.png',format='png',dpi=200)

#------------------------------------------------------------------------- COMPUTING AND SAVING SOME ARRAYS


x_r  = U.tabulate_dof_coordinates()[:,0]
z_r  = U.tabulate_dof_coordinates()[:,1]


non_repeat_xs = list(set(x_r)) #set removes the duplicated
order_x=np.argsort(non_repeat_xs)
xs_nodes=np.zeros(len(non_repeat_xs))

for i in range(len(order_x)):
    xs_nodes[i]=non_repeat_xs[order_x[i]]                        

zs_nodes=np.zeros((len(xs_nodes),Hres*deg_rho-(deg_rho-1)))

for i in range(len(xs_nodes)):

    column_zs=z_r[np.where(x_r==xs_nodes[i])]
    order_z=np.argsort(column_zs)
    zs_nodes[i,:]=column_zs[order_z]



np.save(outfolder+'modeloutput_surface_elevation_K_'+str(K)+'_'+margintag+f'dtf{dtf}hres{Hres}deg{deg_rho}tol{dHdt_tol}.npy',zs_nodes[:,-1])
np.save(outfolder+'modeloutput_nodeZs_K_'+str(K)+'_'+margintag+f'dtf{dtf}hres{Hres}deg{deg_rho}tol{dHdt_tol}.npy',zs_nodes)
np.save(outfolder+'modeloutput_columnXs_K_'+str(K)+'_'+margintag+f'dtf{dtf}hres{Hres}deg{deg_rho}tol{dHdt_tol}.npy',xs_nodes)


#Get the density values at the nodes

rho_nodes=np.zeros((len(xs_nodes),np.shape(zs_nodes)[1]))

#FROGA!!
rho.set_allow_extrapolation(True) 

for i in range(len(xs_nodes)):
    for j in range(np.shape(zs_nodes)[1]):
        rho_nodes[i,j]=rho(xs_nodes[i] ,zs_nodes[i,j])


#Get the BCO depth by checking at which depth is the density=830 kg/m^3
zsBCO=np.zeros(len(xs_nodes))
for i in range(len(xs_nodes)):
    f_zBCO=interp1d(rho_nodes[i,:],zs_nodes[i,:])
    zsBCO[i]=f_zBCO(830)


# ### -----------------------------------------------------GET SURFACE ELEVATION

np.save(outfolder+'modeloutput_rhos_K_'+str(K)+'_'+margintag+f'dtf{dtf}hres{Hres}deg{deg_rho}tol{dHdt_tol}.npy',rho_nodes)
np.save(outfolder+'modeloutput_BCOdepth_K_'+str(K)+'_'+margintag+f'dtf{dtf}hres{Hres}deg{deg_rho}tol{dHdt_tol}.npy',zsBCO)

#-----------------------------------------------------------------------------------------------------------

plt.figure(figsize=(20,5))
plt.plot(np.linspace(10480-deltaL,47920+deltaL,len(zs_nodes[:,-1]))/1000,-(zs_nodes[:,-1]-max(zs_nodes[:,-1])),color='blue',label='Model output surface')
plt.gca().fill_between(np.linspace(10480-deltaL,47920+deltaL,len(zs_nodes[:,-1]))/1000,-(zs_nodes[:,-1]-max(zs_nodes[:,-1])),-5,facecolor='tab:blue', alpha=0.5)
plt.xlim([10480/1000,47920/1000])
plt.ylim([-1,14])
plt.gca().invert_yaxis()
plt.legend(loc='lower right')
plt.xlabel('Offset along line (m)')
plt.ylabel('Depth (m)')
plt.savefig(outfolder+'NEGIS_surface_K_'+str(K)+'_'+margintag+f'dtf{dtf}hres{Hres}deg{deg_rho}tol{dHdt_tol}.png',dpi=150,format='png')



