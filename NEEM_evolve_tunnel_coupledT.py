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
In this sample code in particular, we reproduce the transient collapse of a Greenlandic firn tunnel as a cross-section
model. The simulation is based on the tunnel built at the NEEM drilling site during the 2012 campaign by setting the
initial dimensions and surface temperatures to the ones measured. The results are then compared to the collapse
measurements taken during the two-year-long experiment

"""

##############################################################IMPORTS
#importing modules in this particular order to avoid version conflicts
from math import *
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate as interpolatescipy
from scipy.interpolate import interp1d
import os

try:
    #h5py must be imported before importing fenics
    import h5py
    import gmsh
    import meshio
    #Set shortcut for calling geometry functions
    geom = gmsh.model.geo

    print('before----FENICS',h5py.__version__)
    print('before----FENICS',gmsh.__version__)
    print('before----FENICS',meshio.__version__)

except ImportError:
    print("meshio and/or gmsh not installed. Requires the non-python libraries:\n",
          "- libglu1\n - libxcursor-dev\n - libxinerama1\n And Python libraries:\n"
          " - h5py",
          " (pip3 install --no-cache-dir --no-binary=h5py h5py)\n",
          "- gmsh \n - meshio")
    exit(1)

#now we can import fenics and dolfin
from fenics import *   
from dolfin import *

#control verbosity of solve function
set_log_level(21)

#for plotting and font management
from matplotlib.ticker import AutoMinorLocator
plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams.update({'font.size': 12})

##############################################################


def acc_from_watereqyr_to_snoweqs(acc,rho_snow, verbose=True):
    """Change accumulation units from ice equivalent meters per year to snow equivalent meters per second"""
    
    accum_watereqyr = acc # water equiv. accum, m/yr
    accum_iceeqyr = 1000/rho_ice * accum_watereqyr
    accum_iceeqs = accum_iceeqyr/yr2s # ice accum, m/s 
    acc_rate = rho_ice/rho_snow * accum_iceeqs

    if verbose:
        print(f'{acc=} m of water eq per year')
        print(f'{acc_rate*yr2s=} m of snow eq per year')
        print(f'{acc_rate=} m of snow eq per second')
    
    return acc_rate


    
def get_ab_Z07(rho,rho_ice,phi_snow,ab_phi_lim,nglen,K):
    
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
    
    eps_dot=sym(grad(v))                      
    J1=tr(eps_dot) 
    J2=inner(eps_dot,eps_dot)
    eps_E2=1/a*(J2-J1**2/3) + (3/2)*(1/b)*J1**2
    viscosity = (1/2)**((1-nglen)/(2*nglen)) * Aglen**(-1/nglen) * (eps_E2)**((1-nglen)/(2*nglen))    
    sigma = viscosity * (1/a*(eps_dot-(J1/3)*Identity(2))+(3/2)*(1/b)*J1*Identity(2))
    
    return sigma

"""
Define boundaries of the domain
In order to define the inner irregular boundary, we define it first as the whole
domain of boundaries, and start subtracting the rest of the well defined boundaries 
by properly defining them
"""

class obstacle_boundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

class bottom_boundary(SubDomain):
    def inside(self, x, on_boundary):
        # return on_boundary and near(x[1],0)
        return on_boundary and (x[1]<1)  

class top_boundary(SubDomain):
    def inside(self,x,on_boundary):
        dz=0.1 #tolerance. how many meters from the surface's minimum height (which is updated every timestep)
        return on_boundary and (x[1]> (zmin-dz))
        #the left and right boundary nodes within this zone are not a problem because we
        # will define the other boundaries on top of these definitions

class right_boundary(SubDomain):
    def inside(self,x,on_boundary):
        # return on_boundary and near(x[0],L)
        return on_boundary and (x[0]>L-0.1)  

class left_boundary(SubDomain):
    def inside(self,x,on_boundary):
        # return on_boundary and near(x[0],0)  
        return on_boundary and (x[0]<0.1)
    
#Left and right periodic boundaries
class PeriodicBoundary(SubDomain):
    # Left boundary is "target domain" G
    def inside(self, x, on_boundary):
        return (x[0]<0.1) and on_boundary
    # Map right boundary (H) to left boundary (G)
    def map(self, x, y):
        y[0] = x[0] - L
        y[1] = x[1]

pbc=PeriodicBoundary()      

"""-----------------------------------------------------REMESHING-------------"""

def msh2xdmf(filename,outname="mesh_temp.xdmf"):

    """Change the format of the mesh file we have saved"""
    
    msh = meshio.read(filename)
    
    line_cells = []
    for cell in msh.cells:
        if cell.type == "triangle":
            triangle_cells = cell.data
        elif cell.type == "line":
            if len(line_cells) == 0:
                line_cells = cell.data
            else:
                line_cells = np.vstack([line_cells, cell.data])
    
    line_data = []
    for key in msh.cell_data_dict["gmsh:physical"].keys():
        if key == "line":
            if len(line_data) == 0:
                line_data = msh.cell_data_dict["gmsh:physical"][key]
            else:
                line_data = np.vstack([line_data, msh.cell_data_dict["gmsh:physical"][key]])
        elif key == "triangle":
            triangle_data = msh.cell_data_dict["gmsh:physical"][key]
    
    triangle_mesh = meshio.Mesh(points=msh.points[:, :2], cells={"triangle": triangle_cells},
                                cell_data={"name_to_read": [triangle_data]})
    
    line_mesh = meshio.Mesh(points=msh.points[:, :2], cells=[("line", line_cells)],
                            cell_data={"name_to_read": [line_data]})
    meshio.write(outname, triangle_mesh)
    #meshio.write("mf_ir.xdmf", line_mesh)

def angle(x,z,xc,zc):

    """Compute angle of x,z coordinates with respect to the xc,zc center coordinates"""
    
    if x>=xc:  
        alpha= np.arctan((z-zc)/(x-xc))
    else:
        alpha= np.pi + np.arctan((z-zc)/(x-xc))
        
    if alpha>2*np.pi:
        alpha-=2*np.pi
        
    elif alpha<0:
        alpha+=2*np.pi
        
    return alpha


def dilation(xs,zs,factor=0.95):

    """Dilate set of points taking the mean coordinates as center of reference
    When the factor is bigger than one the image is bigger than the original
    otherwise, it is smaller
    We are interested in the latter because we use this function to order the tunnel's inner boundary nodes effectively
    """
    
    Npoints=len(xs)
    
    xdil=np.zeros(Npoints)
    zdil=np.zeros(Npoints)
    
    xc=np.mean(xs)
    zc=np.mean(zs)
    
    for i in range(Npoints):
        
        r=np.sqrt((xc-xs[i])**2 + (zc-zs[i])**2)
        alpha = angle(xs[i],zs[i],xc,zc)
        
        r_new= factor * r
        
        xdil[i]= xc + r_new*np.cos(alpha)
        zdil[i]= zc + r_new*np.sin(alpha)
        
    return xdil,zdil


    
def sort_hole_fenics_v2(hole_coords,ndmin=4):
    
        """
        Function that defines an order for the tunnel's boundary nodes, which is not trivial to sort since
        it is a closed loop and we can have a lot of identical xs and zs (reason why we use a polar approach)
        and the edges sometimes form bottlenecks, which makes implementing this order from the distance to the
        previous node impossible.
        We do not care where the points start, we just need them to follow a direction around
        the hole and back to the start

        hole_coords = U.tabulate_dof_coordinates()
        ndmin=how many nearest neighbors will be checked

        """
        #initial sort
        xs_hole_xsorted, zs_hole_xsorted, s_numdofs = sort_fenics(hole_coords,axis=0) #just for the algorithm to be quicker
        s_numdofs=len(xs_hole_xsorted)

        #dilate
        xdil,zdil =  dilation(xs_hole_xsorted, zs_hole_xsorted,factor=0.94)
        
        #points sorted to follow the loop
        xs_hole=[]
        zs_hole=[]
        
        #arbitrary initial point. (away from corner spikes)
        
        isafe=int(s_numdofs/2)
        
        xs_hole.append(xs_hole_xsorted[isafe])
        zs_hole.append(zs_hole_xsorted[isafe])
        
        x_rn = xs_hole_xsorted[isafe] #coordinates of the point we are calculating the distance from rn
        z_rn = zs_hole_xsorted[isafe]
        
        xd_rn = xdil[isafe]
        zd_rn = zdil[isafe]
        
        #delete it from the list of points to be sorted
        xs_hole_xsorted= np.delete(xs_hole_xsorted,isafe)
        zs_hole_xsorted= np.delete(zs_hole_xsorted,isafe)
        
        xdil= np.delete(xdil,isafe)
        zdil= np.delete(zdil,isafe)
        
        #calculate closest point and follow the loop from there
        #the direction will be random but not important
        #we can maybe improve the cost of this function by searching just in the closest half
        
        for ii in range(s_numdofs-1):
            
            ndmin=min(ndmin,len(xs_hole_xsorted))
            
            dist= (xs_hole_xsorted - x_rn)**2 + (zs_hole_xsorted - z_rn)**2
            i_mins=np.argsort(dist)[:ndmin] #indexes of the ndmin minimum distances

            ref_angle=angle(x_rn,z_rn,xd_rn,zd_rn)
            
            #angle with respect to inner close point. OXY frame of reference
            alpha_mins=np.zeros(ndmin)
            #angle with respect to inner close point. POINT_rn frame of reference
            alpha_mins_ref=np.zeros(ndmin)
            
            for i in range(ndmin):
                
                alpha_mins[i]=angle(xs_hole_xsorted[i_mins[i]],zs_hole_xsorted[i_mins[i]],xd_rn,zd_rn)
                
                dif = alpha_mins[i] - ref_angle
                if dif > 0:
                    alpha_mins_ref[i] = dif
                else:
                    alpha_mins_ref[i] = dif + 2*np.pi
                 

            i_next= i_mins[np.argmin(alpha_mins_ref)]
            
            #append
            xs_hole.append(xs_hole_xsorted[i_next])
            zs_hole.append(zs_hole_xsorted[i_next])
            
            x_rn = xs_hole_xsorted[i_next] #coordinates of the point we are calculating the distance from rn
            z_rn = zs_hole_xsorted[i_next]
            
            xd_rn = xdil[i_next]
            zd_rn = zdil[i_next]
            
            #delete it from the list of points to be sorted
            xs_hole_xsorted= np.delete(xs_hole_xsorted,i_next)
            zs_hole_xsorted= np.delete(zs_hole_xsorted,i_next)
            
            xdil= np.delete(xdil,i_next)
            zdil= np.delete(zdil,i_next)
            
        
        return xs_hole, zs_hole
        


def remesh_acc(xs_hole,zs_hole,xs_surf,zs_surf,L,tstep,dt,acc_rate,n_itacc, tmsurf=0.15,tmr=0.15,nacclayers=5,maxN=100, mode='linear',outname='mesh_temp.msh'): #jarri txukunago gero
    
    """
    As ALE.move() evolves the mesh, the original discretization of the space stops being appropriate.
    We need to periodically discretize the deformed domain to avoid problems

    Mode can be linear or spline. Linear cuts some corners if the number of tunnel-boundary-nodes is not too high,
    but spline smooths the shape too much

    tmsurf = defines the mesh resolution on the surface (average distance between nodes, smaller is finer)
    tmr = defines the mesh resolution on the tunnel's inner surface (average distance between nodes, smaller is finer)
    These two need to be tuned by hand because if the grid is too fine, the deformation imposed by ALEmove since the last remesh
    can be too big (especially for the biggest K values) and the grid overlaps and Fenics blows up with a [-1] error
    maxN caps the number of nodes that there can be on the tunnel boundary to try to avoid this

    Every n_itacc iterations it accumulates the meters of snow that have fallen in that time.
    how many iterations to accumulate until it can be described by a new surface node layer"""
    
    #Define size of mesh for different places
    tm = 3.0#Exterior resolution (far from tunnel). Much more regular evolution, so can be described with a coarser grid

    acc_iter=True if tstep%n_itacc==0 else False #flag to know if in this iteration the acumulation layer will be added

    deltah=n_itacc*dt*acc_rate if acc_iter else 0 #meters that should be added to account for accumulation

    
    #Initialize mesh
    gmsh.initialize()

    ######### Create tunnel curve using spline ###########
    
    N_hole=len(xs_hole)
    
    """The resolution of each point does not take into account how many 
    points it already has around, so there is a positive feedback loop
    that makes the resolution go crazy from a certain resolution to
    point density ratio. 
    need to limit the maximum number of points by hand"""
    
    # maxN=150 #to be adjusted. assemble() glitches at around 2000
    portion= (N_hole//maxN) + 1 #portion of the points to keep

    xs_hole = xs_hole[::portion]
    zs_hole = zs_hole[::portion]
    N_hole=len(xs_hole)

    ps_hole=[]
    
    for i in range(N_hole):
        ps_hole.append(geom.addPoint(xs_hole[i],zs_hole[i], 0,tmr))
    
    ps_hole.append(1) #First and last points (tags) must be the same to have a close boundary!

    #Create 'line' by interpolating around the give points
    
    if mode=='spline':
        
        curve_hole = geom.addBSpline(ps_hole,-1)
        #Create the curve loop of the hole
        hole = geom.addCurveLoop([curve_hole])
        
    elif mode=='linear':
        
        curve_hole=[]
        for i in range(N_hole):
            curve_hole.append(geom.addLine(ps_hole[i],ps_hole[i+1]));

        #Create the curve loop of the hole
        hole = geom.addCurveLoop(curve_hole);
    
    
    ######### Create exterior boundary using spline ###########

    H_left=zs_surf[np.argmin(xs_surf)]
    H_right=zs_surf[np.argmax(xs_surf)]

    #Irregular surface (left to right)
    xs1=np.concatenate(([0],xs_surf,[L]))
    ys1=np.concatenate(([H_left + deltah],zs_surf + deltah,[H_right + deltah])) #<<<<<<<<<<<<hor gehituta accumulazioa
    
    
    #Add all the surface points to mesh
    ps_surf=[]
    
    for i in range(len(xs1)):
        ps_surf.append(geom.addPoint(xs1[i],ys1[i], 0,tmsurf))
     
    p1=geom.addPoint(L,0,0,tm)
    p2=geom.addPoint(0,0,0,tm)
    
    l1=geom.addBSpline(ps_surf,-1)
    l2=geom.addLine(N_hole+len(xs1),p1)
    l3=geom.addLine(p1,p2)
    l4=geom.addLine(p2,N_hole+1)
    
    ext=geom.addCurveLoop([l1,l2,l3,l4])

    ############ Generate the mesh itself ##############
    
    #Create surface between exterior and hole
    s = geom.addPlaneSurface([ext,hole])
    
    gmsh.model.addPhysicalGroup(2, [s], tag=tstep)
    gmsh.model.setPhysicalName(2, tstep, "Firn/ice")
    
    #Generate mesh and save
    
    geom.synchronize()
    gmsh.model.mesh.generate(2)
    
    gmsh.option.setNumber('Mesh.SurfaceFaces', 1)
    gmsh.option.setNumber('Mesh.Points', 1)
    
    
    gmsh.write(outname) #write msh file, but we ned xdmf to work with fenics
    msh2xdmf(outname,outname="mesh_temp_100.xdmf") #rewrite the file into an xdmf to be read
    
    gmsh.finalize() #important to close gmsh!!
    
    return acc_iter,deltah,n_itacc,xs1,ys1

def sort_fenics(scoords,axis):
    
        """Sort the unsorted arrays that Fenics works with
        
        scoords = U.tabulate_dof_coordinates()
        axis---according to which axis must be sorted---(xs=0,zs=1)
        
        """
        step=1
        
        axis_dof= scoords[:,axis] #axis coords of func space NODES (xs=0,zs=1)
        IS = np.argsort(axis_dof)
        s_numdofs= len(axis_dof)
        
        scoords_x=scoords[:,0][IS] #sorted
        scoords_z=scoords[:,1][IS]

        #depending on space order, points might appear doubled
        if (np.abs(s_numdofs-2*len(np.unique(axis_dof)))<2):
            print('doubled')
            step=2
        
        return scoords_x[::step],scoords_z[::step], int(s_numdofs/2)
    

    
def A_glen_Arrhenius(T):

    """Compute the flow factor A(T) for that temperature. This function allows for temperatures higher than -10 too"""

    R = Constant(8.314)

    #--------------------the activation energy for the creep, Q
    Qminus=60e3
    Qplus=139e3 #this is the zwinger value
    Q=conditional(le(T,-10 + 273.15),Qminus,Qplus) #LessOrEqual 
    
    #-----------------A0 is also defined in two parts
    A0minus=3.985e-13
    A0plus=1.916e3 #zwinger
    A0=conditional(le(T,-10 + 273.15),A0minus,A0plus) #LessOrEqual
    
    return A0*exp(-Q/(R*T))

##########################################

#Define parameters
n=3 #exponent
g=Constant((0.0,-9.82))


#densities
rho_surf=310#.3717043 #300
rho_ice= 910
rho_ice_softened= rho_ice - 0.2

phi_snow= 0.4 #rho_surf/917 #NEEM
ab_phi_lim=0.81

#time units
day2s=24*60*60
yr2s=365.25*day2s

# Thermal parameters of pure ice
c0=2127.5
c1=7.253
T0=273.16

k0=9.828
gammaT=0.0057
kf0=0.138
kf1=1.010e-3
kf2=3.233e-6
c_x=10

#trench density and velocities parameters
#-------------------TO CREATE THE APPROPRIATE INITIAL DENSITY AND VELOCITY FIELD GUESSES
rho_trench= 550 #denser firn after cutting the trench out. measured
deltax_trench=0.5 #distance from the trench to linearly smooth the density
deltaz_trench=0.5
u_trench=3.5 #upper trench
l_trench=2.25
trench_lim=3.1
bump=1
factor_inner=0.9 #0.9 #factor to account for the bigger load induced increased velocities right below the tunnel
factor_outer=1.1 #1.1 #factor to account for the bigger load induced increased velocities to the sides of the tunnel
trench_zoom_factor=1

#----------------------------ACCUMULATION
snow550=True #accumulation in 550kg/m3 instead of snow. to avoid computational issues
acc_meas=0.20*2 #Measured acc in m of water equivalent per year #JP2x factor
if snow550:
    acc_rate=acc_from_watereqyr_to_snoweqs(acc_meas,rho_snow=rho_trench) #in m 550snow/s
else:
    acc_rate=acc_from_watereqyr_to_snoweqs(acc_meas,rho_surf) #in m snow/s
print(f' \n acc in m SNOW/yr {acc_from_watereqyr_to_snoweqs(acc_meas,rho_surf)*yr2s=}')
print(f'  acc in m TRENCHSNOW/yr {acc_from_watereqyr_to_snoweqs(acc_meas,rho_snow=rho_trench)*yr2s=}\n')
print(f'{acc_rate=} m snow per second')
print(f' per step: {acc_rate*dt=} m snow per step')

#--------------SURFACE TEMPERATURE RECORD
#this will have to be handled for the location of the dataset in every particular machine
#LOAD temperatures for this period from GC-Net dataset (Vandecrux and others, 2023)
#period----from 2012.5273785078yr to 2014.5273785078yr (tunnel collapse data period)
# neem_Ts=np.load('NEEM_temperaturesTA1.npy') #raw record
neem_Ts=np.load('NEEM_temperaturesTA1_smooth.npy') #smoothed temperature record
neem_ts=np.load('NEEM_timesTA1.npy')

neem_t0=neem_ts[0]
print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>available temperature data from {np.min(neem_ts)}yr to {np.max(neem_ts)}yr \n {np.max(neem_ts) - np.min(neem_ts) }yr         ')


#>>>>>>>>>>>>>>>>>>>>>User input
print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>USER INPUT<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
print('-------------write value and press enter (or press enter directly for default value)\n')

#-------------------------------------------------TEMPERATURE
Tsite=(np.float(input('     >>>>>Write average site temperature in ºC (default = -28.8ºC NEEM)') or -28.8)+273.15)
T0hotter=(np.float(input('     >>>>>Write trench initial temperature in ºC (default= -5ºC --previous day average at NEEM)') or -5)+273.15)

#---------------------------------------------------VALUE OF K
K=int(input('     >>>>>K? (default=1000)') or 1000)

#---------------------------------------------------AGLEN
print('\n---------------Aglen will be computed with the coupled temperature field\n')

#---------------------------------------TIME DISCRETIZATION
p.float(input('     >>>>>final factor enhancement? (default= 0.5)' ) or 0.5)
dt=(0.025/3*yr2s)*finalfactor
print(f'------------------------------{dt=} s')
print(f'----------------------------TIME DISCRETIZATION ={dt/(60*60*24)=} days')
print(f'----------------------------{dt/yr2s=} yr')

#-----------------------REMESHING
acc_iter=False
tmsurf=np.float(input('     >>>>>surface mesh discretization? (0.15 default for k=1000)') or 0.15)
tmr=np.float(input('     >>>>>tunnel inner surfae mesh discretization? (0.125 default for k=1000)') or 0.125)

#-----------------------------ACCUMULATION  iteration inside the remesh every n_itacc
nacclayers=1 #remeshes with additional accumulation node layer as soon as it snows enough to add a layer
#small calculation to initiate
h_nlayer= tmsurf-0.05  # \sim average width of a 'node layer'
              #the system will be loosing mass from the top if too rough
n_itacc= int(nacclayers*np.ceil(h_nlayer/(dt*acc_rate))) #how many iterations aprox

print(f'Suggested {n_itacc=} \n WATCH OUT! n_itacc needs to be a multiple of how often we remesh\n')
n_itacc=int(input(f'     >>>>>Choose n_itacc (default=4 iterations)') or 4)
remeshstep=int(input(f'     >>>>>introduce an appropiate remeshstep (default= 4 iterations)') or 4)

#------------------------------------------ maximum number of tunnel nodes
print('\nMAXIMUM NUMBER OF TUNNEL SURFACE NODES----------------------------------> WATCH OUT---reduce if [-1] errors start to appear (or reduce dt)\n')
hole_maxN=int(input(f'     >>>>>maximum N_HOLE ? (default= 500)') or 500)

#---------------------------------------------plotting
print('\nPLOTTING----------------------------------> WATCH OUT---consider plotting less or lowering the dpi of the images if memmory issues arise (usually in the form of >>killed)\n')
plotting=input(f'     >>>>>plotting? (default=True)') or True
nplot=int(input(f'     >>>>>how often should it plot? (default=100 iterations)') or 100)


#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#-----------------------------------------INITIALIZE ARRAYS AND VARIABLES
#Parameters for mesh creation
L = 20  # Length of channel
H = 30  # height of channel
#-----------------------

tstep=0
zmin=29 #surface boundary minimum height (updated every timestep)

#-----------------------------------------DEFINE PARAMETERS FOR THE STEADY STATE 

dHdt=100      #So that it enters the while
dHdt_tol=0.01 #Change to consider that we have reached the steady state

igifs=0
filenames_rho = []
filenames_rho_zoom = []
filenames_vel = []
filenames_mesh = []

#-------------------------------------------MESH-----------------------------------------------------#

#mesh to be read
meshfilename="NEEM_tunnel_initialMESH.xdmf"  #with the '.xdmf'  included


mesh = Mesh()
with XDMFFile(meshfilename) as infile:  #READ FILE<<
    infile.read(mesh)


nfinersteps=0
nyears_ref=1.97#
nyears= nyears_ref + 0.02  #some tolerance
lastyear=-1
nsteps= int(nfinersteps + (nyears*yr2s - nfinersteps*dt)/(dt*factorGM97)) + 1


neemsaved=False
still_not_shown_ref=True

volumes=np.array([])
hole_deltaxs=np.array([])
hole_deltazs=np.array([])
hole_zmins=np.array([])
times=np.array([])

roofkgs=np.zeros(nsteps)
theoricload= rho_trench*1.5*np.ones(nsteps) #we will start summing from initial


#------------------------------------------COMPUTE AVERAGE TEMPERATURE FOR THE STEP'S dt WINDOW
avg_Tstep=np.zeros(nsteps)
steps_ts=np.zeros(nsteps)


for ttstep in range(nsteps):
    
    steps_ts[ttstep]= neem_t0 + ttstep*dt/yr2s
    
    if ttstep>0:
        
        tinit_yr = neem_t0 + (ttstep-1)*dt/yr2s
        tfin_yr = neem_t0 + ttstep*dt/yr2s
        
        windowmask= (tinit_yr <= neem_ts) & (neem_ts <= tfin_yr)
        
        Tstep= np.mean(neem_Ts[windowmask])
        avg_Tstep[ttstep]=Tstep
        
    elif ttstep==0:
        
        Tstep= neem_Ts[0]
        avg_Tstep[0]=Tstep


#---------------------------check that it's alright

plt.figure(figsize=(30,15))

plt.axhline(0,c='r',lw=2,ls='-')
plt.axhline(-10,c='r',lw=1,ls='--')

plt.scatter(neem_ts,neem_Ts,c='k',s=5,alpha=0.5,label='Hourly RAW')
plt.plot(steps_ts,avg_Tstep,lw=3.5, c='tab:orange',label='step average')

plt.legend()
plt.savefig('TA1temp_data_STEPavg.png',dpi=300)


#----------------------------------------------------------------------------------------------
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>MAIN LOOP<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#----------------------------------------------------------------------------------------------

while tstep<=nsteps:


    q_degree = 3
    dx = dx(metadata={'quadrature_degree': q_degree})
    
    #----------------------------------------------------REMESH
    
    if (tstep>0 and tstep%(remeshstep)==0):
        print('::::::::::::::REMESHING:::::::::::::::::::::::::::')
        
        #Get coordinates and velocities of the points at the obstacle's surface
        mesh.init(2,1)
        dofs = []
        cell_to_facets = mesh.topology()(2,1)
        for cell in cells(mesh):
            facets = cell_to_facets(cell.index())
            for facet in facets:
                if boundary_subdomains[facet] == 5: #We have given the number 5 to the subdomain of the obstacle
                    dofs_ = V.dofmap().entity_closure_dofs(mesh, 1, [facet])
                    for dof in dofs_:
                        dofs.append(dof)
        
        unique_dofs = np.array(list(set(dofs)), dtype=np.int32)
        hole_coords = V.tabulate_dof_coordinates()[unique_dofs] #coordinates of the tunnel's surface nodes
        
    
        #Get coordinates and velocities of the points at th surface
        mesh.init(2,1)
        dofs = []
        cell_to_facets = mesh.topology()(2,1)
        for cell in cells(mesh):
            facets = cell_to_facets(cell.index())
            for facet in facets:
                if boundary_subdomains[facet] == 2: #We have given the number 2 to the subdomain of the surface
                    dofs_ = V.dofmap().entity_closure_dofs(mesh, 1, [facet])
                    for dof in dofs_:
                        dofs.append(dof)
        
        unique_dofs = np.array(list(set(dofs)), dtype=np.int32)
        surface_coords = V.tabulate_dof_coordinates()[unique_dofs] #surface node coordinates
    
        
        """but we need to sort them before passing them to gmsh!!"""
        surface_xs, surface_zs,_ = sort_fenics(surface_coords,axis=0)
        hole_xs, hole_zs = sort_hole_fenics_v2(hole_coords,ndmin=4)
        
        
        #save hole max deltax and max deltaz throughout simulation
        #in order for this to be representative the hole's shape must be relatively regular
        #(as it is)
        hole_deltaxs=np.append(hole_deltaxs, np.max(hole_xs) - np.min(hole_xs))
        hole_deltazs=np.append(hole_deltazs, np.max(hole_zs) - np.min(hole_zs))
        hole_zmins= np.append(hole_zmins,np.min(hole_zs))
        
        print(f'\n\n{hole_zmins[-1]=}-----------\n    {hole_deltaxs[-1]=}\n    {hole_deltazs[-1]=}')
        
        # volumes=np.append(volumes,Vol)
        times=np.append(times,tstep*dt/yr2s)
        
        # np.save('volumes_k'+str(K)+'.npy',volumes)
        np.save('times_k'+str(K)+'.npy',times)
        np.save('hole_zmins_COUPLED_k'+str(K)+'.npy',hole_zmins)
        np.save('hole_deltaxs_COUPLED_k'+str(K)+'.npy',hole_deltaxs)
        np.save('hole_deltazs_COUPLED_k'+str(K)+'.npy',hole_deltazs)

        #save hole shape once a year throughout the simulation: (uncomment)

        # if  int(tstep*dt/yr2s) != lastyear : #if not saved yet for this year

        #     year=int(tstep*dt/yr2s)
        #     prefixes='OPTIMALhole_K'+str(K)+'_'+str(year)+'yr'

        #     np.save('hole_xs_'+prefixes+'.npy',hole_xs)
        #     np.save('hole_zs_'+prefixes+'.npy',hole_zs)
        #     np.save('surface_xs_'+prefixes+'.npy',surface_xs)
        #     np.save('surface_zs_'+prefixes+'.npy',surface_zs)

        #     lastyear=year

        #------------------------------------------------------------------ REMESH<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        
        #create new mesh file
        temp_meshfile='mesh_temp_100'
        acc_iter,deltah,n_itacc,x_snowed_surf,z_snowed_surf = remesh_acc(hole_xs,hole_zs,surface_xs,surface_zs,L,
                                     tstep,dt=dt,acc_rate=acc_rate,n_itacc=n_itacc, tmsurf=tmsurf,tmr=tmr,nacclayers=nacclayers,maxN=hole_maxN,mode='linear',
                                     outname=temp_meshfile+'.msh')
        
        print('\WATCH OUT!\n last value is duplicated due to periodic boundary conditions\n',x_snowed_surf)      

        #read new mesh file
        mesh = Mesh()
        with XDMFFile(temp_meshfile+'.xdmf') as infile:  #READ FILE<<<<<<<<<<<<<<<<<<<<jarri izena eskuz hor
            infile.read(mesh)
            
            
        # #-----compute and save tunnel volume over time------
        
        # #compute bulk + hole volume:
        
        # # print(surface_xs)
        # # print(surface_zs)

        # vbulk=0
        # # print(len(surface_xs))
        # # print(len(surface_zs))
        
        # for j in range(len(surface_xs)+1):
            
        #     if j==0:  
        #         vbulk += surface_zs[j]*( surface_xs[j] -0)
            
        #     elif j==(len(surface_xs)):
        #         vbulk += surface_zs[j-1]* (L -surface_xs[j-1])
                
        #     else:
        #         vbulk += surface_zs[j]*( surface_xs[j]- surface_xs[j-1])
            
         
        # print(L*H)
        # print(vbulk)
        
        
        # #----------subract the tunnel volume to the whole 'bulk' volume
        
        # one = Constant(1)
        # Vol = vbulk - assemble(one * dx(domain=mesh))
        
        # volumes=np.append(volumes,Vol)
        # times=np.append(times,tstep*dt/yr2s)
        
       
    #--------------------------------------BOUNDARY SUBDOMAINS-------------------------------------------#
    
    #Give a number to each different boundary
    boundary_subdomains = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundary_subdomains.set_all(0)


    obstacle=obstacle_boundary()
    obstacle.mark(boundary_subdomains, 5)
    bottom=bottom_boundary()
    bottom.mark(boundary_subdomains, 1)
    top=top_boundary()
    top.mark(boundary_subdomains, 2)
    #now we define left and right on top of the previous definitions,
    # so no problem if top and bottom include some
    #left and right boundary nodes, because they are properly redefined now
    left=left_boundary()
    left.mark(boundary_subdomains, 3)
    right=right_boundary()
    right.mark(boundary_subdomains, 4)
    
    #--------------------------------------FUNCTION SPACE------------------------------------------------#
    
    #Define function space for density
    deg=2 #Polinomial degree
    U=FunctionSpace(mesh, 'Lagrange', deg, constrained_domain=pbc)

    rho=Function(U) # the unknown function
    wr=TestFunction(U)  # the weight function

    #Define function space for velocity
    V=VectorFunctionSpace(mesh, "CG", 2, constrained_domain=pbc)

    v=Function(V) # the unknown function
    wv=TestFunction(V)  # the weight function 
    
    #Define function space for temperature
    deg=2
    Q=FunctionSpace(mesh, 'Lagrange', deg, constrained_domain=pbc)

    T=Function(Q) # the unknown function
    wT=TestFunction(Q)  # the weight function
    
    #----------------------------------------------------------------BOUNDARY CONDITIONS    
        
    #-----------------------------------TOP
    bc_T_t=DirichletBC(Q,avg_Tstep[tstep] + 273.15,boundary_subdomains,2) #T at the top
    #-----------------------------------BOTTOM
    bc_v_b=DirichletBC(V,(0.0,0.0),boundary_subdomains,1) #Velocity at the bottom
    bc_T_b=DirichletBC(Q,Tsite,boundary_subdomains,1) #T at the bottom
    #-----------------------------------LEFT
    bc_v_l=DirichletBC(V.sub(0),0.0,boundary_subdomains,3) #Velocity at the left boundary
    bc_T_l=DirichletBC(Q,Tsite,boundary_subdomains,3) #T at left
    #-----------------------------------RIGHT
    bc_v_r=DirichletBC(V.sub(0),0.0,boundary_subdomains,4) #Velocity at the right boundary
    bc_T_r=DirichletBC(Q,Tsite,boundary_subdomains,4) #T at right

    bcs_rho=[]
    bcs_v=[bc_v_b,bc_v_l,bc_v_r]
    bcs_T=[bc_T_b,bc_T_t]

    #--------------------------------------INITIAL CONDITION--------------------------------------------#
    
    if tstep==0:

        #first of all, identify the dimensions of the whole automatically
        
        #Get coordinates and velocities of the points at the obstacle's surface
        mesh.init(2,1)
        dofs = []
        cell_to_facets = mesh.topology()(2,1)
        for cell in cells(mesh):
            facets = cell_to_facets(cell.index())
            for facet in facets:
                if boundary_subdomains[facet] == 5: #We have given the number 5 to the subdomain of the obstacle
                    dofs_ = V.dofmap().entity_closure_dofs(mesh, 1, [facet])
                    for dof in dofs_:
                        dofs.append(dof)
        
        unique_dofs = np.array(list(set(dofs)), dtype=np.int32)
        boundary_coords = V.tabulate_dof_coordinates()[unique_dofs]
        hole_zmin=np.min(boundary_coords[:,1])
        print('--:::HOLE-zmin=',hole_zmin)
        hole_xmin=np.min(boundary_coords[:,0])
        hole_xmax=np.max(boundary_coords[:,0])
        print('--:::HOLE-xlims=(',hole_xmin,',',hole_xmax,')')
        
        iorder= np.argsort(boundary_coords[:,0])
        initial_hole_xs = boundary_coords[:,0][iorder] #for plotting it afterwards
        initial_hole_zs = boundary_coords[:,1][iorder] #for plotting it afterwards

        #----------INITIAL CONDITIONS------to properly order the nodes before setting the values--------------------------

        #---------------initial density profile--solution of 1d problem---
        #compacted already

        #old, just for comparison (chosen k100 as initial because in the plots is the one that matches better NEEM)
        rho_init_old=np.load('modeled_initial_condition_profiles/rho_NEEM_n3_H0_180_fittedA(T)_K100.npy')
        z_init_old=np.load('modeled_initial_condition_profiles/z_NEEM_n3_H0_180_fittedA(T)_K100.npy')

        #The initial background density field will be the original density profile measured at NEEM (but smoothed)
        rho_init_neem=np.load('modeled_initial_condition_profiles/neem_densities_raw_smooth.npy')
        z_init_neem=np.load('modeled_initial_condition_profiles/neem_densities_raw_smooth_zcoors.npy')
        #The initial velocity proposal is the 1D SS solution for k=200 (lowest RMSE for NEEM)
        v_init_neem=np.load('modeled_initial_condition_profiles/neem_K=200_SSvelocities.npy')
        zv_init_neem=np.load('modeled_initial_condition_profiles/neem_K=200_SSvelocities_zcoors.npy')
        #we will upload the modelled k200 profile too just to compare
        rho_init_neem_k200=np.load('modeled_initial_condition_profiles/neem_K200_SSdensities.npy')
        z_init_neem_k200=np.load('modeled_initial_condition_profiles/neem_K200_SSdensities_zcoors.npy')

        #The initial trench velocity proposal is the 1D SS solution for k=200 for a 550 unburdened slab
        #but the SS solution contains more than just the surface, so we need to filter the surface area applicable to a
        #real pure 550kg volume
        v_init_trench=np.load('modeled_initial_condition_profiles/vK200_trench550_NEEM.npy')
        zv_init_trench=np.load('modeled_initial_condition_profiles/ZvK200_trench550_NEEM.npy')
        #to properly understand how to scale this surface velocities we need the densities it comes from.
        #we will ZOOM into the surface and interpolate in an appropiate area to form the initial V guess
        rho_init_trench=np.load('modeled_initial_condition_profiles/rhoK200_trench550_NEEM.npy')
        zrho_init_trench=np.load('modeled_initial_condition_profiles/ZrhoK200_trench550_NEEM.npy')

        #----------------------------------------------------------------------
        #adjusting to mesh coordinate system (bump surface starting at 30, unperturbed surface at 29)
        z_init_old_adjusted = z_init_old - (np.max(z_init_old) - H + bump)  # adjusted to the new mesh height
        z_init_neem_adjusted = z_init_neem - (np.max(z_init_neem) - H + bump)  # adjusted to the new mesh height
        zv_init_neem_adjusted = zv_init_neem - (np.max(zv_init_neem) - H + bump)  # adjusted to the new mesh height
        z_init_neem_k200_adjusted = z_init_neem_k200 - (np.max(z_init_neem_k200) - H + bump)  # adjusted to the new mesh height
        zv_init_trench_adjusted = zv_init_trench - (np.max(zv_init_trench) - H + bump)  # adjusted to the new mesh height
        zrho_init_trench_adjusted = zrho_init_trench - (np.max(zrho_init_trench) - H + bump)  # adjusted to the new mesh height

        #subtract the Bottom Boundary Condition -acc(rho_snow/rho_ice) velocity
        #because, unlike the 1D SS moving window, we have a fixed bottom now
        #we have just checked that Zs go from bottom to surface so,
        v_init_neem_nobc= v_init_neem - v_init_neem[0]
        v_init_trench_nobc= v_init_trench - v_init_trench[0]

        #we will interpolate to extract the densities and velocities at the depths of interest
        f_rho_neem = interp1d(z_init_neem_adjusted, rho_init_neem, kind='cubic', fill_value='extrapolate')
        f_v_neem = interp1d(zv_init_neem_adjusted, v_init_neem_nobc, kind='cubic', fill_value='extrapolate')
        
        f_v_trench = interp1d(zv_init_trench_adjusted, v_init_trench_nobc, kind='cubic', fill_value='extrapolate')
        #they all start from 29m, but we will use the last one by zooming on the top 2ms


        #-------------------------------------------------------------plot raw initial conditions

        plt.figure(figsize=(10,5))
        plt.plot(z_init_neem_adjusted,rho_init_neem, label='rawrho_adjusted for bump')
        plt.plot(z_init_neem_k200_adjusted,rho_init_neem_k200,label='rhok200_adjusted')
        plt.plot(z_init_old_adjusted,rho_init_old, label='properformat_adjusted',c='r')
        plt.plot(zrho_init_trench_adjusted,rho_init_trench, label='trench rhos_ adjusted')
        plt.axvline(x=0,c='k',lw=1)
        plt.axvline(x=30,c='k',lw=1)
        plt.axvline(x=29,c='k',lw=1,ls='--')
        plt.legend()
        plt.savefig('modeled_initial_condition_profiles/HBdentsitateprofilak.png',format='png',dpi=300)
        plt.close()
        
        plt.figure(figsize=(10,5))
        plt.plot(zrho_init_trench_adjusted,rho_init_trench, label='trench rhos_ adjusted')
        plt.axvline(x=0,c='k',lw=1)
        plt.axvline(x=30,c='k',lw=1)
        plt.axvline(x=29,c='k',lw=1,ls='--')
        plt.legend()
        plt.xlim(23,30)
        plt.savefig('modeled_initial_condition_profiles/550slabzoom.png',format='png',dpi=300)
        plt.close()
        
        
        plt.figure(figsize=(10,5))
        plt.plot(zv_init_neem_adjusted,v_init_neem,c='r', ls='--',label='V200k_adjusted (original)')
        plt.plot(zv_init_trench_adjusted,v_init_trench,c='b',ls='--', label='trench Vs_adjusted (original)')
        plt.plot(zv_init_neem_adjusted,v_init_neem_nobc,c='r', label='V200k_adjusted (no BC)')
        plt.plot(zv_init_trench_adjusted,v_init_trench_nobc,c='b', label='trench Vs_adjusted (no BC)')
        plt.axvline(x=0, c='k', lw=1)
        plt.axvline(x=30, c='k', lw=1)
        plt.axvline(x=29, c='k', lw=1, ls='--')
        plt.legend()
        plt.savefig('modeled_initial_condition_profiles/HBabiaduraprofilak.png',format='png',dpi=300)
        
        plt.close()

        
        # ------------------extracting the coordinates out of the function spaces---
        # ----------------------------------------------------------------------------

        #-------------------------------------RHO
        # self.S  = FunctionSpace(self.smesh, "CG", 1, constrained_domain=pbc_1D)
        scoords_r0 = U.tabulate_dof_coordinates()
        xs_dof_r0 = scoords_r0[:,0] # x coords of func space NODES
        zs_dof_r0 = scoords_r0[:,1] # z coords of func space NODES
        s_numdofs_r0 = len(zs_dof_r0)
        ISz_r0 = np.argsort(zs_dof_r0)  
        
        print('-------------rho0shape', xs_dof_r0.shape)
        
        #-------------------------------------V
        scoords_v = V.tabulate_dof_coordinates()
        xs_dof_v = scoords_v[:,0] # x coords of func space NODES
        zs_dof_v = scoords_v[:,1] # z coords of func space NODES
        s_numdofs_v = len(zs_dof_v)
        ISz_v = np.argsort(zs_dof_v)
        
        print('-------------v0shape', xs_dof_v.shape)

        #-------------------------------------T
        # self.S  = FunctionSpace(self.smesh, "CG", 1, constrained_domain=pbc_1D)
        scoords_T0 = Q.tabulate_dof_coordinates()
        xs_dof_T0 = scoords_T0[:,0] # x coords of func space NODES
        zs_dof_T0 = scoords_T0[:,1] # z coords of func space NODES
        s_numdofs_T0 = len(zs_dof_T0)
        ISz_T0 = np.argsort(zs_dof_T0)  
        
        print('-------------T0shape', xs_dof_T0.shape)
        
        # #check that they both have the same coordinates-------------------------
        # #they do (and in the same order), it's just that the velocities have two components
        # #so they just appear doubled (but in the same order) 
        # #you can fix it by taking the first out of every two coordinates
        # plt.figure()
        # # plt.plot(xs_dof_r0)
        # # plt.plot(xs_dof_v)
        # plt.plot(xs_dof_r0-xs_dof_v[::2])
        # plt.plot(zs_dof_r0-zs_dof_v[::2])
        #-----------------------------------------------------------------

        #intialize r_init and v_init
        r_init=Function(U)
        v_init=Function(V)
        T_init=Function(Q)
        
        """
        r_init.vector()=[r_init0, r_init1, r_init2....r_init267333]
        v_init.vector()=[vx_init0, vy_init0,vx_init1,vy_init1,...vx_init267333,vy_init267333]
                     but note that velocity has 2*267333 elements!!
                     """
        
        
        for ii in range(s_numdofs_r0):
            
            #coordinates (common for both arrays)        
            xii = xs_dof_r0[ii]# x coord of DOF (node)
            zii = zs_dof_r0[ii] # z coord of DOF (node)
            
            #index for v_init.vector() (two elements for each one in rho.vector(), vx and vy)
            jjx=int(2*ii)
            jjy=int(2*ii + 1)
            v_init.vector()[jjx]=0.0 #because no horizontal velocities in the initial guess
            
            #---------------all nodes same initial temperature for now
            T_init.vector()[ii]= Tsite

            deltarho_trench= rho_trench - f_rho_neem(zii) #to smooth out the transition
        
             
            if ((xii>= (L/2-l_trench) ) and (xii<= (L/2+l_trench) ) and (zii>=hole_zmin)):
                
                T_init.vector()[ii]=T0hotter
                r_init.vector()[ii]= rho_trench

                zref= H-zii #how many meters from surface
                zbase=hole_zmin #trench base depth
                                      
                #intial guess approximation
                v_init.vector()[jjy]= factor_inner*(f_v_neem(zbase)-f_v_neem(0)) \
                                      + f_v_trench(H-bump-zref/trench_zoom_factor)\
                                      - f_v_trench(H-bump-zbase/trench_zoom_factor) 
                                      #because we want the relative (what needs to be added)
                                      
            
            elif ((xii>= (L/2-u_trench) ) and (xii<= (L/2+u_trench) ) and (zii>=zmin-trench_lim)):
                
                T_init.vector()[ii]=T0hotter
                r_init.vector()[ii]= rho_trench
                
                zref= H-zii #how many meters from surface
                zbase=zmin-trench_lim #trench base depth
                
                #intial guess approximation
                v_init.vector()[jjy]= factor_outer*(f_v_neem(zbase)-f_v_neem(0)) \
                                      + f_v_trench(H-bump-zref/trench_zoom_factor)\
                                      - f_v_trench(H-bump-zbase/trench_zoom_factor) 
                                      #because we want the relative (what needs to be added)
            
            elif ((xii>= (L/2-u_trench-bump) ) and (xii<= (L/2+u_trench+bump) ) and (zii>=zmin)):
                
                T_init.vector()[ii]=T0hotter
                r_init.vector()[ii]= rho_trench 

                zref= H-zii #how many meters from surface
                zbase=zmin #trench base depth
                
                #intial guess approximation
                v_init.vector()[jjy]= 1*(f_v_neem(zbase)-f_v_neem(0)) \
                                      + f_v_trench(H-bump-zref/trench_zoom_factor)\
                                      - f_v_trench(H-bump-zbase/trench_zoom_factor) 
                                      #because we want the relative (what needs to be added)
                
           #----------------upper trench
            elif ((xii>= (L/2-u_trench - deltax_trench) ) and (xii<= (L/2-u_trench) ) and (zii>=zmin-trench_lim) and (zii<=zmin-deltaz_trench)):
                
                T_init.vector()[ii]=T0hotter
                
                r_init.vector()[ii]= f_rho_neem(zii) + ((xii-(L/2-u_trench - deltax_trench))/deltax_trench) *deltarho_trench  #linear transition for now

                v_init.vector()[jjy]= 1*(f_v_neem(zii)-f_v_neem(0)) 
                
            elif ((xii>= (L/2+u_trench) ) and (xii<= (L/2+u_trench + deltax_trench) ) and (zii>=zmin-trench_lim) and (zii<=zmin-deltaz_trench)):
                
                T_init.vector()[ii]=T0hotter
                
                r_init.vector()[ii]= f_rho_neem(zii) - ((xii-(L/2+u_trench + deltax_trench))/deltax_trench) *deltarho_trench  #linear transition for now

                v_init.vector()[jjy]= 1*(f_v_neem(zii)-f_v_neem(0)) 
                
           #---------------------- lower trencha----------------------------------------------------------------------------
            elif ((xii>= (L/2-l_trench - deltax_trench) ) and (xii<= (L/2-l_trench) ) and (zii>=hole_zmin) and (zii<=zmin-trench_lim-deltaz_trench)):
                
                T_init.vector()[ii]=T0hotter
                
                r_init.vector()[ii]= f_rho_neem(zii) + ((xii-(L/2-l_trench - deltax_trench))/deltax_trench) *deltarho_trench  #linear transition for now

                v_init.vector()[jjy]= factor_outer*(f_v_neem(zii)-f_v_neem(0)) 
                
            elif ((xii>= (L/2+l_trench) ) and (xii<= (L/2+l_trench + deltax_trench) ) and (zii>=hole_zmin) and (zii<=zmin-trench_lim-deltaz_trench)):
                
                T_init.vector()[ii]=T0hotter
                
                r_init.vector()[ii]= f_rho_neem(zii) - ((xii-(L/2+l_trench + deltax_trench))/deltax_trench) *deltarho_trench  #linear transition for now

                v_init.vector()[jjy]= factor_outer*(f_v_neem(zii)-f_v_neem(0)) 
                
            #-------------------------------------------------------------------------------------------------------
            
            elif ((xii>= (L/2-u_trench-bump) ) and (xii<= (L/2-u_trench-(bump-deltax_trench)) ) and (zii>=zmin-deltaz_trench) and (zii<=zmin)):
                
                T_init.vector()[ii]=T0hotter
                
                r_init.vector()[ii]= f_rho_neem(zii) + ((zii-(zmin - deltaz_trench))/deltaz_trench) *deltarho_trench

                v_init.vector()[jjy]= 1*(f_v_neem(zii)-f_v_neem(0)) 
                
            elif ((xii>= (L/2+u_trench+(bump-deltax_trench)) ) and (xii<= (L/2+u_trench+bump) ) and (zii>=zmin-deltaz_trench) and (zii<=zmin)):
                
                T_init.vector()[ii]=T0hotter
                
                r_init.vector()[ii]= f_rho_neem(zii) + ((zii-(zmin - deltaz_trench))/deltaz_trench) *deltarho_trench

                v_init.vector()[jjy]= 1*(f_v_neem(zii)-f_v_neem(0)) 
                
            #-------------------------------------------------------------------------------------------------------
            
            elif ((xii>= (L/2-u_trench) ) and (xii<= (L/2-l_trench - deltax_trench) ) and (zii>=zmin-trench_lim-deltaz_trench) and (zii<=zmin-trench_lim)):
                
                T_init.vector()[ii]=T0hotter
                
                r_init.vector()[ii]= f_rho_neem(zii) + ((zii-(zmin-trench_lim - deltaz_trench))/deltaz_trench) *deltarho_trench

                v_init.vector()[jjy]= factor_outer*(f_v_neem(zii)-f_v_neem(0)) 
                
            elif ((xii>= (L/2+l_trench+deltax_trench) ) and (xii<= (L/2+u_trench) )  and (zii>=zmin-trench_lim-deltaz_trench) and (zii<=zmin-trench_lim)):
                
                T_init.vector()[ii]=T0hotter
                
                r_init.vector()[ii]= f_rho_neem(zii) + ((zii-(zmin-trench_lim - deltaz_trench))/deltaz_trench) *deltarho_trench

                v_init.vector()[jjy]= factor_outer*(f_v_neem(zii)-f_v_neem(0)) 
                
            #------------------------------------------------------------------------
            
            elif ((xii>= (L/2-l_trench) ) and (xii<= (L/2+l_trench) ) and (zii>=hole_zmin-deltaz_trench) and (zii<=hole_zmin)):
                
                T_init.vector()[ii]=T0hotter
                
                r_init.vector()[ii]= f_rho_neem(zii) + ((zii-(hole_zmin - deltaz_trench))/deltaz_trench) *deltarho_trench

                v_init.vector()[jjy]= factor_inner*(f_v_neem(zii)-f_v_neem(0)) 
                
            #----------------------------------------------------------------------
            #lower corners
            
            elif ((xii>= (L/2-l_trench-deltax_trench) ) and (xii<= (L/2-l_trench) ) and (zii>=hole_zmin-deltaz_trench) and (zii<=hole_zmin)):
                
                T_init.vector()[ii]=T0hotter
                
                r_init.vector()[ii]= min( f_rho_neem(zii) + ((xii-(L/2-l_trench - deltax_trench))/deltax_trench) *deltarho_trench,
                               f_rho_neem(zii) + ((zii-(hole_zmin - deltaz_trench))/deltaz_trench) *deltarho_trench)

                v_init.vector()[jjy]= factor_outer*(f_v_neem(zii)-f_v_neem(0)) 
                
            elif ((xii>= (L/2+l_trench) ) and (xii<= (L/2+l_trench+deltax_trench) ) and (zii>=hole_zmin-deltaz_trench) and (zii<=hole_zmin)):
                
                T_init.vector()[ii]=T0hotter
                
                r_init.vector()[ii]= min( f_rho_neem(zii) - ((xii-(L/2+l_trench + deltax_trench))/deltax_trench) *deltarho_trench,
                               f_rho_neem(zii) + ((zii-(hole_zmin - deltaz_trench))/deltaz_trench) *deltarho_trench)

                v_init.vector()[jjy]= factor_outer*(f_v_neem(zii)-f_v_neem(0)) 
                
            #----------------------------------------------------------------------
            #middle inner corners
            
            elif ((xii>= (L/2-l_trench-deltax_trench) ) and (xii<= (L/2-l_trench) ) and (zii>=zmin-trench_lim-deltaz_trench) and (zii<=zmin-trench_lim)):
                
                T_init.vector()[ii]=T0hotter
                
                r_init.vector()[ii]= max( f_rho_neem(zii) + ((xii-(L/2-l_trench - deltax_trench))/deltax_trench) *deltarho_trench,
                               f_rho_neem(zii) + ((zii-(zmin-trench_lim - deltaz_trench))/deltaz_trench) *deltarho_trench)

                v_init.vector()[jjy]= factor_outer*(f_v_neem(zii)-f_v_neem(0)) 
                
            elif ((xii>= (L/2+l_trench) ) and (xii<= (L/2+l_trench+deltax_trench) ) and (zii>=zmin-trench_lim-deltaz_trench) and (zii<=zmin-trench_lim)):
                
                T_init.vector()[ii]=T0hotter
                
                r_init.vector()[ii]= max( f_rho_neem(zii) - ((xii-(L/2+l_trench + deltax_trench))/deltax_trench) *deltarho_trench,
                               f_rho_neem(zii) + ((zii-(zmin-trench_lim - deltaz_trench))/deltaz_trench) *deltarho_trench)

                v_init.vector()[jjy]= factor_outer*(f_v_neem(zii)-f_v_neem(0)) 
                
            #----------------------------------------------------------------------
            #middle outer corners
            
            elif ((xii>= (L/2-u_trench-deltax_trench) ) and (xii<= (L/2-u_trench) ) and (zii>=zmin-trench_lim-deltaz_trench) and (zii<=zmin-trench_lim)):
                
                T_init.vector()[ii]=T0hotter
                
                r_init.vector()[ii]= min( f_rho_neem(zii) + ((xii-(L/2-u_trench - deltax_trench))/deltax_trench) *deltarho_trench,
                               f_rho_neem(zii) + ((zii-(zmin-trench_lim - deltaz_trench))/deltaz_trench) *deltarho_trench)

                v_init.vector()[jjy]= 1*(f_v_neem(zii)-f_v_neem(0)) 
                
            elif ((xii>= (L/2+u_trench) ) and (xii<= (L/2+u_trench+deltax_trench) ) and (zii>=zmin-trench_lim-deltaz_trench) and (zii<=zmin-trench_lim)):
                
                T_init.vector()[ii]=T0hotter
                
                r_init.vector()[ii]= min( f_rho_neem(zii) - ((xii-(L/2+u_trench + deltax_trench))/deltax_trench) *deltarho_trench,
                               f_rho_neem(zii) + ((zii-(zmin-trench_lim - deltaz_trench))/deltaz_trench) *deltarho_trench)

                v_init.vector()[jjy]= 1*(f_v_neem(zii)-f_v_neem(0)) 
                
            #----------------------------------------------------------------------
            #upper corners
            
            elif ((xii>= (L/2-u_trench-deltax_trench) ) and (xii<= (L/2-u_trench) ) and (zii>=zmin-deltaz_trench) and (zii<=zmin)):
                
                T_init.vector()[ii]=T0hotter
                
                r_init.vector()[ii]= max( f_rho_neem(zii) + ((xii-(L/2-u_trench - deltax_trench))/deltax_trench) *deltarho_trench,
                               f_rho_neem(zii) + ((zii-(zmin - deltaz_trench))/deltaz_trench) *deltarho_trench)

                v_init.vector()[jjy]= 1*(f_v_neem(zii)-f_v_neem(0)) 
                
            elif ((xii>= (L/2+u_trench) ) and (xii<= (L/2+u_trench+deltax_trench) ) and (zii>=zmin-deltaz_trench) and (zii<=zmin)):
                
                T_init.vector()[ii]=T0hotter
                
                r_init.vector()[ii]= max( f_rho_neem(zii) - ((xii-(L/2+u_trench + deltax_trench))/deltax_trench) *deltarho_trench,
                               f_rho_neem(zii) + ((zii-(zmin - deltaz_trench))/deltaz_trench) *deltarho_trench)

                v_init.vector()[jjy]= 1*(f_v_neem(zii)-f_v_neem(0)) 
                
            #--------------------------------------------------------------------
            #from here on, they only are combinations of NEEM's profiles

            else:
                
                r_init.vector()[ii]=f_rho_neem(zii) #they all have the same RAW density profile, only velocity changes
                
                #vertical dimension already filtered in the previous cases
                #middle-----------------------------------
                if ((xii>= (L/2-l_trench) ) and (xii<= (L/2+l_trench) )):
                    
                    v_init.vector()[jjy]= factor_inner*(f_v_neem(zii)-f_v_neem(0))
                
                
                #sum them and smooth the transition later---
                elif ((xii>= (L/2-u_trench-deltax_trench) ) and (xii<= (L/2-l_trench))):
                    
                    v_init.vector()[jjy]= factor_outer*(f_v_neem(zii)-f_v_neem(0))
                    
                #normal outer
                else:
                    
                    v_init.vector()[jjy]= 1*(f_v_neem(zii)-f_v_neem(0))
                    

        # #--------------------------------------    
        
        rho.assign(r_init)
        rho_prev=Function(U)
        rho_prev.assign(r_init)
        
        v.assign(v_init)
        
        T.assign(T_init)
        T_prev=Function(Q)
        T_prev.assign(T_init)

        # ####################################plotting initial coditions

        plt.rcParams["font.family"] = "DejaVu Serif"
        plt.rcParams["font.serif"] = ["Times New Roman"]
        plt.rcParams.update({'font.size': 16})
    
        # --------------------Densities______________
        
        fig,ax=plt.subplots(figsize=(7,7),dpi=300)
        rhoplot = plot(rho_prev,cmap='PuBu',vmin=320,vmax=640, extend='both')
        clb = plt.colorbar(rhoplot, orientation="vertical",label=r'$\rho$ (kgm$^{-3}$)', extend='both')
        clb.set_label(r'$\rho$ (kg m$^{-3}$)', rotation=90, labelpad=10,family='sans serif')
        clb.set_ticks([400,450,500,550,600])
        #add min ticks -----------------------
        ntick=4
        ax.tick_params(axis="both",which='both',direction="out")
        minor_locator_x = AutoMinorLocator(ntick)
        ax.xaxis.set_minor_locator(minor_locator_x)
        minor_locator_y = AutoMinorLocator(ntick)
        ax.yaxis.set_minor_locator(minor_locator_y)
        # -------------------------------------
        positions = [6,8,10,12,14]
        labels = [-4,-2,0,2,4]
        plt.xticks(positions, labels)
        positions = [19,21,23,25,27,29]
        labels = [-10,-8,-6,-4,-2,0]
        plt.yticks(positions, labels)
        #----------------------------
        ax.set_title('Initial condition',family='sans serif')
        ax.set_ylim(20,30)
        ax.set_xlim(5,15)
        ax.set_xlabel(r'$x$ (m)',labelpad=10,family='sans serif')
        ax.set_ylabel(r'$z$ (m)',labelpad=15,family='sans serif')
        # save frame
        plt.savefig('modeled_initial_condition_profiles/initial_densities_smooth.png', dpi=500, format='png')
        plt.close()

        #---------------------------------------------velocities
        
        fig,ax=plt.subplots(figsize=(7,7),dpi=300)
        vplot = plot(v*yr2s,cmap='jet')
        clb = plt.colorbar(vplot, orientation="vertical",label=r'Velocity (m/a)',extend='both')
        ax.set_title('Initial condition')
        ax.set_ylim(20,30)
        ax.set_xlim(5,15)
        ax.set_xlabel(r'$x$ (m)')
        ax.set_ylabel(r'$z$ (m)')
        # save frame
        plt.savefig('modeled_initial_condition_profiles/initial_velocities.png', dpi=300, format='png')
        plt.close()

        #------------------------------temperature    

        plt.figure()
        rhoplot = plot(T_init-273.15,cmap='jet',vmin=-60,vmax=0)
        # plot(mesh,lw=0.25)
        clb = plt.colorbar(rhoplot, orientation="vertical", label='T_prev (ºC)',extend='both')
        plt.title(f'{K=} t={np.round(tstep*dt/yr2s,2)}yr  dt={np.round(dt/(60*60*24),2)}d---{tmr=}tunnel',fontsize=6)
        # create file name and append it to a list
        # save frame
        plt.savefig('modeled_initial_condition_profiles/initial_T.png', dpi=300, format='png')
        plt.close()

        print('\n \n -----------------------------initial definition and plotting finished------\n\n')
    
    else:        
            
        rho_prev.set_allow_extrapolation(True)       
        v_sol.set_allow_extrapolation(True)
        T_prev.set_allow_extrapolation(True)    
        
        rho_init = interpolate(rho_prev,U)
        v_init = interpolate(v_sol,V)
        T_init = interpolate(T_prev,Q)
        
        rho.assign(rho_init)
        v.assign(v_init)
        T.assign(T_init)
            
    
    #--------------------------------------INTERPOLATE RHO------------------------------------------------#
    
    if tstep >0: 
        
        #:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::update density

        #we need to handle the fact that U is now a new and shorter space

        rho_old=rho_prev.copy()
        rho_old.set_allow_extrapolation(True)

        rho_new=Function(U)
        LagrangeInterpolator.interpolate(rho_new,rho_old)

        rho_prev.assign(rho_new)
        rho_prev.set_allow_extrapolation(True)            
        
        #................................KEEP IT BELOW RHO_ICE...........................#
        
        rhovec = rho_prev.vector()[:]
        rhovec[rhovec > rho_ice_softened] = rho_ice_softened
        rho_prev.vector()[:] = rhovec
        
        #:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::update temperature too
        #----------------------------------temperature
        T_old=T_prev.copy()
        T_old.set_allow_extrapolation(True)

        T_new=Function(Q)
        LagrangeInterpolator.interpolate(T_new,T_old)

        T_prev.assign(T_new)
        T_prev.set_allow_extrapolation(True)
        
        #.........................................IMPOSE SNOW
        
        if acc_iter:
               
            r_snowed=Function(U)
            r_snowed = interpolate(rho_prev,U)
            
            #----------------------save rho_snow on top of previous solution if closer than deltah from the surface
            #(surface identified in remesh_acc)
            
            #surface profile (including deltah)
            #WATCH OUT, LAST ELEMENT OF THE LIST IS DUPLICATED DUE TO PERIODIC BOUNDARY CONDITIONS!!-->REMOVED
            f_snowed_surface=interp1d(x_snowed_surf[:-1],z_snowed_surf[:-1],kind='cubic',fill_value='extrapolate')
            
            
            scoords_r0 = U.tabulate_dof_coordinates()
            xs_dof_r0 = scoords_r0[:,0] # x coords of func space NODES
            zs_dof_r0 = scoords_r0[:,1] # z coords of func space NODES
            s_numdofs_r0 = len(zs_dof_r0)
            ISz_r0 = np.argsort(zs_dof_r0)  
            
            
            for ii in range(s_numdofs_r0):
                        
                xii = xs_dof_r0[ii]# x coord of DOF (node)
                zii = zs_dof_r0[ii] # z coord of DOF (node)
                
                snowed= (zii > (f_snowed_surface(xii)-deltah))
                
                if snowed:
                    if snow550:
                        r_snowed.vector()[ii]= rho_trench
                    else:
                        r_snowed.vector()[ii]= rho_surf

            rho_prev = interpolate(r_snowed,U)


        #-----------------------------------------------test roof overload
        Nsamples=1000
        x_section=10
        min_undershoot=24
        maxz_overshoot=H+5 #to make sure that we take the hole column no matter how much it accumulates
        zs_section=np.linspace(min_undershoot,maxz_overshoot,Nsamples)
        rhos_section=np.zeros(Nsamples)
        side_rhos=np.zeros(Nsamples)
        roof_rhos=np.zeros(Nsamples)
        intunnel=False

        rho_prev.set_allow_extrapolation(False)

        print('                                      before assigning rho_crosssections')

        for kk in range(Nsamples):

            z_section=zs_section[kk]

            #roof------------
            try:
                rhos_section[kk]=rho_prev(x_section,z_section)
                if intunnel:  #it has entered the tunnel and has real values again-->roof
                    roof_rhos[kk]=rhos_section[kk]
            except:
                rhos_section[kk]=np.nan
                if (z_section>1 and (not intunnel)): #assuming zs start from 0 bottom
                    intunnel=True

            #side----------

            try:
                side_rhos[kk]=rho_prev(8,z_section)
            except:
                side_rhos[kk]=np.nan


        #just to check an error in the fenics sampling where it evaluates a non-zero point as null
        for ijk in range(1,len(roof_rhos)-1):

            if roof_rhos[ijk]==0:
                if (roof_rhos[ijk-1]>0 and roof_rhos[ijk+1]>0):
                    roof_rhos[ijk]=(roof_rhos[ijk-1] + roof_rhos[ijk+1])/2
                elif (roof_rhos[ijk-1]>0 and roof_rhos[ijk+2]>0):
                    roof_rhos[ijk]=(roof_rhos[ijk-1] + roof_rhos[ijk+2])/2
                elif (roof_rhos[ijk-1]>0 and roof_rhos[ijk+3]>0):
                    roof_rhos[ijk]=(roof_rhos[ijk-1] + roof_rhos[ijk+3])/2
                elif (roof_rhos[ijk-1]>0 and roof_rhos[ijk+4]>0):
                    roof_rhos[ijk]=(roof_rhos[ijk-1] + roof_rhos[ijk+3])/2
                elif (roof_rhos[ijk-1]>0 and roof_rhos[ijk+5]>0):
                    roof_rhos[ijk]=(roof_rhos[ijk-1] + roof_rhos[ijk+3])/2

        print('                                      after assigning rho_crosssections')

        rho_prev.set_allow_extrapolation(True)
        dz_int=(maxz_overshoot-min_undershoot)/Nsamples
        roof_int_estimate=dz_int*np.sum(roof_rhos) #because it's 0s elsewhere

        roofkgs[tstep-1]=roof_int_estimate

        if acc_iter:
            if snow550:
                theoricload[tstep-1:]+=deltah*rho_trench
            else:
                theoricload[tstep-1:]+=deltah*rho_surf

        #-------------------------------------------------------------------
        if (plotting and tstep%nplot==0):

            plt.figure()
            plt.axvline(550,lw=1,c='k',ls='--')
            plt.plot(side_rhos,zs_section,ls='--')
            plt.plot(roof_rhos,zs_section)
            plt.text(400,maxz_overshoot+0.5,f'{roof_int_estimate=} kg/m^2 (along center line)')
            plt.xlim(300,600)
            plt.ylim(26,maxz_overshoot)
            plt.title(f'{tstep=}   {tstep%n_itacc=}   {acc_iter=} ')
            plt.savefig(f'./spaces/section_ZOOM{tstep}.png',format='png',dpi=150)
            plt.close()

            plt.figure()
            plt.plot(theoricload[:tstep],ls='--')
            plt.plot(roofkgs[:tstep])
            # plt.ylim(800,1500)
            plt.xlim(0,tstep)
            plt.savefig(f'./spaces/roofkgs.png',format='png',dpi=150)
            plt.close()



    #____________________________________________________
    rho_prev=project(rho_prev,U)
    T_prev=project(T_prev,Q)
    
    #!!!#############################################################################################
    #----terminal logging
    print(tstep,'/',nsteps,'--------------------------------------------t=',tstep*dt/yr2s,' years')
    print(f'                                           T_average_step={avg_Tstep[tstep]}')
    print(f'                                         {acc_iter=}')
    try:
        print(f'                                     last tunnel height= {np.round(hole_deltazs[-1],3)}m')
    except:
        print('there is still no hole_deltazs defined')

    #-----------------------------------------------------------------------------------------------------#
    #--------------------------------------SOLVE FEM PROBLEM----------------------------------------------#
    #-----------------------------------------------------------------------------------------------------# 
    
    #--------------------------------GET a, b, VARIABLES AND SIGMA-----------------------------------------
    a_,b_=get_ab_Z07(rho_prev,rho_ice,phi_snow,ab_phi_lim,n,K)

    Aglen=A_glen_Arrhenius(T_prev)

    sigma=get_sigma(v,a_,b_,Aglen,n)

    print('                                      computed a,b,A(T) and sigma')

    #-----------------------------------SOLVE MOMENTUM EQUATION--------------------------------------------
    a_v = inner(sigma,grad(wv))*dx
    L_v = rho_prev*inner(g,wv)*dx 
    
    F_v = a_v - L_v

    tol, relax, maxiter = 1e-2, 0.35, 100
    solparams = {"newton_solver":  {"relaxation_parameter":relax, "relative_tolerance": tol, "maximum_iterations":maxiter} }
    solve(F_v==0, v, bcs_v, solver_parameters=solparams)
    
    v_sol=project(v,V)
    print('                                      solved momentum equation')

    #---------------------------------SOLVE MASS BALANCE EQUATION------------------------------------------

    alpha_diff=Constant(1e-9) #FORK=1000 factor for the diffusive term. To be adjusted. 
    
    a_rho = Constant(1/dt)*rho*wr*dx + 0.5*rho*div(v_sol)*wr*dx + 0.5*dot(v_sol,grad(rho))*wr*dx + alpha_diff*dot(grad(rho),grad(wr))*dx
    L_rho =  Constant(1/dt)*rho_prev * wr *dx - 0.5*rho_prev*div(v_sol)*wr*dx - 0.5*dot(v_sol,grad(rho_prev))*wr*dx - alpha_diff*dot(grad(rho_prev),grad(wr))*dx
    
    F_rho = a_rho - L_rho

    tol, relax, maxiter = 1e-2, 0.35, 100
    solparams = {"newton_solver":  {"relaxation_parameter":relax, "relative_tolerance": tol, "maximum_iterations":maxiter} }
    solve(F_rho==0, rho, bcs_rho, solver_parameters=solparams)
    
    rho_prev.assign(rho)  #<-------------UPDATE RHO PROFILE
    print('                                      solved mass balance')

    #-----------------------------------SOLVE HEAT EQUATION--------------------------------------------

    #-----------------------
    a_,b_=get_ab_Z07(rho_prev,rho_ice,phi_snow,ab_phi_lim,n,K)
    sigma=get_sigma(v_sol,a_,b_,Aglen,n)
    
    eps=sym(grad(v_sol))
    ssol=inner(sigma,eps)
    
    #-------------------------------------------
    
    cT=c0+c1*(T-T0)
    kT=(kf0-kf1*rho_prev+kf2*rho_prev**2)/(kf0-kf1*rho_ice+kf2*rho_ice**2) * k0*exp(-gammaT*T)

    #--------------------------------
    a_T = T*wT*dx + dt*kT/(rho*cT)*inner(grad(T),grad(wT))*dx + dt*dot(v_sol,grad(T))*wT*dx     
    L_T = T_prev*wT*dx + dt/(rho*cT)*ssol*wT*dx
    
    F_T = a_T - L_T

    tol, relax, maxiter = 1e-2, 0.35, 100
    solparams = {"newton_solver":  {"relaxation_parameter":relax, "relative_tolerance": tol, "maximum_iterations":maxiter} }
    solve(F_T==0,T, bcs_T, solver_parameters=solparams)
    
    T_prev.assign(T)  #<-------------UPDATE T PROFILE
    print('                                      solved heat equation')

    #-----------------------------------------------------------------------------------------------------#
    #--------------------------------------------EVOLVE MESH----------------------------------------------#
    #-----------------------------------------------------------------------------------------------------#
    
    #Get coordinates and velocities of the points at the obstacle's surface assemble
    mesh.init(2,1)
    dofs = []
    cell_to_facets = mesh.topology()(2,1)
    for cell in cells(mesh):
        facets = cell_to_facets(cell.index())
        for facet in facets:
            if boundary_subdomains[facet] == 2: #We have given the number 5 to the subdomain of the obstacle
                dofs_ = V.dofmap().entity_closure_dofs(mesh, 1, [facet])
                for dof in dofs_:
                    dofs.append(dof)
    
    unique_dofs = np.array(list(set(dofs)), dtype=np.int32)
    boundary_coords = V.tabulate_dof_coordinates()[unique_dofs]
    zmin=np.min(boundary_coords[:,1])

    #-------------------
    disp=Function(V) #displacement
    disp.assign(v_sol*dt)
    ALE.move(mesh, disp)
    print('                                      evolved mesh')
    
    ###################################################################################################
    ############################################ PLOT #################################################
    ###################################################################################################
    
    plotYlimMIN= H - 10
    plotYlimMAX = H

    if (tstep % nplot == 0 or (tstep*dt/yr2s>=1.96 and neemsaved==False) or tstep==(nsteps-1)):
            
        if (tstep*dt/yr2s>=1.96 and neemsaved==False):
            igifsold=igifs
            igifs='_1_96yrfinal'
            neemsaved=True

        #----set paper fonts
        plt.rcParams["font.family"] = "DejaVu Serif"
        plt.rcParams["font.serif"] = ["Times New Roman"]
        plt.rcParams.update({'font.size': 18})
    

        #-------------------------------LOAD
        plt.figure()
        plt.plot(theoricload[:tstep],ls='--')
        plt.plot(roofkgs[:tstep])
        plt.xlim(0,tstep)
        plt.savefig(f'./spaces/roofkgs.png',format='png',dpi=150)         
        plt.close()

        #-------------------- DENSITY zoom on tunnel

        fig,ax=plt.subplots(figsize=(7,7),dpi=200)

        rhoplot = plot(rho_prev,cmap='PuBu',vmin=200,vmax=650, extend='both')
        clb = plt.colorbar(rhoplot, orientation="vertical",label=r'$\rho$ (kgm$^{-3}$)',extend='both')
        clb.set_label(r'$\rho$ (kgm$^{-3}$)', rotation=90, labelpad=10)
        clb.set_ticks([400,450,500,550,600])

        #min ticks-----------------------
        ntick=4 #how many subranges
        ax.tick_params(axis="both",which='both',direction="out")
        minor_locator_x = AutoMinorLocator(ntick)
        ax.xaxis.set_minor_locator(minor_locator_x)
        minor_locator_y = AutoMinorLocator(ntick)
        ax.yaxis.set_minor_locator(minor_locator_y)
        # -------------------------------------
        positions = [6,8,10,12,14]
        labels = [-4,-2,0,2,4]
        plt.xticks(positions, labels)
        
        positions = [19,21,23,25,27,29]
        labels = [-10,-8,-6,-4,-2,0]
        plt.yticks(positions, labels)
        #----------------------------
        ax.set_title(r'$k=$'+str(K))

        # create file name and append it to a list
        filename_rho_paper = 'figures'+str(K)+f'/paper_density{igifs}original.png'
        
        ax.set_ylim(plotYlimMIN,plotYlimMAX)
        ax.set_xlim(5,15)
        
        ax.set_xlabel(r'$x$ (m)',labelpad=10)
        ax.set_ylabel(r'$z$ (m)',labelpad=15)

        # save frame
        plt.savefig(filename_rho_paper, dpi=200, format='png')
        print(filename_rho_paper)
        plt.close()

        #..................-----------VELOCITY zoom on tunnel

        plt.figure()
        vplot = plot(v_sol*yr2s,cmap='jet')
        clb = plt.colorbar(vplot, orientation="vertical", label='V (m/yr)',extend='both')
        plt.title(' k='+str(K)+' '+' ('+str("%.2f"%(tstep*dt/yr2s))+'yr)')
        # create file name and append it to a list
        filename_vel = 'figures'+str(K)+f'/velocityZOOM{igifs}.png'
        filenames_vel.append(filename_vel)
        plt.ylim(plotYlimMIN,plotYlimMAX)
        plt.xlim(10,20) 
        # save frame
        plt.savefig(filename_vel, dpi=200, format='png')
        plt.close()

        #.................. ................ full MESH

        plt.figure()
        plot(mesh, linewidth=0.25)
        plt.title(' k='+str(K)+' '+' ('+str("%.2f"%(tstep*dt/yr2s))+'yr)')
        # create file name and append it to a list
        filename_mesh = 'figures'+str(K)+f'/mesh{igifs}.png'
        filenames_mesh.append(filename_mesh)
        plt.ylim(plotYlimMIN,plotYlimMAX)
        # save frame
        plt.savefig(filename_mesh, dpi=200, format='png')
        plt.close()


        #..................   MESH zoom on tunnel

        plt.figure()
        plot(mesh, linewidth=0.25)
        plt.title(' k='+str(K)+' '+' ('+str("%.2f"%(tstep*dt/yr2s))+'yr)')

        # create file name and append it to a list
        filename_mesh = 'figures'+str(K)+f'/meshZOOM{igifs}.png'
        plt.ylim(29,30)
        plt.xlim(5,15)
        # save frame
        plt.savefig(filename_mesh, dpi=200, format='png')
        plt.close()

        #------------------------------   TEMPERATURE zoom on tunnel

        plt.figure()
        rhoplot = plot(T_prev-273.15,cmap='jet',vmin=-60,vmax=0)
        plot(mesh,lw=0.25)
        clb = plt.colorbar(rhoplot, orientation="vertical", label='T_prev (ºC)',extend='both')
        plt.title(f'{K=} t={np.round(tstep*dt/yr2s,2)}yr  dt={np.round(dt/(60*60*24),2)}d---{tmr=}tunnel',fontsize=6)
        # create file name and append it to a list
        plt.ylim(plotYlimMIN,plotYlimMAX)
        plt.xlim(10,20) 
        # save frame
        plt.savefig('figures'+str(K)+f'/Tzoom{igifs}.png', dpi=200, format='png')
        plt.close()

        #----------------------------------------tunnel collapse
        
        plt.figure()
        plt.plot(times,hole_deltaxs,label='hole_deltaXs')
        plt.plot(times,hole_deltazs,label='hole_deltaZs')
        plt.axhline(4.52,ls='--',c='tab:blue',label='1.96yr reference width')
        plt.axhline(3.79,ls='--',c='tab:orange',label='1.96yr reference height')
        plt.ylabel('max dimension (m)')
        plt.xlabel('evolution time (yr)')
        plt.title(f'{K=} t={np.round(tstep*dt/yr2s,2)}yr  dt={np.round(dt/(60*60*24),2)}d---{tmr=}tunnel',fontsize=6)
        plt.legend()
        plt.savefig('figures'+str(K)+'/hole_dimensions_COUPLED_evolution.png',dpi=200)
        plt.close()


        #---------------------------------------------------------update counter
        
        if isinstance(igifs, str):
            igifs=igifsold

        igifs += 1
        
        #------------------------------------to avoid matplotlib related memory leaks
        # Clear the current axes.
        plt.cla() 
        # Clear the current figure. 
        plt.clf() 
        # Closes all the figure windows. 
        plt.close('all') 
        plt.close(fig) 
        # gc.collect()

        
    #--------------------------------------------------------------

    if (tstep*dt/yr2s>=1.96 and neemsaved==True):
        print('FINAL_____________SAVED_______BREAKING')
        break

    tstep += 1
    
    acc_iter=False

#------------------------------------saving arrays
np.save('figures'+str(K)+'/hole_times_k'+str(K)+'.npy',times)
np.save('figures'+str(K)+'/hole_zmins_COUPLED_k'+str(K)+'.npy',hole_zmins)
np.save('figures'+str(K)+'/hole_deltaxs_COUPLED_k'+str(K)+'.npy',hole_deltaxs)
np.save('figures'+str(K)+'/hole_deltazs_COUPLED_k'+str(K)+'.npy',hole_deltazs)


#------------------------------------plotting
plt.figure()
plt.plot(times,hole_deltaxs,label='hole_deltaXs')
plt.plot(times,hole_deltazs,label='hole_deltaZs')
plt.axhline(4.52,ls='--',c='tab:blue',label='1.96yr reference width')
plt.axhline(3.79,ls='--',c='tab:orange',label='1.96yr reference height')
plt.ylabel('max dimension (m)')
plt.xlabel('evolution time (yr)')
plt.legend()
plt.savefig('figures'+str(K)+'/hole_dimensions_COUPLED_evolution.png',dpi=300)




