
These codes are part of the supplementary materials of the *Journal of Glaciology* article titled:



# Firn densification in two dimensions: modelling the collapse of snow caves and enhanced densification in ice-stream shear margins

Arrizabalaga-Iriarte J, Lejonagoitia-Garmendia L, Hvidberg CS, Grinsted A, Rathmann NM 

> ###  Abstract:
> 
> Accurate modelling of firn densification is necessary for ice-core interpretation and assessing the mass balance of glaciers and ice sheets. In this paper, we revisit the nonlinear-viscous firn rheology introduced by Gagliardini and Meyssonnier (1997) that allows posing multi-dimensional firn densification problems subject to arbitrary stress and temperature fields. First, we extend the calibration of the coefficient functions that control firn compressibility and viscosity to 5 additional Greenlandic sites, showing that the original calibration is not universally valid. Next, we demonstrate that the transient collapse of a Greenlandic firn tunnel can be reproduced in a cross-section model, but that anomalous warm summer temperatures during 2012--2014 reduce confidence in attempts to independently validate the rheology. Finally, we show that the rheology can explain the increased densification rate and varying bubble close-off depth observed across the shear margins of the North-East Greenland Ice Stream. Although we suggest more work is needed to constrain the model’s near-surface compressibility and viscosity functions, our results strengthen the rheology's empirical grounding for future use, such as modelling horizontal firn density variations over ice sheets for mass-loss estimates or estimating ice–gas age differences in ice cores subject to complex strain histories.

---

In the numerical experiments that follow, we solved the coupled density, momentum, and thermal problem using *FEniCS* (Logg and others, 2012), relying on Newton's method to solve nonlinearities. For reasons explained in the article, the ice-stream scenario is not thermally coupled, but the mechanical problem is solved using the same method. The Jacobian of the residual forms (required for Newton iterations) were calculated using the unified form language (UFL) (Alnaes and others, 2015), used by *FEniCS* to specify weak forms of PDEs, which supports automatic symbolic differentiation. All weak forms are presented in Appendix A. For our two-dimensional experiments, meshes were constructed using *gmsh* (Geuzaine and Remacle, 2009) and updated between time-steps to evolve both interior and exterior free-surface boundary.


All the datasets used as reference and initial and boundary conditions are publicly available, but they need to be downloaded from their owners and properly pathed before running these codes.

- 1D Firn core density profiles -----> Bréant and others (2017)
- NEEM firn tunnel initial dimensions -----> technical report, Steffensen (2014), but we provide the initial condition mesh we used
- NEEM 2012--2014 temperature record -----> GCnet, Vandecrux and others (2023)
- NEGIS transect velocity field (for computing the strain-rate profile) ---> MEaSUREs program, Howat (2020)


The codes also expect a particular folder structure (mostly to read the data from an input folder or save the results into an output one) so, if any path issues arise, either create the missing folder (the easiest) or repath it for your purpose.
