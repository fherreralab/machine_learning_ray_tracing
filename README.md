# machine-learning-polarimetry
This repository contains codes that computes the Mueller matrix of an birrefrigent material using as inputs the geometrics parameters of the problem and the Parameters of the material via an ray tracing code (mcclain ray tracing).
## Overview
### Input
#### Geometric Parameters
- angle between the incident ray and the incident plane, the incident angle (a_i)
- normal vector of the incident plane (vnorm)
- the large of the crystal (thick)
#### Parameters of the Crystal
- Ordinary Refractive Index (go)
- Extraordinary Refractive Index (ge)
- Angle of the Optic Axis (a_c)
- Ordinary Gyrotropic Index (ge)
- Extraordinary Gyrotropic Index (go)
#### External Parameters
- refractive index of the outside media (for example: air no=ne=1)
- wavelength of the incident ray (lamda)
### Output
- Mueller matrix of the material.
- the diference of the Optical path lenght inside the crystal.
- The Brewster angle of the material.
