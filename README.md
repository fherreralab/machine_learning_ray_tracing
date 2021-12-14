# Machine Learning Polarimetry

## Overview
## Polarimetry
This repository contains codes that computes the Mueller matrix of an birrefrigent material using as inputs the geometrics parameters of the problem and the Parameters of the material via an ray tracing code (mcclain ray tracing).
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
- Mueller matrix of the material. (M)
- the diference of the Optical path lenght inside the crystal. (OPD_)
- The Brewster angle of the material. (brews)
- <br />
## Machine Learning
Our goal is to invert the Ray tracing algorithm using as input the Mueller matrix of a material and other parameters obtainable by experimental measurements, and as outputs the parameters of the material, in this case the refractive index (<img src="https://latex.codecogs.com/svg.image?n_{e}" title="n_{e}" />,<img src="https://latex.codecogs.com/svg.image?n_{o}" title="n_{o}" />), to do so we gonna use Neural Networks.
### Neural Network Arquitecture
The arquitecture used for this algorthim is shown in the following figure:
![Arquitecture](https://github.com/fherreralab/machine_learning_ray_tracing/blob/main/NN_Architecture.PNG)
### Data
