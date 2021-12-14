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
## Machine Learning
Our goal is to invert the Ray tracing algorithm using as input the Mueller matrix of a material and other parameters obtainable by experimental measurements, and as outputs the parameters of the material, in this case the refractive index (<img src="https://latex.codecogs.com/svg.image?n_{e}" title="n_{e}" />,<img src="https://latex.codecogs.com/svg.image?n_{o}" title="n_{o}" />), to do so we gonna use **Feed-Forward Neural Networks**.
### Neural Network Arquitecture and Training
We used 0.1 as the learning rate with a decay of 0.01 over 30 epochs and a batch size of 10, **elu** as the activation function, **Adam** as the optimization method and for the loss function we used the **Logcosh** function from *Keras*[<img src="https://latex.codecogs.com/svg.image?^{1}" title="^{1}" />]. 

The Arquitecture used for this algorithm is the following:


![Arquitecture](https://github.com/fherreralab/machine_learning_ray_tracing/blob/main/NN_Architecture.PNG)
### Data
For this problem we used the data from the mcclain ray tracing algorithm, that was mention previously.
As input we used the ouputs of the ray tracing algorithm and as outputs the refractive index of the material(<img src="https://latex.codecogs.com/svg.image?n_{e}" title="n_{e}" />,<img src="https://latex.codecogs.com/svg.image?n_{o}" title="n_{o}" />), fixing the other parameteres.
## Example
As an example we gonna used the BBO crystal, and the following parameters:
- ni=1
- a_c=29.2
- a_i=0
- no=1.6589 [<img src="https://latex.codecogs.com/svg.image?^{1}" title="^{2}" />]
- ne = 1.5446 [<img src="https://latex.codecogs.com/svg.image?^{1}" title="^{2}" />]
- go=ge=0
- thick= 3E-3
- lamda= 853*10**-9
- vnorm=np.zeros((3,1))
- vnorm[2]=1 



# Reference
[1] https://keras.io/api/
[2] G. Tamošauskas, G. Beresnevičius, D. Gadonas, A. Dubietis. Transmittance and phase matching of BBO crystal in the 3−5 μm range and its application for the characterization of mid-infrared laser pulses, Opt. Mater. Express 8, 1410-1418 (2018)
