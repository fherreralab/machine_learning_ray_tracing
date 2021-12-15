# Machine Learning Polarimetry

## Overview
A general problem in the experimental Polarimetry is to obtain via this experimental data the properties of the material, this problem is tipycally approached by using Mueller matrix descomposition, but this method is convoluted since it depends of all the basics types of Mueller matrix that are known. So another approached to this problem is to have an algorithm thats go from the material properties to the polarimetry and then invert this problem using any tool that can typically invert physics problem, in this case we used Machine learning more specifically Neural Networks (NN). To do so we used the McClain algorithm[<img src="https://latex.codecogs.com/svg.image?^{1-2}" title="^{1-2}" />]


## Polarimetry


This repository contains codes that computes the Mueller matrix of an birrefrigent material using as inputs the geometrics parameters of the problem and the Parameters of the material via an ray tracing code (mcclain ray tracing).


### Input


#### Geometric Parameters


- angle between the incident ray and the incident plane, the incident angle (a_i)
- normal vector of the incident plane (vnorm)
- the large of the crystal (thick)


#### Parameters of the Crystal
- Ordinary Refractive Index (no)
- Extraordinary Refractive Index (ne)
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
We used 0.1 as the learning rate with a decay of 0.01 over 30 epochs and a batch size of 10, **elu** as the activation function, **Adam** as the optimization method and for the loss function we used the **Logcosh** function from *Keras*[<img src="https://latex.codecogs.com/svg.image?^{3}" title="^{3}" />]. 

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
- no=1.6589 [<img src="https://latex.codecogs.com/svg.image?^{4}" title="^{4}" />]
- ne = 1.5446 [<img src="https://latex.codecogs.com/svg.image?^{4}" title="^{4}" />]
- go=ge=0
- thick= 3E-3
- lamda= 853*10**-9
- vnorm=np.zeros((3,1))
- vnorm[2]=1 

Then use the algorithm as:

- all_polarization(no,ne,go,ge,thick,a_c,a_i,ni,lamda)

If the Brewster angle is required use the following line of command:
- Brewster(no,ne,a_c,go,ge,vnorm,ni,Ei)

For this function to work is needed an input polarization, in this case we use the **s** polarization as an input but the algorithm transform this polarization in **p** as the real polarization used in the algorithm.

This Algorithm is in the repository by the name of **ray_tracing_algorithm.py**


In the repository there is a folder that contain the data used to train the NN using 10.000 data as training data and 3.000 data for validation, there is a folder for tha Brewster angle option and the OPD option, both contain random data for <img src="https://latex.codecogs.com/svg.image?n_{o}" title="n_{o}" /> and <img src="https://latex.codecogs.com/svg.image?n_{e}" title="n_{e}" /> from 1.4 to 1.7, using the **random.uniform** function, this . This folders are *Training data Brewster* and *Training data OPD*.

Also in the repository there is a folder that contain the weight used for this problem, from 10.000 to 100 data for the option of Brewster angle and for OPD from 1.000.000 to 100 data. this due to the OPD data is more easy to obtain that the Brewster angle data.


# Requeriments
- [Numpy](https://numpy.org/) 
- [Keras](https://keras.io/)


# Reference
[1]Stephen C. McClain, Lloyd W. Hillman, and Russell A. Chipman, "Polarization ray tracing in anisotropic optically active media. I. Algorithms," J. Opt. Soc. Am. A 10, 2371-2382 (1993)

[2]Stephen C. McClain, Lloyd W. Hillman, and Russell A. Chipman, "Polarization ray tracing in anisotropic optically active media. II. Theory and physics," J. Opt. Soc. Am. A 10, 2383-2393 (1993)

[3] https://keras.io/api/

[4] G. Tamošauskas, G. Beresnevičius, D. Gadonas, A. Dubietis. Transmittance and phase matching of BBO crystal in the 3−5 μm range and its application for the characterization of mid-infrared laser pulses, Opt. Mater. Express 8, 1410-1418 (2018)
