[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14201547.svg)](https://doi.org/10.5281/zenodo.14201547)
# Machine Learning Polarimetry

## Overview
A general problem in experimental Polarimetry is to obtain via this experimental data the properties of the material, this problem is typically approached by using Mueller matrix decomposition, but this method is convoluted since it depends on all the basic types of Mueller matrix that are known. So another approach to this problem is to have an algorithm that goes from the material properties to the polarimetry and then invert this problem using any tool that can typically invert physics problems, in this case, we used Machine learning more specifically Neural Networks (NN). To do so we used the **McClain algorithm**[<img src="https://latex.codecogs.com/svg.image?^{1-2}" title="^{1-2}" />] and **Feed-Forward Neural Networks**.


## Polarimetry: ray_tracing_algorithm.py


This python file computes the Mueller matrix of an birrefrigent material using as inputs the geometrics parameters of the problem and the Parameters of the material via an ray tracing code (McClain ray tracing).


### Inputs


#### Geometric Parameters


- Angle between the incident ray and the incident plane, the incident angle (a_i).
- Normal vector of the incident plane (vnorm).
- The large of the crystal (thick).


#### Parameters of the Crystal
- Ordinary Refractive Index (no).
- Extraordinary Refractive Index (ne).
- Angle of the Optic Axis (a_c).
- Ordinary Gyrotropic Index (ge).
- Extraordinary Gyrotropic Index (go).


#### External Parameters
- refractive index of the outside media (for example: air no=ne=1).
- wavelength of the incident ray (lamda).


### Output
- Mueller matrix of the material (M).
- the diference of the Optical path lenght inside the crystal or Optical Path Difference (OPD).
- The Brewster angle of the material (brews).


## Neural Network Arquitecture and Training Parameters
We used 0.1 as the learning rate with a decay of 0.01 over 30 epochs and a batch size of 10, **elu** as the activation function, **Adam** as the optimization method and for the loss function we used the **Logcosh** function from *Keras*[<img src="https://latex.codecogs.com/svg.image?^{3}" title="^{3}" />]. 

The Arquitecture used for this algorithm is the following:


![Arquitecture](https://github.com/fherreralab/machine_learning_ray_tracing/blob/main/NN_Architecture.PNG)

### Training Dataset
For this problem, we used the data from McClain ray tracing algorithm, which was mentioned previously.
As input, we used the outputs of the ray tracing algorithm and as outputs the refractive index of the material(<img src="https://latex.codecogs.com/svg.image?n_{e}" title="n_{e}" />,<img src="https://latex.codecogs.com/svg.image?n_{o}" title="n_{o}" />)and fixing the other parameters that are used as input of the ray tracing algorithm. This leads to an input of 17 as the shape, 16 for the Mueller matrix (4x4 matrix) and 1 from the OPD (This data has been adjusted by a parameter of <img src="https://latex.codecogs.com/svg.image?10^{6}" title="10^{6}" /> in the code) or the Brewster angle.

#### Folders:
Training data MM_Brewster contains the training data for the Mueller matrix plus Brewster angle data, where input_Brewster.npy corresponds to the input data on which the neural network was trained, output_Brewster.npy the output data on which it was trained, input_val_Brewster.npy the input data on which it was validated, and output_val_Brewster.npy the output data on which it was validated. 

Training data MM_OPD, similar to the brewster angle case, contains the training data for the Mueller matrix plus OPD, where input_OPD.npy corresponds to the input data on which the neural network was trained, output_OPD.npy the output data on which it was trained, input_val_OPD.npy the input data on which it was validated, and output_val_OPD.npy the output data on which it was validated. 

weights_MM_Brewster and weights_MM_OPD contain the weights for the neural network described above. Each file has the name model_#.h5, where ‘#’ corresponds to the number of data with which the network was trained, for Mueller Matrix more Brewster or OPD, respectively.

## Automatic differentiation
We used a JAX [<img src="https://latex.codecogs.com/svg.image?^{5}" title="^{5}" />] code of the ray tracing algorithm to automatically differentiate it, we use 200 different starting points for each sample, with experimental refractive index values, obtained from https://refractiveindex.info/, with a value added to the refractive index of 0.1 max for each iteration. 

To use the automatic differencing algorithm just run the file main_JAX.py with the file ray_tracing_JAX.py in the same folder, in the main_JAX.py file you can change the parameters of the ray tracing algorithm and the parameters of the crystals. 


## Case Use Example:

As an example we use the BBO crystal, and the following parameters:
- ni = 1
- a_c = 29.2
- a_i = 0
- no = 1.6589 [<img src="https://latex.codecogs.com/svg.image?^{4}" title="^{4}" />]
- ne = 1.5446 [<img src="https://latex.codecogs.com/svg.image?^{4}" title="^{4}" />]
- go = ge =0
- thick = 3E-3
- lamda = 853*10**-9
- vnorm = np.zeros((3,1)) and vnorm[2] = 1

Then use the algorithm as:

- M,OPD = all_polarization(no,ne,go,ge,thick,a_c,a_i,ni,lamda)

Where M is the Mueller matrix of the material for that especific set of inputs and OPD is the Optical path difference between the ordinary and the extraordinary ray incide the crystal.

If the Brewster angle is required use the following line of command:
- brews = Brewster(no,ne,a_c,go,ge,vnorm,ni,Ei)

Where brews is the Brewster angle calculated in this case for this setup between 50 and 90 from the reflection index, this can be changed in the line 529 in the code.


For this function to work is needed an input polarization, in this case we use the **s** polarization as an input but the algorithm transform this polarization in **p** as the real polarization used in the algorithm.

This Algorithm is in the repository by the name of **ray_tracing_algorithm.py**.


IIn the repository there is a folder that contains the data used to train the NN using 10.000 data as training data and 3.000 data for validation, there is a folder for the Brewster angle option and the OPD option, both contain random data for <img src="https://latex.codecogs.com/svg.image?n_{o}" title="n_{o}" /> and <img src="https://latex.codecogs.com/svg.image?n_{e}" title="n_{e}" /> from 1.4 to 1.7, using the **random.uniform** function. These folders are *Training data MM_Brewster* and *Training data MM_OPD*.

Also, in the repository, there is a folder that contains the weight used for this problem: from 10.000 to 100 data for the option of Brewster angle and for OPD from 1.000.000 to 100 data. This is because the OPD data is more easy to obtain than the Brewster angle data.

### Results
Using the weight of the 10.000 data for OPD and Brewster angle we have the following NN predictions :

For OPD:
- no = 1,6153855
- ne = 1,48315930

Giving us an percentage relative error of 2,62% for the <img src="https://latex.codecogs.com/svg.image?n_{o}" title="n_{o}" /> and a 3,98% for the <img src="https://latex.codecogs.com/svg.image?n_{e}" title="n_{e}" />. For Brewster we have:

- no = 1,65347790
- ne = 1,53919400

Giving us an percentage relative error of 0,33% for the <img src="https://latex.codecogs.com/svg.image?n_{o}" title="n_{o}" /> and a 0,35% for the <img src="https://latex.codecogs.com/svg.image?n_{e}" title="n_{e}" />.

# Library Requeriments
- [Numpy](https://numpy.org/) 
- [Keras](https://keras.io/)
- [JAX](https://jax.readthedocs.io/en/latest/)


# References
[1]Stephen C. McClain, Lloyd W. Hillman, and Russell A. Chipman, "Polarization ray tracing in anisotropic optically active media. I. Algorithms," J. Opt. Soc. Am. A 10, 2371-2382 (1993)

[2]Stephen C. McClain, Lloyd W. Hillman, and Russell A. Chipman, "Polarization ray tracing in anisotropic optically active media. II. Theory and physics," J. Opt. Soc. Am. A 10, 2383-2393 (1993)

[3] https://keras.io/api/

[4] G. Tamošauskas, G. Beresnevičius, D. Gadonas, A. Dubietis. Transmittance and phase matching of BBO crystal in the 3−5 μm range and its application for the characterization of mid-infrared laser pulses, Opt. Mater. Express 8, 1410-1418 (2018)

[5] https://jax.readthedocs.io/en/latest/user_guides.html


