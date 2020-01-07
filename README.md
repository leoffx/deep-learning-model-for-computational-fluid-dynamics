# Deep Learning model for Computational Fluid Dynamics

## Prerequisites

### Setup TensorFlow

This code was created and tested using TensorFlow 2.0 GPU on Google Colab. The installation can be done by running the command `!pip install tensorflow-gpu==2.0.0` before the imports.


### Generate Dataset

The dataset can be generated using the lattice_boltzmann_method.ipynb notebook. In this example we generate a simple 2D dataset with 128x128 resolution.

By completely running the notebook it will be generated two files, one that stores the simulations' Lattice Boltzmann distributions, and one that stores the objects placement info. The files will have 40 simulations in it, with 100 frames each.

Depending on the avaiable RAM, they can be split on multiple files by tweaking the `files_number` value. Others parameters like the simulation resolution and the number of examples created can be changed easily too, by adjusting, respectively, `simulation_size ` and `examples_number`.
