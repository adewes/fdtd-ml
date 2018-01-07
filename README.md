
This repository contains example scripts that show how to implement Yee's
finite difference time domain (FDTD) method for electromagnetic simulation
on a GPU using NVIDIA CUDA.

## Motivation

Today we use a variety of Physics-based measurement techniques to investigate
and characterize objects. For example, we use acoustic and
electromagnetic waves at a wide range of frequencies to characterize objects
and even generate images of them in 2D or 3D:

* RADAR systems use the reflection of electromagnetic or ultrasound waves
  by an object to measure the distance and speed of the object.
* MRT systems use static magnetic fields in combination with high-frequency
  electromagnetic waves to characterize the absorption properties of a slice
  of an object, reconstructing detailed 3D images of the object.
* Ultrasound systems use high-frequency ultrasonic waves to characterize the
  density of an object by measuring how it reflects emitted waves,
  producing a 2D or 3D image.

All of these measurement systems require us to solve so-called inverse problems,
where we need to estimate from a series of complex measurements the properties
of the object that could have generated these measurements. Traditionally,
signal processing and numerical optimization techniques are used to achieve
this.

With this project, I want to investigate how we might use machine learning to
help us solve such inverse problems. Machine learning (in particular deep
learning) seems to be well suited for such a task, as an inverse problem
requires us to find the inverse of a very complex transformation function that
often is cumbersome (or even impossible) to describe by hand but which is based
on a (largely) deterministic physical process that can be modeled and
understood. Having an automated way to learn object representations from input
signals would be a powerful tool that could allow us to implement systems that
are less demanding than currently used approaches for Physics-based imaging in
the sense that the number and accuracy of sensors as well as the modeling effort
could be drastically reduced.

To train a machine learning system on inverse problems, we need suitable
training data. There are two ways to generate such training data:

* Use a real-world system with an existing high-fidelity sensor that provides
  the ground truth of the objects under investigation, as well as a simpler
  sensor system that provides training data from which we want to reproduce
  the ground truth. For example, in our lab we use a Kinect sensor that produces
  a depth image of a scene which we can use as ground truth, and a series of
  ultrasound transducers that measure reflection of ultrasonic waves by the
  objects in the scene and whose data we can use to train our machine learning
  system.
* Use simulation techniques to generate synthetic data about a known scene, and
  use that synthetic data to train the machine learning system.


# CUDA-based Finite Difference Time Domain (FDTD) Simulation

Simulating electromagnetic (or acoustic) waves in a 2D or 3D environment allows
us to generate data that we can use to train machine learning systems and/or
design and optimize measurement devices.

## Basics

Yee's method uses two grids that are shifted by half a spatial step in either
direction to model electric and magnetic fields. Electric field values are
calculated in the center of each grid cell, whereas magnetic field values are
calculated on the edges of each cell. Using this trick, Yee's method avoids
field singularities that would otherwise plague our simulation e.g. on
boundaries between conducting and non-conducting grid cells.

## Implementation

We implement Yee's algorithm in two dimensions according to [1] (p. 397 ff.). 
Using two-dimensional arrays for E_z, H_x, H_y, J_z as well as \epsilon, \mu.
Whenever possible, we avoid transferring memory between the GPU and Python as
this is a time-consuming process. For example, the array J_z only contains the
amplitude of the currents and the corresponding oscillating values are
calculated on the GPU in each time step.

The resulting electric and magnetic fields can be exported after each time
step (or after a given number of steps) and visualized using e.g. Matplotlib.

## Perfectly Matched Layer (PML)

A simulation always needs to provide proper boundary conditions. Often, when
simulating electromagnetic waves we want to model an open, infinite space.
However, in a simulation we can only provide a fully reflective (perfect
conductor) boundary, or use periodic boundary conditions (i.e. x[n+1] = x[0],
y[n+1] = y[0]), that both do not provide a suitable simulation of an open space.

There are several ways to simulate such an open space, here we use a so-called
perfectly matched layer (PML), which is a layer that contains a non-physical
dispersive material with a perfectly matched impedance (hence the name) that
strongly attenuates waves entering the boundary region. If the attenuation
factor is sufficiently large, waves will be almost entirely attenuated when
traveling through the absorptive layer (to avoid reflections due to numerical
inaccuracies, we increase the absorptiveness of the layer gradually).

In our simulation, we use the split-field method to model the PML. Typically,
we choose a thickness of 20 cells and a normalized conductivity of 16 inside
the layer, which is sufficient to strongly attenuate incoming waves.

## Plotting

We use matplotlib to interactively visualize electromagnetic fields during the
simulation. As this slows down the framerate considerable due to the transfer
of data between the GPU and main memory as well as the processing by Python,
we avoid this in longer simulation runs though.

To generate movies from simulations we use Pillow to visualize the values of
the electromagnetic fields and store the resulting images as PNG, which we can
then process using e.g. `ffmpeg`:

    ffmpeg -r 60 -i "img-%*.png" -c:v libx264 -r 24 -pix_fmt yuv420p out.mp4

Example videos can be found on Youtube:

* Scattering of an electromagnetic wave on a dispersive sphere with a
  strong gradient in the permittivity: https://www.youtube.com/watch?v=lGz4L0yR4ug
* Scattering of an electromagnetic wave on a dispersive sphere (Mie scattering):
  https://www.youtube.com/watch?v=oBL5CJyHQjg
* Simulation of a parabolic mirror antenna: https://www.youtube.com/watch?v=ZPSzAaxkg5c
