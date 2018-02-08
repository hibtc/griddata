pepperpot
=========

Simple CLI tools for analysis of pepperpot files:

- naive interpolation of the 4D probability density, given a file of particle
  4D phase space coordinates and intensities
- particle generation using a precomputed 4D probability matrix
- plotting particles or the probability distribution


Setup
~~~~~

Install in development mode::

    python setup.py develop


Usage
~~~~~

Interpolation::

    pepperpot interpol gauss emidata.ppp pdist_gauss.npy

Particle generation::

    pepperpot generate pdist_gauss.npy particles_gauss.txt -n 500

Plotting the original data::

    pepperpot plot point emidata.ppp scatter_emidata.pdf
    pepperpot plot gauss emidata.ppp normsum_emidata.pdf

…and the generated particles for comparison::

    pepperpot plot point particles_gauss.txt scatter_particles_gauss.pdf
    pepperpot plot gauss particles_gauss.txt normsum_particles_gauss.pdf

…and the probability density::

    pepperpot plot pdist pdist_gauss.npy pdist_gauss.pdf


For more info, see::

    pepperpot -h
