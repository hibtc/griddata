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

    pepperpot interpol gauss emidata.ppp pdist.npy

Particle generation::

    pepperpot generate pdist.npy particles.txt -n 500

Plot the probability density::

    pepperpot plot pdist pdist.npy pdist.pdf

…and the generated particles::

    pepperpot plot point particles.txt scatter_particles.pdf
    pepperpot plot gauss particles.txt normsum_particles.pdf

…and the original data for comparison::

    pepperpot plot point emidata.ppp scatter_emidata.pdf
    pepperpot plot gauss emidata.ppp normsum_emidata.pdf


For more info, see::

    pepperpot -h
