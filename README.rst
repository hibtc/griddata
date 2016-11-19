pepperpot
=========

Simple CLI tools for analysis of pepperpot files:

- naive interpolation of the 4D probability density, given a file of particle
  4D phase space coordinates and intensities
- particle generation using a precomputed 4D probability matrix
- plotting particles or the probability distribution


Usage
~~~~~

Interpolation::

    python pepperpot.py interpol gauss emidata.ppp pdist_gauss.npy

Particle generation::

    python pepperpot.py generate pdist_gauss.npy particles_gauss.txt -n 500

Plotting the original data::

    python pepperpot.py plot point emidata.ppp scatter_emidata.pdf
    python pepperpot.py plot gauss emidata.ppp normsum_emidata.pdf

…and the generated particles for comparison::

    python pepperpot.py plot point particles_gauss.txt scatter_particles_gauss.pdf
    python pepperpot.py plot gauss particles_gauss.txt normsum_particles_gauss.pdf

…and the probability density::

    python pepperpot.py plot pdist pdist_gauss.npy pdist_gauss.pdf


For more info, see::

    python pepperpot.py -h
