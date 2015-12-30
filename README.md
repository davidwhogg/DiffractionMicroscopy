# DiffractionMicroscopy
Inferring an object from it's diffraction illumination pattern.

## Authors
- David W. Hogg (NYU CCPP) (NYU CDS) (Simons SCDA)

## License
Copyright 2015 David W. Hogg.
**DifractionMicroscopy** is open-source software with an **MIT License**.
See the file `LICENSE` for details.

## Acknowledgements
This project is heavily influenced by discussions with
- Charlie Epstein (Penn)
- Leslie Greengard (SCDA) (NYU)
- Jeremy Magland (SCDA)

This project depends on a great stack of *open-source software:*
- Python3
- numpy, scipy, and matplotlib
- TeX and LaTeX

## Projects
- Infer the real-space or full complex FT from a (noisy, censored)
measurement of the squared modulus only, plus perhaps some
onstraints (like non-negativity or compact support).
- Consider cases of Gaussian and Poisson noise.
- Consider cases where we get (noisy, censored) two-dimensional
Ewald-sphere slices of (the squared modulus of) the Fourier
transform, and we don't know the Euler angles for any of the slices!

## Comments
- This repository is currently just a stub.
- Let's go fully Bayesian if we can.
- The problem of inferring the phase of a Fourier transform
(of, say, an image) from an observation of the squared-norm
of that Fourier transform comes up in problems of
**x-ray diffraction** and **diffraction microscopy**.
In principle it also might arise in **adaptive optics**.
- The problem is analogous to (or identical to, in some sense)
inferring a function from an **observation of its auto-correlation**.
- We will concentrate on methods that involve
either explicit **optimization of a justified objective function**
or else some kind of probabilistic inference.
