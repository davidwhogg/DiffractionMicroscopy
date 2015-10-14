# PhaseRetrieval

A playground for looking at Fourier phase retrieval
(from an observation of the squared norm of a Fourier transform)
in microscopy applications.

## Author

- **David W. Hogg** (SCDA) (NYU) (MPIA)

## Acknowledgements

This project is heavily influenced by discussions with
- Charlie Epstein (Penn)
- Leslie Greengard (SCDA) (NYU)
- Jeremy Magland (Penn) (SCDA)

## License

**Copyright 2015 David W. Hogg**.
Any code in this repository is licensed for use and re-use
under the open-source **MIT License**.
See the file `LICENSE` for more details.

## Comments

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
