Basic Idea:
-----------
We have a bunch of very-low photon images of the same object, each taken from a many different viewing angles, but we don't know those angles. We want to infer the 3D structure of the original object (a molecule in this case) from many of these extremely low-count images.

Isotropic case:
---------------

Here is a small sample of the test dataset of very low photon count images (average 16 photons each).

Example low-photon images:
<img src="./datum_000000_None.png" width=200 height=150>
<img src="./datum_000001_None.png" width=200 height=150>
<img src="./datum_000002_None.png" width=200 height=150>
<!-- ![png](./datum_000003_None.png)
![png](./datum_000004_None.png)
![png](./datum_000005_None.png)
![png](./datum_000006_None.png)
![png](./datum_000007_None.png)
![png](./datum_000008_None.png)
![png](./datum_000009_None.png)
![png](./datum_000010_None.png)
![png](./datum_000011_None.png)
![png](./datum_000012_None.png)
![png](./datum_000013_None.png)
![png](./datum_000014_None.png)
![png](./datum_000015_None.png)
![png](./datum_000016_None.png)
![png](./datum_000017_None.png)
![png](./datum_000018_None.png)
![png](./datum_000019_None.png) -->

Here is a video of the stochastic gradient inferring the 3D structure from 16,384 such images.

Video of fitting process:
https://youtu.be/RkA5-lhMlLI
<iframe width="560" height="315" src="https://www.youtube.com/embed/pqTb3y8Agx4" frameborder="0" allowfullscreen></iframe>

Here is an image of the true 3D structure that generated the low-photon images – note NOT a mixture of gaussians like the model used for inference.

True data these were generated from:
![png](./truth_3d.png)

Anisotropic case:
-----------------

We also tried with an awful anisotropic distribution of angles. The fitting still works surprisingly well (video)!


<img src="./datum_000000_dipole.png" width=200 height=150>
<img src="./datum_000001_dipole.png" width=200 height=150>
<img src="./datum_000002_dipole.png" width=200 height=150>
<!-- ![png](./datum_000003_dipole.png)
![png](./datum_000004_dipole.png)
![png](./datum_000005_dipole.png)
![png](./datum_000006_dipole.png)
![png](./datum_000007_dipole.png)
![png](./datum_000008_dipole.png)
![png](./datum_000009_dipole.png)
![png](./datum_000010_dipole.png)
![png](./datum_000011_dipole.png)
![png](./datum_000012_dipole.png)
![png](./datum_000013_dipole.png)
![png](./datum_000014_dipole.png)
![png](./datum_000015_dipole.png)
![png](./datum_000016_dipole.png)
![png](./datum_000017_dipole.png)
![png](./datum_000018_dipole.png)
![png](./datum_000019_dipole.png) -->

Fitting:
https://youtu.be/pqTb3y8Agx4
<iframe width="560" height="315" src="https://www.youtube.com/embed/RkA5-lhMlLI" frameborder="0" allowfullscreen></iframe>

True data:
![png](./truth_3d_dipole.png)
Angle distribution
![png](./aniso.png)