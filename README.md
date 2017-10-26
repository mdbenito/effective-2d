# lvk

Energy minimization schemes for the generalised von Kármán model in [1].
Currently, vanilla gradient descent on $P_2$ spaces converges to the
expected minimum, while an attempt at circumventing the use of Kirhhoff
Discrete Triangles in the method of [2] does not.


## Contents

* `descent-curl.ipynb`: Gradient descent for a modified functional
  with only first order derivatives. Experimental evidence that
  minimisers fulfil a vanishing curl constraint on the gradient of the
  vertical displacements.
* `von Karman.ipynb`: Implementation of the model in [2]. (Not working)
* `von Karman mixed.ipynb`: Implementation of the model in [2] using a
  mixed model formulation. (Not working)

## Dependencies

All that is needed is a working installation of FEniCS 2017.1.0.
[This docker container]() contains all of it plus a few extra goodies.

## License

GNU GPL v3.

## References

[1] de Benito Delgado, “Effective two-dimensional theories for
multilayered plates.”
[2] Bartels, “Numerical Solution of a Foppl-von Karman Model.”
