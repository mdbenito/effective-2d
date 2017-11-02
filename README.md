# lvk

Energy minimization schemes for the generalised von Kármán model in [1].

## Model and discretisation

[1] describes an effective two dimensional theory for multilayered plates in
the von Kármán regime. The resulting functional is minimised here with vanilla
gradient descent on $P_2$ spaces with penalty formulations.

The model is characterised by the interplay of both membrane and bending
energy, with a parameter $\theta$ interpolating it from the Kirchhoff to the
linearised von Kármán regimes. We explore qualitative aspects of the
minimisers as a function of $\theta$, in particular the breaking of symmetry
at a critical value.

Gradient descent provides convergence to the expected minimisers (proven to be
global for small $\theta$ in [1])

We try to circumvent the use of Kirchhoff Discrete Triangles in the method 
of [2] without success. Some progress in the implementation of these
elements for FEniCS is [here](https://bitbucket.org/mdbenito/hermite). Note
however that this requires extensive changes and additions to FIAT and FFC, which
are available in my forks but are quite hacky and not thoroughly tested.

Results of the computations are stored both as VTK files in folders and as
pickled objects containing relevant quantities gathered during the
computations. It is ugly and should probably be replaced by some database like
sqlite or some document store / nosql thingy.

## Contents

* `descent.py`: Gradient descent for a modified functional with only first
  order derivatives and a penalty term enforcing the condition that 
  $z = \nabla v$. The fact that the penalty term can be left out provides
  some experimental evidence that minimisers of the new functional automatically
  fulfill a vanishing curl constraint.
* `descent-curl.ipynb`: Notebook accompanying `descent-curl.py` with some
  exploratory plots.
* `descent-mixed.py`: Mixed formulation for gradient descent.
* `descent-mixed.ipynb`: Notebook accompanying `descent-mixed.py` with some
  exploratory plots.
* `von Karman.ipynb`: Implementation of the model in [2]. (Not working)
* `von Karman mixed.ipynb`: Implementation of the model in [2] using a
  mixed model formulation. (Not working)
* `report.py`: A minimal web server to explore results in tabular form and
  easily combine them into plots. Not *very* useful, but it was fun and quick
  to do.

## Dependencies

* A working installation of FEniCS 2017.1.0. [This docker container]()
 contains all of it plus a few extra goodies.
* [tqdm](https://github.com/tqdm/tqdm) for the silly progress bars.
 I really shouldn't have made that necessary.

## To do

* Fix the issues with nbimporter and leave the models in the notebooks,
  instead of having copies (yuk!) in python scripts for parallel runs.
* Store results in a sensible database and update it while simulations are
  still running.
* Be more systematic with "unique" identifiers for runs (crappy and fragile now).
* Make the reports more flexible. Possibly ditch that webserver nonsense
  altogether and implement some cool iPython widgets (or even just use something
  like [qgrid](https://github.com/quantopian/qgrid)).


## License

All code released under the GNU GPL v3.

## References

[1] de Benito Delgado, “Effective two-dimensional theories for
multilayered plates.”, 2018

[2] Bartels, “Numerical Solution of a Föppl-von Karman Model.”, 2016
