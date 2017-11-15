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

### Reports

The script `report.py` implements a minimal web server to explore results in tabular
form and easily combine them into plots. Not *very* useful, but it was fun and quick
to do. The nice jQuery table is done with [FooTable](http://fooplugins.github.io/FooTable/),
the fixed header with [stickyTableHeaders](https://github.com/jmosbech/StickyTableHeaders).
I also used some js and css from [codepen](https://codepen.io).

To use it just run

```shell
python3 report.py
```

## Plotting with ParaView

After opening the `pvd` file, only two filters are necessary in the pipeline:
 1. **Calculator** to de-scale the displacements. Recall that in-plane
  displacements scale with the square of the plate's thickness $h$ whereas
  out-of-plane do linearly with it: $u_h = h^2 u, v_h = h v$
  Physically meaningful values like $h = 0.01$ mean that one can't see anything
  in the plot, so that in the end one needs e.g. $h = 0.3$. The expression for the
  output is: 
  ```
  (0.3^2*disp_X)*iHat + (0.3^2*disp_Y)*jHat + (0.3*disp_Z)*kHat
  ```
 2. **Warp by vector** to see the displacement.


## Contents

* `descent-curl.py`: Gradient descent for a modified functional with only first
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
* `report.py`

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
  altogether and implement some cool iPython widgets based on pandas dataframes
  (or even just use something like [qgrid](https://github.com/quantopian/qgrid)).
* Update licenses and acknowledgements to include all packages used.

## License

All my code is released under the GNU GPL v3. See the licenses of the included
software too.

## References

[1] de Benito Delgado, “Effective two-dimensional theories for
multilayered plates.”, 2018

[2] Bartels, “Numerical Solution of a Föppl-von Karman Model.”, 2016
