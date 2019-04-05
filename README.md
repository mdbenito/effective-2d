# lvk

Energy minimization schemes for the generalised von Kármán model in [1].

## Model and discretisation

[1] describes an effective two dimensional theory for multilayered
plates in the von Kármán regime. The resulting functional is minimised
here with vanilla gradient descent on mixed continuous Lagrange spaces
with a penalty formulation.

The model is characterised by the interplay of both membrane and
bending energies, with a parameter $\theta$ interpolating it from the
Kirchhoff to the linearised von Kármán regimes. We explore qualitative
aspects of the minimisers as a function of $\theta$, in particular the
breaking of symmetry at a critical value.

Gradient descent provides convergence to the expected minimisers
(proven to be global for small $\theta$ in [1])

We try to circumvent the use of Kirchhoff Discrete Triangles in the
method of [2] without success. Some progress in the implementation of
these elements for FEniCS is
[here](https://bitbucket.org/mdbenito/hermite). Note however that this
requires extensive changes and additions to FIAT and FFC, which are
available in [my]([https://bitbucket.org/mdbenito/fiat-fork)
[forks](https://bitbucket.org/mdbenito/ffc-fork) but are quite hacky
and not thoroughly tested.

Displacement fields at different steps during the minimisation are
stored as VTK files in folders. Additionally relevant metrics are 
logged to a MongoDB using [sacred](https://github.com/IDSIA/sacred).

There is (legacy) custom results explorer which only supports pickled
metrics, so we store those at well, but it should be trivial to
replace them by custom mongo queries.


## Dependencies and usage

This project requires [FEniCS 2017.1.0](https://fenicsproject.org) and
a few python libraries, but everything is packaged using [docker
compose](https://docs.docker.com/compose/). All you need is a working
[docker](https://docker.io) installation to build the image and start
all services.

To build and start everything, from the root of the project run:

```
cd docker && sudo docker-compose -p lvk up
```

After building, that command will start four services: MongoDB,
Omniboard, jupyter and the custom results server. The output and
source directories will be shared with some containers for
convenience. In particular, this allows local edition of the source
files which the notebook service sees.

**Password and token authentication have been disabled for all
services!**

* Juyter will be accessible at http://localhost:8888
* Omniboard will be accessible at http://localhost:9000
* The custom results server will be at http://localhost:8080

In order to run experiments from the console, you can open a console
into the notebooks container, which also mounts the code from your
local repository (the whole `src/` folder is shared). Assuming you
only have one instance of the whole setup running, run:

```
sudo docker exec -it lvk_notebooks_1 bash
cd src
python3 descent-curl.py help
```

Parameters can be changed in the command line using `with`. Parallel
execution of multiple experiments can be launched with `parallel` and
so on. Please read [sacred's
documentation](https://sacred.readthedocs.io/en/latest/quickstart.html)
for more on how to use the command line interface.


## Experiment tracking and reports

Experiments are stored in a MongoDB and can be individually browsed
using Omniboard (see above).

However, omniboard is not enough for the investigation of multiple
experiments jointly. For this a minimal results server displays them
in tabular form and allows to easily combine them into plots (the
"custom results server" mentioned above, available at
http://localhost:8080).

Using this browser, it is possible to plot the evolution of the method
in time for different runs side by side. Also, links to the ParaView
files are displayed and should open, after properly configuring the
system (e.g. adding mime types and handlers for xdg-open)

Additionally, the notebooks provide some basic code to query the
results database and create plots across the range of values of theta
tested in the experiments. See ...

**Disclaimer:** The report server is by far not the _definitive_
dashboard, but it was fun and quick to do.  The nice jQuery table in
the report server is done with
[FooTable](http://fooplugins.github.io/FooTable/), the fixed header
with
[stickyTableHeaders](https://github.com/jmosbech/StickyTableHeaders).
I also used some js and css from [codepen](https://codepen.io).


## Detailed contents

* `docker`: `docker-compose.yml` to build docker images and scripts to
  be installed in it.
* `src`: Source files. Inside you will find:
   * `descent_curl.py`: Gradient descent for a modified functional
     with only first order derivatives and a penalty term enforcing
     the condition that $z = \nabla v$. The fact that the penalty term
     can be left out provides some experimental evidence that
     minimisers of the new functional automatically fulfill a
     vanishing curl constraint.
   * `descent_mixed.py`: Mixed formulation for gradient descent
     (outdated, use the other model instead).
   * `explore.ipynb`: Notebook with some exploratory plots.
   * `von Karman.ipynb`: Implementation of the model in [2]. (Not
     working)
   * `von Karman mixed.ipynb`: Implementation of the model in [2]
     using a mixed model formulation. (Not working)
   * `report.py`: a server to explore results. See above.
 

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


## Things to do that ~~probably~~ won't happen

* Implement polling of the sacred db for jobs and queued execution
  ([see here](https://github.com/IDSIA/sacred/issues/215)))
* Make the reports more flexible. Possibly ditch that webserver nonsense
  altogether and implement some cool iPython widgets based on pandas dataframes
  (or even just use something like [qgrid](https://github.com/quantopian/qgrid)).
* Update licenses and acknowledgements to include all packages used.
* Fix dependency conflicts in requiremets.txt


## Known issues

* Sometimes, running `descent_*.py` with multiprocessing will result in dolfin's
  JIT compiler throwing strange errors. Typically, restarting execution will solve
  those, but sometimes the cache gets corrupted. The easiest workaround is to exit
  the container and start a new one. Because of these problems, it might pay off to
  do a first run with low max steps so as to be sure to have all forms precompiled
  for the real run.

## License

All my code is released under the GNU GPL v3. See the licenses of the included
software too.


## References

[1] de Benito Delgado, *“Effective two-dimensional theories for
multilayered plates.”*, 2019

[2] Bartels, *“Numerical Solution of a Föppl-von Karman Model.”*, 2016
