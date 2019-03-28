# lvk

Energy minimization schemes for the generalised von Kármán model in [1].

## Model and discretisation

[1] describes an effective two dimensional theory for multilayered plates in
the von Kármán regime. The resulting functional is minimised here with vanilla
gradient descent on mixed continuous Lagrange spaces with a penalty formulation.

The model is characterised by the interplay of both membrane and bending
energies, with a parameter $\theta$ interpolating it from the Kirchhoff to the
linearised von Kármán regimes. We explore qualitative aspects of the
minimisers as a function of $\theta$, in particular the breaking of symmetry
at a critical value.

Gradient descent provides convergence to the expected minimisers (proven to be
global for small $\theta$ in [1])

We try to circumvent the use of Kirchhoff Discrete Triangles in the method 
of [2] without success. Some progress in the implementation of these
elements for FEniCS is [here](https://bitbucket.org/mdbenito/hermite). Note
however that this requires extensive changes and additions to FIAT and FFC, which
are available in [my]([https://bitbucket.org/mdbenito/fiat-fork) 
[forks](https://bitbucket.org/mdbenito/ffc-fork) but are quite hacky and not
thoroughly tested.

Results of the computations are stored both as VTK files in folders and as
pickled objects containing relevant quantities gathered during the
computations. It is ugly and should probably be replaced by some database like
sqlite or some document store / nosql thingy. 

## Requirements

All you need is a working docker installation to build the image and start
the container. All dependencies are contained in the image. To build it,
from the root of the project run:

```
docker build -f docker/Dockerfile -t lvk:latest .
```

## Usage

To start a notebook server run (this will share the output and source
directories with the container, so they persist after exiting it):

```shell
docker run -v $(pwd)/output:/home/fenics/lvk/output \
           -v $(pwd)/src:/home/fenics/lvk/src \
           -p 8888:8888 \
           --rm -it --name lvk lvk fenics-notebook
```
The jupyter server will be accessible at http://localhost:8888
Password and token authentication have been disabled for the server!

Alternatively, you can run `sudo docker/start-notebook`, which basically
does the above.

### Reports

The script `src/report.py` implements a minimal web server to explore results
in tabular form and easily combine them into plots. It is not the _definitive_
dashboard, but it was fun and quick to do. The nice jQuery table is done with
 [FooTable](http://fooplugins.github.io/FooTable/),
the fixed header with
 [stickyTableHeaders](https://github.com/jmosbech/StickyTableHeaders).
I also used some js and css from [codepen](https://codepen.io).

It is possible to plot the evolution of the method in time for different runs
side by side. Also, links to the ParaView files are displayed and should open, after
properly configuring the system (e.g. adding mime types and handlers for xdg-open)


To start the results server, run:

```shell
docker run -v $(pwd)/output:/home/fenics/lvk/output \
           -v $(pwd)/src:/home/fenics/lvk/src \
           -p 8080:8080 \
           --rm -it --name lvk-report \
           lvk "report-server 0.0.0.0 8080"
```

Then go to http://localhost:8080

Alternatively, you can run `sudo docker/start-report`, which basically
does the above.


## Detailed contents

* `docker`: `Dockerfile` to build an image and scripts to be installed in it.
* `src`: Source files. Inside you will find:
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
   * `report.py`: a server to explore results. See "Reports" below.
 
  

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


## Dependencies

* A working installation of FEniCS 2017.1.0 with Python3 support.
 [This docker container]() contains all of it plus a few extra goodies.
* [tqdm](https://github.com/tqdm/tqdm) for the silly progress bars.
 I really shouldn't have made that necessary.


## Things to do that ~~probably~~ won't happen

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

[1] de Benito Delgado, *“Effective two-dimensional theories for
multilayered plates.”*, 2019

[2] Bartels, *“Numerical Solution of a Föppl-von Karman Model.”*, 2016
