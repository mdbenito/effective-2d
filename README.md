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
stored as VTK files in folders. Additional relevant metrics are logged
to a MongoDB using [sacred](https://github.com/IDSIA/sacred) and are
visualised using
[Omniboard](https://github.com/vivekratnavel/omniboard).

There is (legacy) custom results explorer which only supports pickled
metrics, so we store those at well, but it should be trivial to
replace them by custom mongo queries.


## Usage

This project requires [FEniCS 2017.1.0](https://fenicsproject.org) and
a few python libraries, but everything is packaged using [docker
compose](https://docs.docker.com/compose/). All you need is a working
[docker](https://docker.io) installation to build the image and start
all services.

To build and start everything, from the root of the project run:

```
cd docker && sudo docker-compose -p lvk up
```

After building, the first command will start five services: MongoDB,
Omniboard, JupyterLab, the custom results server and a container to
run scripts.

**Password and token authentication have been disabled for all
services!**

* JupyterLab will be accessible at http://localhost:8888
* Omniboard will be accessible at http://localhost:9000
* The custom results server will be at http://localhost:8080
* Access the mongo console with `docker exec -it lvk_mongo_1 mongo`
* Open a shell to run scripts with `docker exec -it -u fenics
  lvk_compute_1 bash`

The last command will open a shell into one of the containers, with
the output and source directories shared for convenience in
`/home/fenics/lvk`. In particular, this allows local edition of the
source files which the notebook service sees. You can also open a
terminal inside JupyterLab.

The main script is `src/descent_curl.py`. Remember that the code is
mounted from your local repository: the whole `src/` folder is shared
as `/home/fenics/lvk/src`. To see command line options, from that
location, run

```
python3 descent-curl.py help
```

Parameters can be changed in the command line using `with`. Parallel
execution of multiple experiments can be launched with `parallel` and
so on. Please read [sacred's
documentation](https://sacred.readthedocs.io/en/latest/quickstart.html)
for more on how to use the command line interface.

In order to tear down all containers:
```
cd docker && sudo docker-compose -p lvk down
```
This leaves the experiment database as a docker volume in the system,
as well as the VTK files for ParaView under `output/`.

## Experiment tracking and reports

Experiments are stored in a (persistent) MongoDB and can be
individually browsed using Omniboard (see above).

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
tested in the experiments. See `src/explore.ipynb`.

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

Displacement fields are stored as collections of `vtu` files, indexed
by a `pvd` file. In paraview, after opening one of the latter, only
two filters are necessary in the visualization pipeline in order to
see the output:
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

* Config script:
  - generate random user and password
  - build the mongo and jupyterlab containers configured with those
* Implement polling of the sacred db for jobs and queued execution
  ([see here](https://github.com/IDSIA/sacred/issues/215)))
* Make the reports more flexible. Possibly ditch that webserver
  nonsense altogether and implement some cool iPython widgets based on
  pandas dataframes (or even just use something like
  [qgrid](https://github.com/quantopian/qgrid)).
* Update licenses and acknowledgements to include all packages used.

## Known issues

* Running `descent_*.py` with multiprocessing will often result in
  dolfin's JIT compiler throwing strange I/O errors and jobs failing.
  This seems to be caused by multiple jobs writing to the cache and
  relaunching will typically solve the issue since forms will be read
  from disk. For this reason, there is a "dry run" mode which does not
  create any db entry nor files but does the precompilation of all
  required forms. Just run `python3 descent_curl.py parallel with
  dry_run=True` and wait until actual computation starts, then kill
  the job and restart without the `dry_run` parameter. However,
  sometimes the cache gets corrupted and it must be deleted (it is in
  `~/.cache/dijitso` and `~/.cache/instant`).
* `docker-compose` can fail to start the network under Windows systems
  with messages like `driver failed programming external connectivity
  on endpoint` or `network not found`. If this happens, try restarting
  the docker daemon. Also remember to tear down all services using
  `docker-compose down`.
* If working with the WSL under Windows, mounting folders with the
  linux client and the windows docker daemon cannot work, so one has
  to use `docker-compose.exe` from a windows console and inside the
  regular windows file system.
* Omniboard has a tendency to run out of memory, both the server and
  the browser. The server is given 4GB of ram in `docker-compose.yml`,
  and it's easy to adjust. As to the browser app, the current
  (v.1.4.0) workaround is to filter results based on experiment name,
  id, etc. so as to never have more than a few hundred. Additionally,
  reducing the frequency at which metrics are stored might help. See
  [this issue](https://github.com/vivekratnavel/omniboard/issues/87)
  for more information.

## License

All my code is released under the GNU GPL v3. See the licenses of the
included software too.


## References

[1] de Benito Delgado, *“Effective two-dimensional theories for
multilayered plates.”*, 2019

[2] Bartels, *“Numerical Solution of a Föppl-von Karman Model.”*, 2016
