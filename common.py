from dolfin import *
import numpy as np
import matplotlib.pyplot as pl
import os
import pickle as pk
from plots import plots1
import mshr

def make_initial_data_mixed(which: str, degree=2) -> Expression:
    initial_data = {'zero': lambda x: [0.0, 0.0, 0.0, 0.0, 0.0],
                    'parab': lambda x: [0.0, 0.0, 0.5*x[0]**2 + 0.5*x[1]**2, x[0], x[1]],
                    'ani_parab': lambda x: [0.0, 0.0, 0.15*x[0]**2 + 0.15*x[1]**2, 0.3*x[0], x[1]]}

    class InitialDisplacements(Expression):
        """ The first two components of this expression correspond
        to in-plane-displacements, the last two to the *gradient* of
        out-of-plane displacements.
        """
        def eval(self, values, x):
            vals = initial_data[which](x)
            values[0] = vals[0]
            values[1] = vals[1]
            values[2] = vals[2]
            values[3] = vals[3]
            values[4] = vals[4]

        def value_shape(self):
            return (5,)
    return InitialDisplacements(degree=degree)


def make_initial_data_penalty(which: str, degree=2) -> Expression:
    initial_data = {'zero': lambda x: [0.0, 0.0, 0.0, 0.0],
                    'rot': lambda x: [0.0, 0.0, -x[1], x[0]],
                    'parab': lambda x: [0.0, 0.0, x[0], x[1]],
                    'ani_parab': lambda x: [0.0, 0.0, 0.3 * x[0], x[1]],
                    'iso_compression': lambda x: [-0.2 * x[0], -0.2 * x[1], 0.0, 0.0],
                    'ani_compression': lambda x: [-0.4 * x[0], -0.1 * x[1], 0.0, 0.0],
                    'bartels': lambda x: [0.0, -x[1] / 10.0,
                                          x[0] * (1 - x[0]) * (1 - 2 * x[0]) * sin(4 * DOLFIN_PI * x[1]),
                                          2 * DOLFIN_PI * x[0] ** 2 * (1 - x[0]) ** 2 * cos(4 * DOLFIN_PI * x[1])]}

    class InitialDisplacements(Expression):
        """ The first two components of this expression correspond
        to in-plane-displacements, the last two to the *gradient* of
        out-of-plane displacements.
        """

        def eval(self, values, x):
            vals = initial_data[which](x)
            values[0] = vals[0]
            values[1] = vals[1]
            values[2] = vals[2]
            values[3] = vals[3]

        def value_shape(self):
            return (4,)

    return InitialDisplacements(degree=degree)


def circular_symmetry(disp: Function) -> float:
    """ Computes the quotient of the lenghts of the principal axes of an ellipse
    This assumes that the domain before the deformation is a circle."""
    if circular_symmetry.pts is None:
        cc = disp.function_space().mesh().coordinates()
        xmin = cc[:,0].argmin()
        xmax = cc[:,0].argmax()
        ymin = cc[:,1].argmin()
        ymax = cc[:,1].argmax()
        circular_symmetry.pts = [Point(cc[xmax]), Point(cc[ymax]), Point(cc[xmin]), Point(cc[ymin])]
    newpts = [p.array()+disp(p) for p in circular_symmetry.pts]
    a = np.linalg.norm(newpts[0] - newpts[2])
    b = np.linalg.norm(newpts[1] - newpts[3])
    return a / b

circular_symmetry.pts = None


def compute_potential(z: Function, V: FunctionSpace) -> Function:
    """ Takes a gradient and computes its potential (up to a constant)

    We solve the linear problem:
     Find $(p,q) \in W = V \times Z_{\Gamma_D}$ such that for all $(\phi,
     \psi) \in W$:

      $$(\nabla p - q, \nabla \phi)_{L^2}- (\nabla p - q, \psi)_{L^2} +
      (q, \psi)_{L^2} = (z, \psi)_{L^2} $$

    Note that we would need to set Dirichlet conditions on the
    potential to fix the constant.

    Arguments
    ---------
        z: gradient
        V: space for the potential
    Returns
    -------
        Function $v \in V$ such that $\nabla v = z$
    """
    msh = z.function_space().mesh()

    # Construct a mixed function space for the potential and gradient
    PE = V.ufl_element()
    QE = z.function_space().ufl_element()
    W = FunctionSpace(msh, PE * QE)

    # Essential boundary conditions will only be set for the
    # subspace of gradients
    class GradientsBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary

    class ValuesBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return False

    bcP = DirichletBC(W.sub(0), Constant(42.0), ValuesBoundary())  # void...
    bcQ = DirichletBC(W.sub(1), z, GradientsBoundary())
    p, q = TrialFunctions(W)
    phi, psi = TestFunctions(W)
    a = inner(grad(p) - q, grad(phi)) * dx - inner(grad(p) - q, psi) * dx + inner(q, psi) * dx
    L = inner(z, psi) * dx
    w = Function(W)
    solve(a == L, w, [bcP, bcQ])

    ret = Function(V)
    fa = FunctionAssigner(V, W.sub(0))
    fa.assign(ret, w.sub(0))
    ret.rename("pot", "potential")
    return ret


def test_potential(fun_exp: str, grad_exp: (str, str), eps: float = 1e-4) -> bool:
    """ Usage: test_potential("x[0]*x[0] + x[1]*x[1]", ("2*x[0]", "2*x[1]"))
    Caveats:
        * compute_potential() is not very accurate: the tolerance parameter
          here cannot be set too small (1e-4)
        * the expressions used cannot have any poles in the square [1,2]x[1,2]
    """
    print("Testing potential (with integration constant hack)... ", end='')
    msh = RectangleMesh(Point(1, 1), Point(2, 2), 10, 10, "crossed")
    V = FunctionSpace(msh, "Lagrange", 3)
    u = interpolate(Expression(fun_exp, element=V.ufl_element()), V)
    Z = VectorFunctionSpace(msh, "Lagrange", 2, dim=2)
    z = interpolate(Expression(grad_exp, element=Z.ufl_element()), Z)
    p = compute_potential(z, V)

    # HACK: fix the integration constant
    hack = norm(project(u - p, V))
    p = project(p + Constant(hack), V)
    test1 = project(u - p, V)
    test2 = project(z - grad(p), Z)
    print("OK." if norm(test1) < eps and norm(test2) < eps else "FAILED.")

# test_potential("x[0]*x[0] + x[1]*x[1]", ("2*x[0]", "2*x[1]"))
# test_potential("x[0]/x[1] + x[0]*x[1]", ("1/x[1]+x[1]", "-x[0]/(x[1]*x[1])+x[0]"))
# test_potential("0.5*x[0]*x[0]*(1-x[0])*(1-x[0])*sin(4*DOLFIN_PI*x[1])",
#                ("x[0]*(1-x[0])*(1-2*x[0])*sin(4*DOLFIN_PI*x[1])",
#                 "2*DOLFIN_PI*x[0]*x[0]*(1-x[0])*(1-x[0])*cos(4*DOLFIN_PI*x[1])"))


def name_run(r:dict) -> str:
    """ Returns a 'unique' string identifier for the given run.
    Should be enough :/
    """
    return "%s_%07.2f_%3.1f_%.2e_%d" % (r['init'], r['theta'], r['mu'], r['e_stop'], r['steps'])


def save_results(results: list, results_file: str):

    # Load any previous results
    if os.path.isfile(results_file):
        with open(results_file, "rb") as fd:
            old_results = pk.load(fd)
    else:
        old_results = {}

    # Merge new results in and prepare plots for summary view (TODO)
    new_results = {name_run(r): r for r in results}
    for k, r in new_results.items():
        r['plots1_fname'] = "output/plots1-" + k + ".eps"
        plots1(r)
        pl.savefig(r['plots1_fname'])
        pl.close()

    old_results.update(new_results)

    with open(results_file, "wb") as f:
        pk.dump(old_results, f)


def generate_mesh(kind:str, m:int, n:int) -> str:
    """ (Cached) Generates a mesh, saves it to a file and returns the file name
     """
    mesh_file = 'mesh-%dx%d-%s.xml.gz' % (m, n, kind)

    if not os.path.isfile(mesh_file):
        if kind.lower() == 'circle':
            domain = mshr.Circle(Point(0.0, 0.0), 1, m)
            msh = mshr.generate_mesh(domain, 18)
        elif kind.lower == 'rectangle':
            msh = RectangleMesh(Point(-1, -1), Point(1, 1), m, n)
        else:
            raise TypeError('Unhandled mesh type "%s"' % kind)
        File(mesh_file) << msh

    return mesh_file