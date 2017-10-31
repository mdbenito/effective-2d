# coding: utf-8

# # Problem definition
# 
# We wish to minimize
# $$ I(u,v) = \frac{\theta}{2} \int_{\omega} |\nabla_s u + \tfrac{1}{2} \nabla v \otimes \nabla v|^{2} \mathrm{d}x
#    + \frac{1}{24} \int_{\omega} |\nabla^2 v - \mathrm{Id}|^{2} \mathrm{d}x. $$
# 
# Because we only have $C^0$ elements we set $z$ for $\nabla v$ and minimize instead
# 
# $$ J(u,z) = \frac{\theta}{2} \int_{\omega} |\nabla_s u + \tfrac{1}{2} z \otimes z|^{2} \mathrm{d}x 
#           + \frac{1}{24} \int_{\omega} |\nabla z - \mathrm{Id}|^{2} \mathrm{d}x 
#           + \mu \int_{\omega} |\mathrm{curl}\ z|^{2} \mathrm{d}x, $$
# 
# then recover the vertical displacements (up to a constant) by minimizing
# 
# $$ F(p,q) = \tfrac{1}{2} || \nabla p - q ||^2 + \tfrac{1}{2} || q - z ||^2. $$
# 
# This we do by solving the linear problem $D F = 0$.
# 
# Minimization of the energy functional $J$ is done via gradient descent and a line search. In plane displacements
# and gradients of out of plane displacements form a mixed function space $U \times Z$. We also have another scalar
# space $V$ where the potential of the out of plane gradients lives. The model is defined and solved in `run_model()`
# below. Experiments can be easily run in parallel with `joblib`.

from dolfin import *
import numpy as np
import os
from tqdm import tqdm
from common import make_initial_data_penalty, circular_symmetry, compute_potential, save_results


def run_model(init: str, theta: float, mu: float = 0.0,
              e_stop_mult: float = 1e-5, max_steps: int = 400, fname_prefix: str = "descent-curl",
              save_funs: bool = True, n=0):
    """
    
    """
    t = tqdm(total=max_steps, desc='th=% 7.2f' % theta, position=n, dynamic_ncols=True)

    # debug = print
    def noop(*args, **kwargs):
        pass

    def tout(s, **kwargs):
        """ FIXME: Does not work as intended... """
        t.write(s, end='')

    debug = noop

    # in plane displacements (IPD)
    UE = VectorElement("Lagrange", msh.ufl_cell(), 2, dim=2)
    # Gradients of out of plane displacements (OPD)
    VE = VectorElement("Lagrange", msh.ufl_cell(), 2, dim=2)
    W = FunctionSpace(msh, UE * VE)
    # will store out of plane displacements
    V = FunctionSpace(msh, "Lagrange", 2)

    # class DirichletBoundary(SubDomain):
    #    def inside(self, x, on_boundary):
    #        return False
    # bcU = DirichletBC(W.sub(0), Constant((0.0, 0.0)), DirichletBoundary())
    # bcV = DirichletBC(W.sub(1), Constant((0.0, 0.0)), DirichletBoundary())

    # We gather in-plane and out-of-plane displacements into one
    # Function for visualization with ParaView.
    P = VectorFunctionSpace(msh, "Lagrange", 2, dim=3)
    fax = FunctionAssigner(P.sub(0), W.sub(0).sub(0))
    fay = FunctionAssigner(P.sub(1), W.sub(0).sub(1))
    faz = FunctionAssigner(P.sub(2), V)

    disp = Function(P)
    disp.rename("disp", "displacement")

    dir = "output/" + fname_prefix.strip('-')
    try:
        os.mkdir(dir)
    except:
        pass

    file = File(dir + "/" + fname_prefix + ".pvd")  # .vtu files will have the same prefix

    w = Function(W)
    w_ = Function(W)
    u, v = w.split()
    u_, v_ = w_.split()

    w_init = make_initial_data_penalty(init)
    w.interpolate(w_init)
    w_.interpolate(w_init)

    def eps(u):
        return (grad(u) + grad(u).T) / 2.0

    e_stop = msh.hmin() * e_stop_mult
    max_line_search_steps = 20
    step = 0
    omega = 0.25  # Gradient descent fudge factor in (0, 1/2)
    _hist = {'init': init, 'mu': mu, 'theta': theta, 'e_stop': e_stop,
             'J': [], 'alpha': [], 'du': [], 'dv': [], 'curl': [],
             'symmetry': []}

    Id = Identity(2)
    zero_energy = assemble((1. / 24) * inner(Id, Id) * dx(msh))

    def energy(u, v, mu=mu):
        J = (theta / 2) * inner(eps(u) + outer(v, v) / 2, eps(u) + outer(v, v) / 2) * dx(msh) + (1. / 24) * inner(
            grad(v) - Id, grad(v) - Id) * dx(msh) + mu * inner(curl(v), curl(v)) * dx(msh)
        return assemble(J)

    phi, psi = TestFunctions(W)
    # CAREFUL!! Picking the right scalar product here is essential
    # Recall the issues with boundary values: integrate partially and only boundary terms survive...
    # dtu, dtv = TrialFunctions(W)
    # L = inner(dtu, phi)*dx + inner(dtv, psi)*dx \
    #    + inner(grad(dtu), grad(phi))*dx + inner(grad(dtv), grad(psi))*dx
    # The previous lines are equivalent to:
    dtw = TrialFunction(W)
    z = TestFunction(W)
    L = inner(dtw, z) * dx + inner(grad(dtw), grad(z)) * dx

    dw = Function(W)
    du, dv = dw.split()

    # Output initial condition
    opd = compute_potential(v, V)
    fax.assign(disp.sub(0), u.sub(0))
    fay.assign(disp.sub(1), u.sub(1))
    faz.assign(disp.sub(2), opd)
    file << (disp, float(step))

    cur_energy = energy(u, v)
    alpha = ndu = ndv = 1.0

    debug("Solving with theta = %.2e, mu = %.2e, eps=%.2e for at most %d steps."
          % (theta, mu, e_stop, max_steps))
    begin = time.time()
    while alpha * (ndu ** 2 + ndv ** 2) > e_stop and step < max_steps:
        _curl = assemble(curl(v_) * dx)
        _symmetry = circular_symmetry(disp)
        _hist['curl'].append(_curl)
        _hist['symmetry'].append(_symmetry)
        debug("Step %d, energy = %.3e, curl = %.3e, symmetry = %.3f"
              % (step, cur_energy, _curl, _symmetry))

        #### Gradient
        dJ = theta * inner(eps(u_) + outer(v_, v_) / 2, eps(phi)) * dx(msh) \
             + theta * inner(eps(u_) + outer(v_, v_) / 2, outer(v_, psi)) * dx(msh) \
             + (1. / 12) * inner(grad(v_) - Id, grad(psi)) * dx(msh) \
             + 2 * mu * inner(curl(v_), curl(psi)) * dx(msh)

        debug("\tSolving...", end='')
        solve(L == -dJ, dw, [])

        du, dv = dw.split()
        # dw is never reassigned to a new object so it's ok
        # to reuse du, dv without resplitting
        ndu = norm(du)
        ndv = norm(dv)

        debug(" done with |du| = %.3f, |dv| = %.3f" % (ndu, ndv))

        #### Line search
        debug("\tSearching... ", end='')
        while True:
            w = project(w_ + alpha * dw, W)
            u, v = w.split()
            new_energy = energy(u, v)
            if new_energy <= cur_energy - omega * alpha * (ndu ** 2 + ndv ** 2):
                debug(" alpha = %.2e" % alpha)
                _hist['J'].append(cur_energy)
                _hist['alpha'].append(alpha)
                _hist['du'].append(ndu)
                _hist['dv'].append(ndv)
                cur_energy = new_energy
                alpha = min(1.0, 2.0 * alpha)  # Use a larger alpha for the next line search
                break
            if alpha < (1. / 2) ** max_line_search_steps:
                raise Exception("Line search failed after %d steps" % max_line_search_steps)
            alpha /= 2.0  # Repeat with smaller alpha

        step += 1

        #### Write displacements to file
        debug("\tSaving... ", end='')
        opd = compute_potential(v, V)
        fax.assign(disp.sub(0), u.sub(0))
        fay.assign(disp.sub(1), u.sub(1))
        faz.assign(disp.sub(2), opd)
        file << (disp, float(step))
        debug("Done.")

        w_.vector()[:] = w.vector()
        u_, v_ = w_.split()
        t.update()

    _hist['time'] = time.time() - begin

    if step < max_steps:
        t.total = step
        t.update()

    _hist['steps'] = step
    if save_funs:
        _hist['disp'] = disp
        _hist['u'] = u
        _hist['v'] = v
        _hist['dtu'] = du
        _hist['dtv'] = dv
    debug("Done after %d steps" % step)

    #t.close()
    return _hist


if __name__ == "__main__":

    from joblib import Parallel, delayed
    import mshr

    parameters["form_compiler"]["optimize"] = True
    parameters["form_compiler"]["cpp_optimize"] = True

    set_log_level(ERROR)

    domain = mshr.Circle(Point(0.0, 0.0), 1, 18)
    msh = mshr.generate_mesh(domain, 18)

    theta_values = np.arange(8.7, 8.8, 0.1, dtype=float)

    # Careful: hyperthreading won't help (we are probably bound by memory channel bandwidth)
    n_jobs = min(2, len(theta_values))

    new_res = Parallel(n_jobs=n_jobs)(delayed(run_model)('ani_parab', theta=theta, mu=0.0,
                                                         fname_prefix='ani-parab-%07.2f-' % theta,
                                                         max_steps=10000, save_funs=False,
                                                         e_stop_mult=1e-8, n=n)
                                      for n, theta in enumerate(theta_values))

    save_results(new_res, "results-combined.pickle")
