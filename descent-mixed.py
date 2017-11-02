from dolfin import *
import numpy as np
import os
from tqdm import tqdm
from common import *
from time import time

# nbimporter has stopped working!
# This sucks...
#import nbimporter
#from descent_mixed import run_model as mixed_model


def run_model(init: str, qform: str, mesh_file: str, theta: float, mu: float = 1.0,
              e_stop_mult: float = 1e-7, max_steps: int = 1000, save_funs: bool = True, n=0):
    """
    """

    qform = qform.lower()

    msh = Mesh(mesh_file)

    t = tqdm(total=max_steps, desc='th=%7.2f' % theta, position=n, dynamic_ncols=True)

    def noop(*args, **kwargs):
        pass

    def tout(s, **kwargs):
        """ FIXME: Does not work as intended... """
        t.write(s, end='')

    debug = noop
    # debug = print

    # in plane displacements (IPD)
    UE = VectorElement("Lagrange", msh.ufl_cell(), 2, dim=2)
    # out of plane displacements (OPD)
    VE = FiniteElement("Lagrange", msh.ufl_cell(), 2)
    # Gradients of OPD
    ZE = VectorElement("Lagrange", msh.ufl_cell(), 2, dim=2)
    ME = MixedElement([UE, VE, ZE])
    W = FunctionSpace(msh, ME)

    # We gather in-plane and out-of-plane displacements into one
    # Function for visualization with ParaView.
    P = VectorFunctionSpace(msh, "Lagrange", 2, dim=3)
    fax = FunctionAssigner(P.sub(0), W.sub(0).sub(0))
    fay = FunctionAssigner(P.sub(1), W.sub(0).sub(1))
    faz = FunctionAssigner(P.sub(2), W.sub(1))

    disp = Function(P)
    disp.rename("disp", "displacement")

    fname_prefix = "%s-%s-%07.2f-%05.2f-" % (init, qform, theta, mu)
    dir = "output-mixed/" + fname_prefix.strip('-')
    try:
        os.mkdir(dir)
    except:
        pass

    file_name = dir + "/" + fname_prefix + ".pvd"
    file = File(file_name)  # .vtu files will have the same prefix

    w = Function(W)
    w_ = Function(W)
    u, v, z = w.split()
    u_, v_, z_ = w_.split()

    w_init = make_initial_data_mixed(init)
    w.interpolate(w_init)
    w_.interpolate(w_init)

    if qform == 'frobenius':
        Q2, L2 = frobenius_form()
    elif qform == 'isotropic':
        # Isotropic density for some material or other...
        E, nu = 1e9, 0.3
        Q2, L2 = isotropic_form(E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 + 2 * nu))
    else:
        raise Exception("Unknown quadratic form name '%s'" % qform)

    def eps(u):
        return (grad(u) + grad(u).T) / 2.0

    e_stop = msh.hmin() * e_stop_mult
    max_line_search_steps = 20
    step = 0
    omega = 0.25  # Gradient descent fudge factor in (0, 1/2)
    _hist = {'init': init, 'mu': mu, 'theta': theta, 'e_stop': e_stop,
             'J': [], 'alpha': [], 'du': [], 'dv': [], 'dz': [], 'constraint': [],
             'Q2': {'form_name': Q2.__name__, 'arguments': Q2.arguments},
             'symmetry': [], 'file_name': file_name}

    B = Identity(2)
    zero_energy = assemble((1. / 24) * inner(B, B) * dx(msh))

    def energy(u, v, z, mu=mu):
        J = (theta / 2) * Q2(eps(u) + outer(grad(v), grad(v)) / 2) * dx(msh) \
            + (1. / 24) * Q2(grad(z) - Id) * dx(msh) \
            + (1. / 2) * mu * inner(z - grad(v), z - grad(v)) * dx(msh)
        return assemble(J)

    # CAREFUL!! Picking the right scalar product here is essential
    # Recall the issues with boundary values: integrate partially
    # and only boundary terms survive...
    dtu, dtv, dtz = TrialFunctions(W)
    phi, psi, eta = TestFunctions(W)
    L = inner(dtu, phi) * dx + inner(grad(dtu), grad(phi)) * dx \
        + inner(dtv, psi) * dx + inner(grad(dtv), grad(psi)) * dx \
        + inner(dtz, eta) * dx + inner(grad(dtz), grad(eta)) * dx

    dw = Function(W)
    du, dv, dz = dw.split()

    # Output initial condition
    fax.assign(disp.sub(0), u.sub(0))
    fay.assign(disp.sub(1), u.sub(1))
    faz.assign(disp.sub(2), v)
    file << (disp, float(step))

    cur_energy = energy(u, v, z)
    alpha = ndu = ndv = ndz = 1.0

    debug("Solving with theta = %.2e, mu = %.2e, eps=%.2e for at most %d steps."
          % (theta, mu, e_stop, max_steps))

    # FIXME: check whether it makes sense to add ndz**2 here and below
    begin = time()
    while alpha * (ndu ** 2 + ndv ** 2 + ndz ** 2) > e_stop and step < max_steps:
        _constraint = assemble(inner(grad(v_) - z_, grad(v_) - z_) * dx)
        _symmetry = circular_symmetry(disp)
        _hist['constraint'].append(_constraint)
        _hist['symmetry'].append(_symmetry)
        debug("Step %d, energy = %.3e, |grad v - z| = %.3e, symmetry = %.3f"
              % (step, cur_energy, _constraint, _symmetry))

        #### Gradient
        # for some reason I'm not able to use derivative(J, w_, dtw)
        dJ = theta * L2(eps(u_) + outer(grad(v_), grad(v_)) / 2,
                        eps(phi) + sym(outer(grad(v_), grad(psi)))) * dx(msh) \
             + (1. / 12) * L2(grad(z_) - Id, grad(eta)) * dx(msh) \
             + mu * inner(grad(v_) - z_, grad(psi) - eta) * dx(msh)

        debug("\tSolving...", end='')
        solve(L == -dJ, dw, [])

        du, dv, dz = dw.split()
        # dw is never reassigned to a new object so it should be ok
        # to reuse du, dv without resplitting
        ndu = norm(du)
        ndv = norm(dv)
        ndz = norm(dz)

        debug(" done with |du| = %.3f, |dv| = %.3f, |dz| = %.3f" % (ndu, ndv, ndz))

        #### Line search
        debug("\tSearching... ", end='')
        while True:
            w = project(w_ + alpha * dw, W)
            u, v, z = w.split()
            new_energy = energy(u, v, z)
            if new_energy <= cur_energy - omega * alpha * (ndu ** 2 + ndv ** 2 + ndz ** 2):
                debug(" alpha = %.2e" % alpha)
                _hist['J'].append(cur_energy)
                _hist['alpha'].append(alpha)
                _hist['du'].append(ndu)
                _hist['dv'].append(ndv)
                _hist['dz'].append(ndz)
                cur_energy = new_energy
                alpha = min(1.0, 2.0 * alpha)  # Use a larger alpha for the next line search
                break
            if alpha < (1. / 2) ** max_line_search_steps:
                # If this happens, it's unlikely that we had computed an actual gradient
                raise Exception("Line search failed after %d steps" % max_line_search_steps)
            alpha /= 2.0  # Repeat with smaller alpha

        step += 1

        #### Write displacements to file
        debug("\tSaving... ", end='')
        fax.assign(disp.sub(0), u.sub(0))
        fay.assign(disp.sub(1), u.sub(1))
        faz.assign(disp.sub(2), v)
        file << (disp, float(step))
        debug("Done.")

        w_.vector()[:] = w.vector()
        u_, v_, z_ = w_.split()
        t.update()

    _hist['time'] = time() - begin

    if step < max_steps:
        t.total = step
        t.update()

    _hist['steps'] = step
    if save_funs:
        _hist['disp'] = disp
        _hist['u'] = u
        _hist['v'] = v
        _hist['z'] = z
        _hist['dtu'] = du
        _hist['dtv'] = dv
        _hist['dtz'] = dz
    debug("Done after %d steps" % step)

    t.close()
    return _hist


if __name__ == '__main__':

    from joblib import Parallel, delayed

    set_log_level(ERROR)

    parameters["form_compiler"]["optimize"] = True
    parameters["form_compiler"]["cpp_optimize"] = True

    results_file = "results-mixed-combined.pickle"
    mesh_file = generate_mesh('circle', 18, 18)
    theta_values = np.arange(20, 24, 1.0, dtype=float)

    # Careful: hyperthreading won't help (we are probably bound by memory channel bandwidth)
    n_jobs = min(2, len(theta_values))

    new_res = Parallel(n_jobs=n_jobs)(delayed(run_model)('ani_parab', 'isometric', mesh_file,
                                                         theta=theta, mu=10.0,
                                                         max_steps=20000, save_funs=False,
                                                         e_stop_mult=1e-9, n=n)
                                      for n, theta in enumerate(theta_values))

    save_results(new_res, results_file)
